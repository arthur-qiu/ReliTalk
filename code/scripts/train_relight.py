"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import torchvision
import random

import utils.general as utils
import utils.plots as plt
import model.resnet_network as ResNet
import model.unet_network as UNet
from utils.light_util import add_SHlight, normal_shading, normalize

import wandb
import pygit2
from functools import partial
print = partial(print, flush=True)

class TrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = self.conf.get_int('train.batch_size')
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(project=kwargs['wandb_workspace'], name=self.subject + '_' + self.methodname, config=self.conf)

        self.optimize_inputs = False

        self.optimize_inputs = self.optimize_latent_code or self.optimize_expression or self.optimize_pose
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['is_continue']:
            if kwargs['load_path'] != '':
                load_path = kwargs['load_path']
            else:
                load_path = self.train_dir
            if os.path.exists(load_path):
                is_continue = True
            else:
                is_continue = False
        else:
            is_continue = False

        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "LightModelParameters"
        self.specmodel_params_subdir = "SpecModelParameters"
        self.nmmodel_params_subdir = "NormalModelParameters"
        self.optimizer_params_subdir = "LightOptimizerParameters"
        self.scheduler_params_subdir = "LightSchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.specmodel_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.nmmodel_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.train_dir, 'runconf.conf')))

        with open(os.path.join(self.train_dir, 'runconf.conf'), 'a+') as f:
            f.write(str(pygit2.Repository('.').head.shorthand) + '\n'  # branch
                    + str(pygit2.Repository('.').head.target) + '\n') # commit hash)


        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          sample_size=-1,
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_semantics=self.conf.get_bool('loss.gt_w_seg'),
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         sample_size=-1,
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_semantics=self.conf.get_bool('loss.gt_w_seg'),
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )

        self.albedo_net = ResNet.Albedo_ResnetGenerator()
        self.normal_net = ResNet.ResnetGenerator(input_nc=6)
        self.spec_net = UNet.UnetGenerator(output_nc=1, input_nc=6)
        # self.normal_net = UNet.UnetGenerator()

        ResNet.init_net(self.albedo_net)
        ResNet.init_net(self.normal_net)
        UNet.init_net(self.spec_net)
        # UNet.init_net(self.normal_net)
        if torch.cuda.is_available():
            self.albedo_net.cuda()
            self.spec_net.cuda()
            self.normal_net.cuda()
            self.albedo_net.constant_factor = self.albedo_net.constant_factor.cuda()
            self.albedo_net.sample_lights = self.albedo_net.sample_lights.cuda()
            self.light_var = torch.cuda.FloatTensor([0.000e+00,
                            0.000e+00, 
                            0.999e+00,
                            0.000e+00,
                            0.000e+00, 
                            0.000e+00,
                            0.000e+00,
                            0.000e+00,
                            0.000e+00])
        else:
            self.light_var = torch.FloatTensor([0.000e+00,
                            0.000e+00, 
                            0.999e+00,
                            0.000e+00,
                            0.000e+00, 
                            0.000e+00,
                            0.000e+00,
                            0.000e+00,
                            0.000e+00])
        self.light_var.requires_grad = True            

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')

        self.optimizer = torch.optim.Adam([{'params': self.albedo_net.parameters(), 'lr': self.lr}, {'params': self.spec_net.parameters(), 'lr': self.lr}, {'params': self.normal_net.parameters(), 'lr': self.lr}, {'params': self.light_var, 'lr': self.lr * 0.1}])
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        if self.optimize_inputs:
            num_training_frames = len(self.train_dataset)
            param = []
            if self.optimize_latent_code:
                self.latent_codes = torch.nn.Embedding(num_training_frames, 32, sparse=True).cuda()
                torch.nn.init.uniform_(
                    self.latent_codes.weight.data,
                    0.0,
                    1.0,
                )
                param += list(self.latent_codes.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3], sparse=True).cuda()
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(load_path, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'LightModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.albedo_net.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'LightOptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.optimize_inputs:

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_inputs_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])
                except:
                    print("input and camera optimizer parameter group doesn't match")
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    if self.optimize_expression:
                        self.expression.load_state_dict(data["expression_state_dict"])
                    if self.optimize_pose:
                        self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                        self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                except:
                    print("expression or pose parameter group doesn't match")
                if self.optimize_latent_code:
                    self.latent_codes.load_state_dict(data["latent_codes_state_dict"])

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.plot_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.0)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
            self.loss.lbs_weight = 0.

    def save_checkpoints(self, epoch, only_latest=False):
        if not only_latest:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.albedo_net.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "model_state_dict": self.spec_net.state_dict()},
                os.path.join(self.checkpoints_path, self.specmodel_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "model_state_dict": self.normal_net.state_dict()},
                os.path.join(self.checkpoints_path, self.nmmodel_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))

        torch.save(
            {"epoch": epoch, "model_state_dict": self.albedo_net.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.spec_net.state_dict()},
            os.path.join(self.checkpoints_path, self.specmodel_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.normal_net.state_dict()},
            os.path.join(self.checkpoints_path, self.nmmodel_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.optimize_inputs:
            if not only_latest:
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))\

            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_latent_code:
                dict_to_save["latent_codes_state_dict"] = self.latent_codes.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))

    def run(self):
        acc_loss = {}

        sample_index = list(range(0, 256*256, 64))
        s = 8
        pi = torch.acos(torch.zeros(1)).item() * 2
        gaussian_blur = torchvision.transforms.GaussianBlur(5)

        for epoch in range(self.start_epoch, self.nepochs + 1):
            # For geometry network annealing frequency band
            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
            if len(self.GT_lbs_milestones) > 0 and epoch >= self.GT_lbs_milestones[-1]:
                self.loss.lbs_weight = 0.

            if epoch % 5 == 0 or (epoch < self.nepochs - 5):
                light = torch.clamp(self.light_var.clone().detach(), -2.0, 2.0)
                light[5:] = light[5:] / 10
                self.albedo_net.light = nn.parameter.Parameter(light)
                self.save_checkpoints(epoch)
            else:
                light = torch.clamp(self.light_var.clone().detach(), -2.0, 2.0)
                light[5:] = light[5:] / 10
                self.albedo_net.light = nn.parameter.Parameter(light)
                self.save_checkpoints(epoch, only_latest=True)

            if (epoch % self.plot_freq == 0):
                self.albedo_net.eval()
                self.spec_net.eval()
                self.normal_net.eval()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.eval()
                    if self.optimize_latent_code:
                        self.latent_codes.eval()
                    if self.optimize_pose:
                        self.flame_pose.eval()
                        self.camera_pose.eval()
                eval_iterator = iter(self.plot_dataloader)
                for img_index in range(len(self.plot_dataset)):
                    start_time = time.time()
                    if img_index >= self.conf.get_int('plot.plot_nimgs'):
                        break
                    indices, model_input, ground_truth = next(eval_iterator)

                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.cuda()
                        except:
                            model_input[k] = v

                    for k, v in ground_truth.items():
                        try:
                            ground_truth[k] = v.cuda()
                        except:
                            ground_truth[k] = v

                    skin_mask = ground_truth["semantics"][:,:,0].view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)
                    face_mask = torch.sum(ground_truth["semantics"][:,:,:-2], 2).view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)
                    shoulder_mask = ground_truth["semantics"][:,:,-2].view(-1, 1, 256, 256)
                    foreground_mask = torch.sum(ground_truth["semantics"][:,:,:-1], 2).view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)

                    light = torch.clamp(self.light_var, -2.0, 2.0)
                    light[5:] = light[5:] / 10
                    space_normal, space_shading = normal_shading(light)
                    albedo = self.albedo_net(ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256))
                    normal = ground_truth['normal'].transpose(2, 1).view(-1, 3, 256, 256)
                    fine_normal = (self.normal_net(torch.cat([ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256), normal], 1)) + 1) / 2 * face_mask * 2 - 1
                    img_light = torch.cuda.FloatTensor(space_shading/255)

                    length = space_normal.shape[0]
                    space_shading = space_shading.reshape(length, -1) / 255
                    space_shading = space_shading[sample_index, ...]
                    space_normal = space_normal[sample_index, ...]
                    h = torch.cuda.FloatTensor(normalize(space_normal + np.array([[0, 0, 1]]).repeat(space_shading.shape[0], 0))).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                    nh = fine_normal[:, [0] , :, :] * h[:, :, 0, :, :] + fine_normal[:, [1] , :, :] * h[:, :, 1, :, :] + fine_normal[:, [2] , :, :] * h[:, :, 2, :, :]
                    z = torch.cuda.FloatTensor(space_shading[np.newaxis, :, np.newaxis]) * nh
                    sep_spec = (s + 2) / (2 * pi) * torch.pow(z, s)
                    spec = torch.sum(sep_spec, 1, keepdim=True)

                    masked_spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec)) * face_mask
                    masked_spec = (masked_spec - torch.min(masked_spec)) / (torch.max(masked_spec) - torch.min(masked_spec))
                    shading = add_SHlight(self.albedo_net.constant_factor, fine_normal, light.view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
                    masked_shading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading)) * face_mask
                    masked_shading_nonzero = masked_shading[masked_shading.nonzero(as_tuple=True)]
                    masked_shading_min = torch.min(masked_shading_nonzero)
                    masked_shading_max = torch.max(masked_shading_nonzero)
                    masked_shading = (masked_shading - masked_shading_min) / (masked_shading_max - masked_shading_min)
                    specmap = self.spec_net(torch.cat([ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256), fine_normal.detach()], 1))
                    scaled_specmap = (specmap + 1) / 2 * 0.2
                    final_image = (torch.clamp(albedo * (masked_shading + masked_spec * scaled_specmap), 0.0, 1.0) * (1 - shoulder_mask) + shoulder_mask * (ground_truth["rgb"].transpose(2, 1).reshape(-1, 3, 256, 256) / 2 + 0.5)) * 2 - 1

                    sample_light_index = random.randint(0,15)
                    sample_light = self.albedo_net.sample_lights[sample_light_index]

                    sample_space_normal, sample_space_shading = normal_shading(sample_light)
                    sample_img_light = torch.cuda.FloatTensor(sample_space_shading/255*2-1).view(-1, 1, 256, 256)

                    sample_space_shading = sample_space_shading.reshape(length, -1) / 255
                    sample_space_shading = sample_space_shading[sample_index, ...]
                    sample_space_normal = sample_space_normal[sample_index, ...]
                    sample_h = torch.cuda.FloatTensor(normalize(sample_space_normal + np.array([[0, 0, 1]]).repeat(sample_space_shading.shape[0], 0))).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                    sample_nh = fine_normal[:, [0] , :, :] * sample_h[:, :, 0, :, :] + fine_normal[:, [1] , :, :] * sample_h[:, :, 1, :, :] + fine_normal[:, [2] , :, :] * sample_h[:, :, 2, :, :]
                    sample_z = torch.cuda.FloatTensor(sample_space_shading[np.newaxis, :, np.newaxis]) * sample_nh
                    sample_sep_spec = (s + 2) / (2 * pi) * torch.pow(sample_z, s)
                    sample_spec = torch.sum(sample_sep_spec, 1, keepdim=True)

                    sample_masked_spec = (sample_spec - torch.min(sample_spec)) / (torch.max(sample_spec) - torch.min(sample_spec)) * face_mask
                    sample_masked_spec = (sample_masked_spec - torch.min(sample_masked_spec)) / (torch.max(sample_masked_spec) - torch.min(sample_masked_spec))
        
                    sample_shading = add_SHlight(self.albedo_net.constant_factor, fine_normal, sample_light.view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
                    sample_masked_shading = (sample_shading - torch.min(sample_shading)) / (torch.max(sample_shading) - torch.min(sample_shading)) * face_mask
                    sample_masked_shading_nonzero = sample_masked_shading[sample_masked_shading.nonzero(as_tuple=True)]
                    sample_masked_shading_min = torch.min(sample_masked_shading_nonzero)
                    sample_masked_shading_max = torch.max(sample_masked_shading_nonzero)
                    sample_masked_shading = torch.clamp((sample_masked_shading - sample_masked_shading_min) / (sample_masked_shading_max - sample_masked_shading_min), 0.0, 1.0)                

                    sample_final_image = (torch.clamp(albedo.detach() * (sample_masked_shading + sample_masked_spec * scaled_specmap.detach()), 0.0, 1.0) * (1 - shoulder_mask) + shoulder_mask * (ground_truth["rgb"].transpose(2, 1).reshape(-1, 3, 256, 256) / 2 + 0.5) * model_input['object_mask'].view(-1, 1, 256, 256)
                    + (1 - model_input['object_mask'].view(-1, 1, 256, 256).float())) * 2 - 1

                    model_outputs = {
                        'rgb_values': final_image.view(3, 256*256).transpose(1, 0),
                        'spec_values': masked_spec.view(1, 256*256).repeat(3, 1).transpose(1, 0),
                        'specmap_values': specmap.view(1, 256*256).repeat(3, 1).transpose(1, 0),
                        'normal_values': ground_truth['normal'].view(-1, 3),
                        'fine_normal_values': fine_normal.view(-1, 3, 256*256).transpose(2, 1).view(-1, 3),
                        'albedo_values': albedo.view(3, 256*256).transpose(1, 0),
                        'shading_values': masked_shading.view(1, 256*256).repeat(3, 1).transpose(1, 0),
                        'object_mask': model_input["object_mask"].reshape(-1).unsqueeze(1).float(),
                        'light_values': img_light.view(256*256, 1).repeat(1, 3),
                        'face_mask': face_mask.view(-1, 1),
                        'foreground_mask': foreground_mask.view(-1, 1),
                        'sample_light_values': (sample_img_light/2+0.5).view(1, 256*256).repeat(3, 1).transpose(1, 0),
                        'sample_rgb_values': sample_final_image.view(3, 256*256).transpose(1, 0),
                    }

                    if self.optimize_inputs:
                        if self.optimize_latent_code:
                            # use the latent code from the first scripts frame
                            model_input['latent_code'] = self.latent_codes(torch.LongTensor([0]).cuda()).squeeze(1).detach()

                    batch_size = ground_truth['rgb'].shape[0]
                    plot_dir = os.path.join(self.eval_dir, model_input['sub_dir'][0], 'epoch_'+str(epoch))
                    img_name = model_input['img_name'][0,0].cpu().numpy()
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                    print("Saving image {} into {}".format(img_name, plot_dir))
                    plt.light_plot(img_name,
                             model_outputs,
                             model_input['cam_pose'],
                             ground_truth,
                             plot_dir,
                             epoch,
                             self.img_res,
                             is_eval=False,
                             **self.plot_conf
                             )
                    print("Plot time per image: {}".format(time.time() - start_time))
                    print(light)
                    del model_outputs, ground_truth

                self.albedo_net.train()
                self.spec_net.train()
                self.normal_net.train()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.train()
                    if self.optimize_latent_code:
                        self.latent_codes.train()
                    if self.optimize_pose:
                        self.flame_pose.train()
                        self.camera_pose.train()
            start_time = time.time()

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                skin_mask = ground_truth["semantics"][:,:,0].view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)
                face_mask = torch.sum(ground_truth["semantics"][:,:,:-2], 2).view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)
                shoulder_mask = ground_truth["semantics"][:,:,-2].view(-1, 1, 256, 256)

                light = torch.clamp(self.light_var, -2.0, 2.0)
                light[5:] = light[5:] / 10
                space_normal, space_shading = normal_shading(light)
                albedo = self.albedo_net(ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256))
                normal = ground_truth['normal'].transpose(2, 1).view(-1, 3, 256, 256)
                fine_normal = (self.normal_net(torch.cat([ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256), normal], 1)) + 1) / 2 * face_mask * 2 - 1

                length = space_normal.shape[0]
                space_shading = space_shading.reshape(length, -1) / 255
                space_shading = space_shading[sample_index, ...]
                space_normal = space_normal[sample_index, ...]
                h = torch.cuda.FloatTensor(normalize(space_normal + np.array([[0, 0, 1]]).repeat(space_shading.shape[0], 0))).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                nh = fine_normal[:, [0] , :, :] * h[:, :, 0, :, :] + fine_normal[:, [1] , :, :] * h[:, :, 1, :, :] + fine_normal[:, [2] , :, :] * h[:, :, 2, :, :]
                z = torch.cuda.FloatTensor(space_shading[np.newaxis, :, np.newaxis]) * nh
                sep_spec = (s + 2) / (2 * pi) * torch.pow(z, s)
                spec = torch.sum(sep_spec, 1, keepdim=True)

                masked_spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec)) * face_mask
                masked_spec = (masked_spec - torch.min(masked_spec)) / (torch.max(masked_spec) - torch.min(masked_spec))
     
                if epoch < (self.nepochs // 3):
                    shading = add_SHlight(self.albedo_net.constant_factor, fine_normal, light.detach().view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
                else:
                    shading = add_SHlight(self.albedo_net.constant_factor, fine_normal, light.view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
                masked_shading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading)) * face_mask
                masked_shading_nonzero = masked_shading[masked_shading.nonzero(as_tuple=True)]
                masked_shading_min = torch.min(masked_shading_nonzero)
                masked_shading_max = torch.max(masked_shading_nonzero)
                masked_shading = (masked_shading - masked_shading_min) / (masked_shading_max - masked_shading_min)                

                specmap = self.spec_net(torch.cat([ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256), fine_normal.detach()], 1))
                scaled_specmap = (specmap + 1) / 2 * 0.2
                final_image = (torch.clamp(albedo * (masked_shading + masked_spec * scaled_specmap), 0.0, 1.0) * (1 - shoulder_mask) + shoulder_mask * (ground_truth["rgb"].transpose(2, 1).reshape(-1, 3, 256, 256) / 2 + 0.5)) * 2 - 1

                sample_light_index = random.randint(0,15)
                sample_light = self.albedo_net.sample_lights[sample_light_index]

                sample_space_normal, sample_space_shading = normal_shading(sample_light)
                sample_img_light = torch.cuda.FloatTensor(sample_space_shading/255*2-1).view(-1, 1, 256, 256)

                sample_space_shading = sample_space_shading.reshape(length, -1) / 255
                sample_space_shading = sample_space_shading[sample_index, ...]
                sample_space_normal = sample_space_normal[sample_index, ...]
                sample_h = torch.cuda.FloatTensor(normalize(sample_space_normal + np.array([[0, 0, 1]]).repeat(sample_space_shading.shape[0], 0))).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                sample_nh = fine_normal[:, [0] , :, :] * sample_h[:, :, 0, :, :] + fine_normal[:, [1] , :, :] * sample_h[:, :, 1, :, :] + fine_normal[:, [2] , :, :] * sample_h[:, :, 2, :, :]
                sample_z = torch.cuda.FloatTensor(sample_space_shading[np.newaxis, :, np.newaxis]) * sample_nh
                sample_sep_spec = (s + 2) / (2 * pi) * torch.pow(sample_z, s)
                sample_spec = torch.sum(sample_sep_spec, 1, keepdim=True)

                sample_masked_spec = (sample_spec - torch.min(sample_spec)) / (torch.max(sample_spec) - torch.min(sample_spec)) * face_mask
                sample_masked_spec = (sample_masked_spec - torch.min(sample_masked_spec)) / (torch.max(sample_masked_spec) - torch.min(sample_masked_spec))
     
                sample_shading = add_SHlight(self.albedo_net.constant_factor, fine_normal, sample_light.view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
                sample_masked_shading = (sample_shading - torch.min(sample_shading)) / (torch.max(sample_shading) - torch.min(sample_shading)) * face_mask
                sample_masked_shading_nonzero = sample_masked_shading[sample_masked_shading.nonzero(as_tuple=True)]
                sample_masked_shading_min = torch.min(sample_masked_shading_nonzero)
                sample_masked_shading_max = torch.max(sample_masked_shading_nonzero)
                sample_masked_shading = torch.clamp((sample_masked_shading - sample_masked_shading_min) / (sample_masked_shading_max - sample_masked_shading_min), 0.0, 1.0)               

                sample_final_image = (torch.clamp(albedo.detach() * (sample_masked_shading + sample_masked_spec * scaled_specmap.detach()), 0.0, 1.0) * (1 - shoulder_mask) + shoulder_mask * (ground_truth["rgb"].transpose(2, 1).reshape(-1, 3, 256, 256) / 2 + 0.5) * model_input['object_mask'].view(-1, 1, 256, 256) 
                    + (1 - model_input['object_mask'].view(-1, 1, 256, 256).float())) * 2 - 1

                model_outputs = {
                    'rgb_values': final_image.view(-1, 3, 256*256).transpose(2, 1).reshape(-1, 3),
                    'spec_values': masked_spec.view(-1, 1),
                    'specmap_values': specmap.view(-1, 1),
                    'normal_values': ground_truth['normal'].view(-1, 3),
                    'fine_normal_values': fine_normal.view(-1, 3, 256*256).transpose(2, 1).reshape(-1, 3),
                    'albedo_values': albedo.view(-1, 3, 256*256).transpose(2, 1).reshape(-1, 3),
                    'object_mask': model_input["object_mask"].reshape(-1),
                    'face_mask': face_mask,
                    'skin_mask': skin_mask,
                    'sample_rgb_values': sample_final_image.view(-1, 3, 256*256).transpose(2, 1).reshape(-1, 3),
                }

                if self.optimize_inputs:
                    if self.optimize_expression:
                        ground_truth['expression'] = model_input['expression']
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_latent_code:
                        model_input['latent_code'] = self.latent_codes(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        ground_truth['flame_pose'] = model_input['flame_pose']
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        ground_truth['cam_pose'] = model_input['cam_pose']
                        model_input['cam_pose'] = torch.eye(4).unsqueeze(0).repeat(ground_truth['cam_pose'].shape[0], 1, 1).cuda()
                        model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

                loss_output = self.loss(model_outputs, ground_truth, epoch)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.optimize_inputs:
                    self.optimizer_cam.zero_grad()

                loss.backward()

                self.optimizer.step()
                if self.optimize_inputs:
                    self.optimizer_cam.step()

                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                if data_index % 50 == 0:
                    for k, v in acc_loss.items():
                        acc_loss[k] = sum(v) / len(v)
                    print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                    for k, v in acc_loss.items():
                        print_str += '{}: {} '.format(k, v)
                    print(print_str)
                    wandb.log(acc_loss)
                    acc_loss = {}

            self.scheduler.step()
            print("Epoch time: {}".format(time.time() - start_time))
            del model_outputs, ground_truth


