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
import time
import math

import utils.general as utils
import utils.plots as plt

import model.unet_network as UNet
import model.resnet_network as ResNet
from utils.light_util import add_SHlight, normal_shading_sh, normalize

from functools import partial
print = partial(print, flush=True)
class TestRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path)

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         sample_size=-1,
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_semantics=self.conf.get_bool('loss.gt_w_seg'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=1, # only support batch_size = 1
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )

        self.albedo_net = ResNet.Albedo_ResnetGenerator()
        self.spec_net = UNet.UnetGenerator(output_nc=1, input_nc=6)
        self.normal_net = ResNet.ResnetGenerator(input_nc=6)
        if torch.cuda.is_available():
            self.albedo_net.cuda()
            self.spec_net.cuda()
            self.normal_net.cuda()
            self.albedo_net.constant_factor = self.albedo_net.constant_factor.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        assert os.path.exists(old_checkpnts_dir)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'LightModelParameters', str(kwargs['checkpoint']) + ".pth"))
        self.albedo_net.load_state_dict(saved_model_state["model_state_dict"], strict=False)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'SpecModelParameters', str(kwargs['checkpoint']) + ".pth"))
        self.spec_net.load_state_dict(saved_model_state["model_state_dict"], strict=False)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'NormalModelParameters', str(kwargs['checkpoint']) + ".pth"))
        self.normal_net.load_state_dict(saved_model_state["model_state_dict"], strict=False)

        self.start_epoch = saved_model_state['epoch']
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        if self.optimize_latent_code:
            self.input_params_subdir = "InputParameters"
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.latent_codes = torch.nn.Embedding(data["latent_codes_state_dict"]['weight'].shape[0], 32, sparse=True).cuda()
            self.latent_codes.load_state_dict(data["latent_codes_state_dict"])
        self.total_pixels = self.plot_dataset.total_pixels
        self.img_res = self.plot_dataset.img_res
        self.plot_conf = self.conf.get_config('plot')

    def run(self):
        new_light = True
        sample_index = list(range(0, 256*256, 64))
        s = 8
        pi = torch.acos(torch.zeros(1)).item() * 2

        pre_light = torch.cuda.FloatTensor([0.000e+00,
                            0.000e+00, 
                            0.999e+00,
                            0.000e+00,
                            0.000e+00, 
                            0.000e+00,
                            0.000e+00,
                            0.000e+00,
                            0.000e+00])

        print_all = True
        self.albedo_net.eval()
        self.spec_net.eval()
        self.normal_net.eval()
        eval_iterator = iter(self.plot_dataloader)
        for img_index in range(len(self.plot_dataset)):
            start_time = time.time()
            if img_index >= self.conf.get_int('plot.plot_nimgs') and not print_all:
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


            # a = img_index / len(self.plot_dataset) * 2 - 1
            theta = math.pi * (img_index % 125 / 125)
            a = -math.cos(theta)
            b = -math.sin(theta)
            sh = np.array([0.000e+00,
                            b, 
                            0.000e+00,
                            a,
                            0.000e+00,
                            0.000e+00,
                            0.000e+00,
                            0.000e+00,
                            0.000e+00])
            print(a, b)
            light = torch.FloatTensor(sh).cuda()

            skin_mask = ground_truth["semantics"][:,:,0].view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)
            face_mask = torch.sum(ground_truth["semantics"][:,:,:-2], 2).view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)
            shoulder_mask = ground_truth["semantics"][:,:,-2].view(-1, 1, 256, 256)
            foreground_mask = torch.sum(ground_truth["semantics"][:,:,:-1], 2).view(-1, 1, 256, 256) * model_input["object_mask"].view(-1, 1, 256, 256)

            space_normal, space_shading = normal_shading_sh(light)

            albedo = self.albedo_net(ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256))
            normal = ground_truth['normal'].transpose(2, 1).view(-1, 3, 256, 256)
            fine_normal = (self.normal_net(torch.cat([ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256), normal], 1)) + 1) / 2 * face_mask * 2 - 1
            fine_normal_sh = fine_normal.clone()

            fine_normal_sh[:, 1, ...] = -((fine_normal[:, 2, ...] + 1) / 2)
            fine_normal_sh[:, 2, ...] = fine_normal[:, 1, ...]
            img_light = torch.cuda.FloatTensor(space_shading/255)

            length = space_normal.shape[0]
            space_shading = space_shading.reshape(length, -1) / 255
            space_shading = space_shading[sample_index, ...]
            space_normal = space_normal[sample_index, ...]

            h = torch.cuda.FloatTensor(normalize(space_normal + np.array([[0, -1, 0]]).repeat(space_shading.shape[0], 0))).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
            nh = fine_normal_sh[:, [0] , :, :] * h[:, :, 0, :, :] + fine_normal_sh[:, [1] , :, :] * h[:, :, 1, :, :] + fine_normal_sh[:, [2] , :, :] * h[:, :, 2, :, :]
            z = torch.cuda.FloatTensor(space_shading[np.newaxis, :, np.newaxis]) * nh
            sep_spec = (s + 2) / (2 * pi) * torch.pow(z, s)
            spec = torch.sum(sep_spec, 1, keepdim=True)

            masked_spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec)) * face_mask
            masked_spec = (masked_spec - torch.min(masked_spec)) / (torch.max(masked_spec) - torch.min(masked_spec))
            
            shading = add_SHlight(self.albedo_net.constant_factor, fine_normal_sh, light.view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
            
            masked_shading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading)) * face_mask
            
            if new_light:
                pre_shading = add_SHlight(self.albedo_net.constant_factor, fine_normal_sh, self.albedo_net.light.view(1, -1 , 1).repeat(ground_truth['normal'].shape[0], 1, 1))
                masked_pre_shading = (shading - torch.min(pre_shading)) / (torch.max(pre_shading) - torch.min(pre_shading)) * face_mask
                adjust_coef = torch.sum(masked_pre_shading) / torch.sum(masked_shading)
                masked_shading_nonzero = masked_shading[masked_shading.nonzero(as_tuple=True)]
                masked_shading_min = torch.min(masked_shading_nonzero)
                masked_shading_max = torch.max(masked_shading_nonzero)
                new_light = False

            masked_shading = torch.clamp((masked_shading - masked_shading_min) / (masked_shading_max - masked_shading_min), 0.0, 1.0)
            adjust_shading = torch.clamp(torch.pow(torch.pow(adjust_coef, 0.5) * masked_shading, 1 / adjust_coef) * 1.2, 0.0, 1.2)

            specmap = self.spec_net(torch.cat([ground_truth['rgb'].transpose(2, 1).view(-1, 3, 256, 256), fine_normal.detach()], 1))
            scaled_specmap = (specmap + 1) / 2 * 0.2
            final_image = (torch.clamp(albedo * (adjust_shading + masked_spec * scaled_specmap), 0.0, 1.0) * (1 - shoulder_mask) + shoulder_mask * (ground_truth["rgb"].transpose(2, 1).reshape(-1, 3, 256, 256) / 2 + 0.5)) * 2 - 1

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
                'bg_img': ground_truth['bg_img'].view(-1, 3),
            }

            if self.optimize_latent_code:
                # use the latent code from the first scripts frame
                model_input['latent_code'] = self.latent_codes(torch.LongTensor([0]).cuda()).squeeze(1).detach()

            batch_size = model_input['expression'].shape[0]
            plot_dir = os.path.join(self.eval_dir, model_input['sub_dir'][0], 'epoch_'+str(self.start_epoch))
            img_name = model_input['img_name'][0,0].cpu().numpy()
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if print_all:
                utils.mkdir_ifnotexists(plot_dir)
            print("Saving image {} into {}".format(img_name, plot_dir))
            plt.light_plot(img_name,
                     model_outputs,
                     model_input['cam_pose'],
                     ground_truth,
                     plot_dir,
                     self.start_epoch,
                     self.img_res,
                     is_eval=print_all,
                     **self.plot_conf
                     )
            print("Plot time per frame: {}".format(time.time() - start_time))
            del model_outputs, ground_truth