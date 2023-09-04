"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from facenet_pytorch import InceptionResnetV1

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=torch.device('cuda:0')).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x

class Loss(nn.Module):
    def __init__(self, mask_weight, lbs_weight, flame_distance_weight, alpha, expression_reg_weight, pose_reg_weight, cam_reg_weight, gt_w_seg=False):
        super().__init__()
        
        self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.flame_distance_weight = flame_distance_weight
        self.expression_reg_weight = expression_reg_weight
        self.cam_reg_weight = cam_reg_weight
        self.pose_reg_weight = pose_reg_weight
        self.gt_w_seg = gt_w_seg
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.tv_loss = TVLoss()

        self.resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()

    def get_albedo_loss(self, albedo_values, network_object_mask):
        masked_albedo = albedo_values[network_object_mask]
        albedo_entropy = 0
        for i in range(3):
            channel = masked_albedo[..., i]
            hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
            h = hist(channel)
            if h.sum() > 1e-6:
                h = h.div(h.sum()) + 1e-6
            else:
                h = torch.ones_like(h).to(h)
            albedo_entropy += torch.sum(-h*torch.log(h))
        return albedo_entropy

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l2_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_l1_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_gray_loss(self, gray_values, gray_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        gray_values = gray_values[network_object_mask & object_mask]
        gray_gt = gray_gt.reshape(-1, 1)[network_object_mask & object_mask]
        gray_loss = self.l2_loss(gray_values, gray_gt) / float(object_mask.shape[0])
        return gray_loss

    def get_id_loss(self, values, gt):

        predict_embedding = self.resnet(values)
        gt_embedding = self.resnet(gt)
        id_loss = self.l2_loss(predict_embedding, gt_embedding.detach()) / gt_embedding.shape[1]
        return id_loss

    def forward(self, model_outputs, ground_truth, epoch=0):
        network_object_mask = model_outputs['object_mask']
        object_mask = model_outputs['object_mask']
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], ground_truth['rgb'], network_object_mask, object_mask)

        albedo_loss = 0.001 * self.get_albedo_loss(model_outputs['albedo_values'] * model_outputs['skin_mask'].view(-1, 1), network_object_mask)
        normal_loss = 1 * self.get_rgb_loss(model_outputs['fine_normal_values'], model_outputs['normal_values'], network_object_mask, object_mask)
        normal_tv_loss = 1 * self.tv_loss(model_outputs['fine_normal_values'].permute(1, 0).reshape(-1, 3, 256, 256) * model_outputs['skin_mask']) 
        tv_loss = 1 * self.tv_loss(model_outputs['albedo_values'].permute(1, 0).reshape(-1, 3, 256, 256) * model_outputs['skin_mask'])
        spec_tv_loss =  0.5 * self.tv_loss(model_outputs['spec_values'].reshape(-1, 1, 256, 256) * model_outputs['skin_mask'])
        specmap_tv_loss =  0.01 * self.tv_loss(model_outputs['specmap_values'].reshape(-1, 1, 256, 256) * model_outputs['skin_mask'])
        
        if epoch < 3:
            sample_id_loss = 0 * self.get_id_loss(model_outputs['sample_rgb_values'].reshape(-1, 3).transpose(1,0).reshape(-1, 3, 256, 256), ground_truth['rgb'].reshape(-1, 3).transpose(1,0).reshape(-1, 3, 256, 256))
        else:
            sample_id_loss = 3 * self.get_id_loss(model_outputs['sample_rgb_values'].reshape(-1, 3).transpose(1,0).reshape(-1, 3, 256, 256), ground_truth['rgb'].reshape(-1, 3).transpose(1,0).reshape(-1, 3, 256, 256))

        loss = rgb_loss + albedo_loss + normal_loss + sample_id_loss + tv_loss + normal_tv_loss + spec_tv_loss + specmap_tv_loss

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'albedo_loss': albedo_loss,
            'normal_loss': normal_loss,
            'sample_id_loss': sample_id_loss,
            'tv_loss': tv_loss,
            'normal_tv_loss': normal_tv_loss,
            'spec_tv_loss': spec_tv_loss,
            'specmap_tv_loss': specmap_tv_loss,
        }

        return out
