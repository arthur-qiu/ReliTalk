"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util
import os
import torch.nn as nn
from utils import mesh_util

def plot(img_index, sdf_function, model_outputs, pose, ground_truth, path, epoch, img_res, plot_nimgs, min_depth, max_depth, res_init, res_up, is_eval=False):
    # arrange data to plot
    batch_size = pose.shape[0]
    num_samples = int(model_outputs['rgb_values'].shape[0] / batch_size)
    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    # plot rendered images

    depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = (depth.reshape(batch_size, num_samples, 1) - min_depth) / (max_depth - min_depth)
    if (depth.min() < 0.) or (depth.max() > 1.):
        print("Depth out of range, min: {} and max: {}".format(depth.min(), depth.max()))
        depth = torch.clamp(depth, 0., 1.)

    plot_images(model_outputs, depth, ground_truth, path, epoch, img_index, 1, img_res, batch_size, num_samples, is_eval)
    del depth, points, network_object_mask
    # Generate mesh.
    if is_eval:
        with torch.no_grad():
            import time
            start_time = time.time()
            meshexport = mesh_util.generate_mesh(sdf_function, level_set=0, res_init=res_init, res_up=res_up)
            meshexport.export('{0}/surface_{1}.ply'.format(path, img_index), 'ply')
            print("Plot time per mesh:", time.time() - start_time)
            del meshexport

def plot_depth_maps(depth_maps, path, epoch, img_index, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/depth'.format(path)):
        os.mkdir('{0}/depth'.format(path))
    img.save('{0}/depth/{1}.png'.format(path, img_index))


def plot_image(rgb, path, epoch, img_index, plot_nrow, img_res, type):
    rgb_plot = lin2img(rgb, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=True,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/{1}'.format(path, type)):
        os.mkdir('{0}/{1}'.format(path, type))
    img.save('{0}/{2}/{1}.png'.format(path, img_index, type))


def plot_images(model_outputs, depth_image, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, num_samples, is_eval):
    if 'rgb' in ground_truth:
        rgb_gt = ground_truth['rgb']
        rgb_gt = (rgb_gt.cuda() + 1.) / 2.
    else:
        rgb_gt = None
   		
    if 'albedo_values' in model_outputs:	
        albedo = model_outputs['albedo_values']	
        spec = model_outputs['spec_values'].repeat(1,3)	
        specmap = model_outputs['specmap_values'].repeat(1,3)	
        rough_rgb = model_outputs['rough_rgb_values']	
    else:	
        albedo = None	
        spec = None	
        specmap = None	
        rough_rgb = None

    rgb_points = model_outputs['rgb_values']
    rgb_points = rgb_points.reshape(batch_size, num_samples, 3)

    normal_points = model_outputs['normal_values']
    normal_points = normal_points.reshape(batch_size, num_samples, 3)

    rgb_points = (rgb_points + 1.) / 2.
    normal_points = (normal_points + 1.) / 2.

    output_vs_gt = rgb_points
    if rgb_gt is not None:
        output_vs_gt = torch.cat((output_vs_gt, rgb_gt, depth_image.repeat(1, 1, 3), normal_points), dim=0)
    else:
        output_vs_gt = torch.cat((output_vs_gt, depth_image.repeat(1, 1, 3), normal_points), dim=0)

    if 'albedo_values' in model_outputs:	
        rough_rgb_points = rough_rgb.reshape(batch_size, num_samples, 3)	
        rough_rgb_points = (rough_rgb_points + 1.) / 2.	
        albedo_points = albedo.reshape(batch_size, num_samples, 3)	
        spec_points = spec.reshape(batch_size, num_samples, 3)	
        specmap_points = specmap.reshape(batch_size, num_samples, 3)	
        output_vs_gt = torch.cat((output_vs_gt, rough_rgb_points, albedo_points, spec_points, specmap_points), dim=0)	

    if 'lbs_weight' in model_outputs:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap('Paired')
        red = cmap.colors[5]
        cyan = cmap.colors[3]
        blue = cmap.colors[1]
        pink = [1, 1, 1]

        lbs_points = model_outputs['lbs_weight']
        lbs_points = lbs_points.reshape(batch_size, num_samples, -1)
        if lbs_points.shape[-1] == 5:
            colors = torch.from_numpy(np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()
        else:
            colors = torch.from_numpy(np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()

        lbs_points = (colors * lbs_points[:, :, :, None]).sum(2)
        mask = torch.logical_not(model_outputs['network_object_mask'])
        lbs_points[mask[None, ..., None].expand(-1, -1, 3)] = 1.
        output_vs_gt = torch.cat((output_vs_gt, lbs_points), dim=0)
    if 'shapedirs' in model_outputs:
        shapedirs_points = model_outputs['shapedirs']
        shapedirs_points = shapedirs_points.reshape(batch_size, num_samples, 3, 50)[:, :, :, 0] * 50.

        shapedirs_points = (shapedirs_points + 1.) / 2.
        shapedirs_points = torch.clamp(shapedirs_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, shapedirs_points), dim=0)
    if 'semantics' in ground_truth:
        gt_semantics = ground_truth['semantics'].squeeze(0)
        semantic_gt = rend_util.visualize_semantics(gt_semantics).reshape(batch_size, num_samples, 3)/ 255.
        output_vs_gt = torch.cat((output_vs_gt, semantic_gt), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=output_vs_gt.shape[0]).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    wo_epoch_path = path.replace('/epoch_{}'.format(epoch), '')
    if not os.path.exists('{0}/rendering'.format(wo_epoch_path)):
        os.mkdir('{0}/rendering'.format(wo_epoch_path))
    img.save('{0}/rendering/epoch_{1}_{2}.png'.format(wo_epoch_path, epoch, img_index))
    if is_eval:
        plot_image(rgb_points, path, epoch, img_index, plot_nrow, img_res, 'rgb')
        plot_image(normal_points, path, epoch, img_index, plot_nrow, img_res, 'normal')
    del output_vs_gt

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

def light_plot(img_index, model_outputs, pose, ground_truth, path, epoch, img_res, plot_nimgs, min_depth, max_depth, res_init, res_up, is_eval=False):
    # arrange data to plot
    batch_size = pose.shape[0]
    num_samples = int(model_outputs['rgb_values'].shape[0] / batch_size)
    # plot rendered images

    if 'rgb' in ground_truth:
        rgb_gt = ground_truth['rgb']
        rgb_gt = (rgb_gt.cuda() + 1.) / 2.
    else:
        rgb_gt = None


    if 'bg_img' in model_outputs:
        rgb_points = model_outputs['rgb_values'] * (model_outputs['object_mask'] * model_outputs['foreground_mask']) + model_outputs['bg_img'] * (1 - model_outputs['object_mask'] * model_outputs['foreground_mask'])
    else:
        rgb_points = model_outputs['rgb_values'] * (model_outputs['object_mask'] * model_outputs['foreground_mask']) + (1 - model_outputs['object_mask'] * model_outputs['foreground_mask'])
    rgb_points = rgb_points.reshape(batch_size, num_samples, 3)

    normal_points = model_outputs['normal_values'] * model_outputs['face_mask'] + (1 - model_outputs['face_mask'])
    normal_points = normal_points.reshape(batch_size, num_samples, 3)

    fine_normal_points = model_outputs['fine_normal_values'] * model_outputs['face_mask'] + (1 - model_outputs['face_mask'])
    fine_normal_points = fine_normal_points.reshape(batch_size, num_samples, 3)

    albedo_points = model_outputs['albedo_values'] * model_outputs['face_mask'] + (1 - model_outputs['face_mask'])
    albedo_points = albedo_points.reshape(batch_size, num_samples, 3)

    specmap_points = model_outputs['specmap_values'] * model_outputs['face_mask'] + (1 - model_outputs['face_mask'])
    specmap_points = specmap_points.reshape(batch_size, num_samples, 3) 

    shading_points = model_outputs['shading_values'] * model_outputs['face_mask']
    shading_points = shading_points.reshape(batch_size, num_samples, 3)

    spec_points = model_outputs['spec_values'] * model_outputs['face_mask'] 
    spec_points = spec_points.reshape(batch_size, num_samples, 3)

    light_points = model_outputs['light_values']
    light_points = light_points.reshape(batch_size, num_samples, 3)

    rgb_points = (torch.clamp(rgb_points, -1.0, 1.0) + 1.) / 2.
    specmap_points = (torch.clamp(specmap_points, -1.0, 1.0) + 1.) / 2.
    normal_points = (torch.clamp(normal_points, -1.0, 1.0) + 1.) / 2.
    fine_normal_points = (torch.clamp(fine_normal_points, -1.0, 1.0) + 1.) / 2.

    output_vs_gt = rgb_points
    if rgb_gt is not None:
        output_vs_gt = torch.cat((rgb_gt, output_vs_gt, albedo_points, normal_points, fine_normal_points, specmap_points, spec_points, shading_points, light_points), dim=0)
    else:
        output_vs_gt = torch.cat((output_vs_gt, albedo_points,normal_points, fine_normal_points, specmap_points, spec_points, shading_points, light_points), dim=0)
   
    if 'sample_light_values' in model_outputs:

        sample_rgb_points = model_outputs['sample_rgb_values'] * model_outputs['object_mask'] + (1 - model_outputs['object_mask'])
        sample_rgb_points = sample_rgb_points.reshape(batch_size, num_samples, 3)

        sample_light_points = model_outputs['sample_light_values']
        sample_light_points = sample_light_points.reshape(batch_size, num_samples, 3)

        sample_rgb_points = (torch.clamp(sample_rgb_points, -1.0, 1.0) + 1.) / 2.
        output_vs_gt = torch.cat((output_vs_gt, sample_rgb_points, sample_light_points), dim = 0)

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=output_vs_gt.shape[0]).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if is_eval:
        wo_epoch_path = path.replace('/epoch_{}'.format(epoch), '')
        if not os.path.exists('{0}/rendering_test'.format(wo_epoch_path)):
            os.mkdir('{0}/rendering_test'.format(wo_epoch_path))
        img.save('{0}/rendering_test/epoch_{1}_{2}.png'.format(wo_epoch_path, epoch, img_index))

        plot_image(rgb_gt, path, epoch, img_index, 1, img_res, 'gt')
        plot_image(rgb_points, path, epoch, img_index, 1, img_res, 'rgb')
        plot_image(albedo_points, path, epoch, img_index, 1, img_res, 'albedo')
        plot_image(normal_points, path, epoch, img_index, 1, img_res, 'normal')
        plot_image(fine_normal_points, path, epoch, img_index, 1, img_res, 'fine_normal')
        plot_image(specmap_points, path, epoch, img_index, 1, img_res, 'specmap')
        plot_image(spec_points, path, epoch, img_index, 1, img_res, 'spec')
        plot_image(shading_points, path, epoch, img_index, 1, img_res, 'shading')
        plot_image(light_points, path, epoch, img_index, 1, img_res, 'light')
    else:
        wo_epoch_path = path.replace('/epoch_{}'.format(epoch), '')
        if not os.path.exists('{0}/rendering'.format(wo_epoch_path)):
            os.mkdir('{0}/rendering'.format(wo_epoch_path))
        img.save('{0}/rendering/epoch_{1}_{2}.png'.format(wo_epoch_path, epoch, img_index))

    del output_vs_gt