import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class Albedo_ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(Albedo_ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [Upsample_Layer(ngf * mult, int(ngf * mult / 2), bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        
        self.constant_factor = torch.tensor([0.5/np.sqrt(np.pi),
                                    np.sqrt(3)/2/np.sqrt(np.pi), 
                                    np.sqrt(3)/2/np.sqrt(np.pi),
                                    np.sqrt(3)/2/np.sqrt(np.pi),
                                    np.sqrt(15)/2/np.sqrt(np.pi),
                                    np.sqrt(15)/2/np.sqrt(np.pi),
                                    np.sqrt(5)/4/np.sqrt(np.pi),
                                    np.sqrt(15)/2/np.sqrt(np.pi),
                                    np.sqrt(15)/4/np.sqrt(np.pi)]).float()

        self.sample_lights = torch.tensor([[0.000e+00,
                                    0.999e+00, 
                                    0.333e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    -0.999e+00, 
                                    0.333e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.000e+00, 
                                    0.333e+00,
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.000e+00, 
                                    0.333e+00,
                                    -0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.999e+00, 
                                    0.333e+00,
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.999e+00, 
                                    0.333e+00,
                                    -0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    -0.999e+00, 
                                    0.333e+00,
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    -0.999e+00, 
                                    0.333e+00,
                                    -0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.999e+00, 
                                    0.111e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    -0.999e+00, 
                                    0.111e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.000e+00, 
                                    0.111e+00,
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.000e+00, 
                                    0.111e+00,
                                    -0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.999e+00, 
                                    0.111e+00,
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    0.999e+00, 
                                    0.111e+00,
                                    -0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    -0.999e+00, 
                                    0.111e+00,
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00],
                                    [0.000e+00,
                                    -0.999e+00, 
                                    0.111e+00,
                                    -0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00]]).float()

        self.light = nn.parameter.Parameter(torch.tensor([0.000e+00,
                                    0.000e+00, 
                                    0.999e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00,
                                    0.000e+00]).float())

    def forward(self, input):
        """Standard forward"""
        albedo = self.model(input) / 2 + 0.5
        return albedo


class Upsample_Layer(nn.Module):
    def __init__(self, input_channel, output_channel, bias):
        super(Upsample_Layer, self).__init__()
        model = [nn.Upsample(scale_factor=2),
                 nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=bias),]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
        
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [Upsample_Layer(ngf * mult, int(ngf * mult / 2), bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
        
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetCatGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', cat_num=53):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetCatGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        up_model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            up_model += [ResnetBlock(ngf * mult + cat_num, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if i == 0:
                up_model += [Upsample_Layer(ngf * mult + cat_num, int(ngf * mult / 2), bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            else:
                up_model += [Upsample_Layer(ngf * mult, int(ngf * mult / 2), bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        up_model += [nn.ReflectionPad2d(3)]
        up_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up_model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.up_model = nn.Sequential(*up_model)

    def forward(self, input, cat = None):
        """Standard forward"""
        a = self.model(input)
        b = self.up_model(torch.cat([a, cat.unsqueeze(-1).unsqueeze(-1).repeat(1,1,64,64)], 1))
        return b
        