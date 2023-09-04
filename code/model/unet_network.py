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

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class Albedo_UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Albedo_UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        # pi = np.pi
        # self.constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
        #                     ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
        #                     (pi/4)*(1/2)*(np.sqrt(5/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi)))]).float()

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
                                    0.000e+00]]).float()

        self.light = nn.parameter.Parameter(torch.tensor([0.000e+00,
                                    0.333e+00, 
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


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


# dilated convs, without downsampling
class UnetSkipConnectionBlock_DC(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, dilation=1):
        super(UnetSkipConnectionBlock_DC, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        #use_norm = False
        use_norm = True

        #downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, dilation=dilation, padding=1, bias=use_bias)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, dilation=dilation, padding=1*dilation, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
     
        if use_norm: downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        if use_norm: upnorm = norm_layer(outer_nc)

        if outermost:
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv = nn.Sequential(
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, dilation=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            #upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, dilation=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            down = [downrelu, downconv]
            if use_norm: up = [uprelu, upconv, upnorm]
            else: up = [uprelu, upconv]
            model = down + up
        else:
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, dilation=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            if use_norm: down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            if use_norm: up = [uprelu, upconv, upnorm]
            else: up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class DcUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(DcUnetGenerator, self).__init__()
        # construct unet structure
        dilation = 1
        unet_block = UnetSkipConnectionBlock_DC(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, dilation=dilation)
        for i in range(num_downs - 5):
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, dilation=dilation)
        dilation *= 2
        unet_block = UnetSkipConnectionBlock_DC(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dilation=dilation)
        dilation *= 2
        unet_block = UnetSkipConnectionBlock_DC(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dilation=dilation)
        dilation *= 2
        unet_block = UnetSkipConnectionBlock_DC(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dilation=dilation)
        dilation *= 2
        unet_block = UnetSkipConnectionBlock_DC(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, dilation=dilation)

        self.model = unet_block

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class WholeGenerator(nn.Module):

    def __init__(self):
        super(WholeGenerator, self).__init__()

        import model.resnet_network as ResNet
        self.render_net = ResNet.ResnetGenerator(output_nc=3, input_nc=3)
        self.blend_net = DcUnetGenerator(output_nc=3, input_nc=9)

    def forward(self, x, y, mask1, mask2):
        local = self.render_net(x)
        out = self.blend_net(torch.cat([local.detach() * mask1, y * mask2, x], 1))
        return local, out

class WavGenerator(nn.Module):

    def __init__(self):
        super(WavGenerator, self).__init__()

        import model.resnet_network as ResNet
        self.render_net = ResNet.ResnetGenerator(output_nc=3, input_nc=6)
        self.blend_net = DcUnetGenerator(output_nc=3, input_nc=9)

    def forward(self, x, y, wav, mask1, mask2):
        local = self.render_net(torch.cat([x, wav], 1))
        out = self.blend_net(torch.cat([local.detach() * mask1, y * mask2, x], 1))
        return local, out
