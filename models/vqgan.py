import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VQGAN(nn.Module):
    """
    VQ-VAE-2, top and bottom layer
    traditional official VQ-VAE-2 input: 256x256, bottom: 64x64, top: 32x32 1, 4, 8 downsample
    Timing: input: 256x256, downsample: 1, 1, 2, 4, 8 layer
    """
    def __init__(self):
        super(VQGAN, self).__init__()


def nonlinear(x):
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    """
    TODO: asymmetric padding, still confused... maybe in the original VQ-VAE-2 paper...?
    maybe ACNe? see https://arxiv.org/pdf/1908.03930.pdf and VQ-VAE-2 and also Taming paper for more details
    """
    def __init__(self, in_channel, with_conv = True):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            # TODO: still confused, copy from Taming official code
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """
    in_channel: input channel or can be seen as dim
    out_channel: out dim
    """
    def __init__(self, in_channel,out_channel, dropout, eps=1e-6, conv_shortcut=False):
        super(ResnetBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channel, eps=eps, affine=True)
        # after each norm, add a nonlinear(x) layer in forward process
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channel, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.shortcut = None
        if in_channel != out_channel:
            if conv_shortcut:
                self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
                # 1x1 convolution, no padding

    def forward(self, input):
        output = self.norm1(input)
        output = nonlinear(output)
        output = self.conv1(output)

        output = self.norm2(output)
        output = nonlinear(output)
        output = self.dropout(output)
        output = self.conv2(output)

        output = self.shortcut(output)

        return output + input


class ResnetStack(nn.Module):
    def __init__(self, in_channel,out_channel, dropout, n_res_layers):
        super(ResnetStack, self).__init__()
        self.stack = nn.ModuleList([ResnetBlock(in_channel, out_channel, dropout=dropout)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channel, eps=1e-6):
        super(AttnBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channel, eps=eps, affine=True)
        self.q = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channel,in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        output = self.norm(input)
        q = self.q(output)
        k = self.k(output)
        v = self.v(output)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1) # b, h*w, c
        k = k.reshape(b, c, h * w) # b, c, h*w
        w = torch.bmm(q, k) # b, hw, hw
        w = F.softmax(w, dim=2) # row softmax

        # attend to values
        v = v.reshape(b, c, h*w)
        w = w.permute(0, 2, 1) # col softmax, multi v = attention value
        h = torch.bmm(v, w) # b, c, hw
        h = h.reshape(b, c, h, w)

        h = self.proj_out(h)

        return input + h # residual connection,




class ResidualLayer(nn.Module):
    """
    in_dim: input dim
    h_dim: hidden layer dim
    res_h_dim: hidden dim of residual block
    residual block: output = input + F(input)
    abandoned, used in VQ-VAE, but here we use VQ-VAE-2.
    """
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, input):
        output = self.res_block(input)
        output += input
        return output


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h__dim, res_h_dim, n_res_layer):
        super(ResidualStack, self).__init__()
        self.n_res_layer = n_res_layer
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h__dim, res_h_dim)] * self.n_res_layer)

    def foreward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    """
    q_theta(z|x), input x to latent code z
    in_dim, h_dim, residual_dim, n_residual_layers
    multi resolution layers
    input: 256x256, downsample: 1, 1, 2, 4, 8 layer
    ##########
    h_channel: hidden dim, 128 default
    z_channel: 256
    resolution: 256 (256, 256, 3)
    attn_resolution = [16]
    """
    def __init__(self, in_channel, z_channel, h_channel,
                 n_res_layers, resolution, attn_resolution, ch_mult=(1, 2, 4, 8), dropout=0.0, resample_with_conv=True,
                 eps=1e-6):
        super(Encoder, self).__init__()
        self.num_resolutions = len(ch_mult) # 4
        self.num_res_layers = n_res_layers
        in_ch_mult = (1, ) + tuple(ch_mult) # (1, 1, 2, 4, 8), multi-resolution input channel
        # input and output channel: (128, 128) -> (128, 128*2) ->(128*2, 128*4) -> (128*4, 128*8)

        # downsample
        self.conv_in = nn.Conv2d(in_channel, h_channel, kernel_size=3, stride=1, padding=1)
        self.downsample_layers = nn.ModuleList()
        cur_resolution = resolution

        for i_level in range(self.num_resolutions):
            resstack = nn.ModuleList()
            attnstack = nn.ModuleList()
            downstack = nn.ModuleList()
            res_in, res_out = h_channel * in_ch_mult[i_level], h_channel * ch_mult[i_level]
            # 这里为了好写multi-resolution的循环，不用加first layer and last layer的判断所以额外加了前面一层1_input_channel
            for i_block in range(self.num_res_layers):
                # 这里由于multi-resolution还加了一个attention的原因，所以没有直接用ResnetStack
                # 不是很清楚为什么在resolution=16的时候加了一个AttenBlock，感觉可以去掉试试看看对比效果
                resstack.append(ResnetBlock(res_in, res_out, dropout=dropout))
                if cur_resolution in attn_resolution:
                    attnstack.append(AttnBlock(res_out))

            if i_level != self.num_resolutions - 1: # not last layer, downsample, current resolution need to adjust
                downstack.append(Downsample(res_out, resample_with_conv))
                cur_resolution //= 2

            self.downsample_layers.append(resstack)
            self.downsample_layers.append(attnstack)
            self.downsample_layers.append(downstack)

        # middle layer
        self.middle_layers = nn.ModuleList([
            ResnetBlock(res_out, res_out, dropout),
            AttnBlock(res_out),
            ResnetBlock(res_out, res_out, dropout),
        ])
        # last layer, top layer
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=res_out, eps=eps, affine=True)
        self.out_conv = nn.Conv2d(res_out, z_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        hidden_layers = [self.conv_in(x)]
        # bottom layer, downsample
        x = self.downsample_layers(x)
        # middle layer
        # top layer, end




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()



class VectorQuantizer(nn.Module):
    def __init__(self):
        super(VectorQuantizer, self).__init__()
        pass
