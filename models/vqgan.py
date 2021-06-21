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
        output = self.conv2(output)





class ResidualLayer(nn.Module):
    """
    in_dim: input dim
    h_dim: hidden layer dim
    res_h_dim: hidden dim of residual block
    residual block: output = input + F(input)
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
    """
    def __init__(self, in_dim, h_dim, r_dim, n_residual_layers, kernel=4, stride=2):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
        )


    def forward(self, x):
        return x




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()



class VectorQuantizer(nn.Module):
    def __init__(self):
        super(VectorQuantizer, self).__init__()
        pass
