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
    def __init__(self, in_channel, z_channel, h_channel, n_res_layers, resolution, attn_resolution,
                 embed_dim, n_embed, beta=0.25):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(in_channel=in_channel, z_channel=z_channel, h_channel=h_channel,
                               n_res_layers=n_res_layers, resolution=resolution, attn_resolution=attn_resolution)
        self.quant_conv = nn.Conv2d(z_channel, embed_dim, kernel_size=1, stride=1, padding=0)
        self.quantize = VectorQuantizer(n_embed=n_embed, embed_dim=embed_dim, beta=beta)
        self.pos_quant_conv = nn.Conv2d(embed_dim, z_channel, kernel_size=1, stride=1, padding=0)
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z = self.quant_conv(z)
        quant, embed_loss, info = self.quantize(z)
        quant = self.pos_quant_conv(quant)
        dec = self.decoder(quant)



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


class Upsample(nn.Module):
    def __init__(self, in_channel, with_conv=True):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
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
    ————————————————————
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
        # and last layer conv input into z_channel(256): 128*8, 256

        # downsample
        self.conv_in = nn.Conv2d(in_channel, h_channel, kernel_size=3, stride=1, padding=1)
        self.downsample_layers = nn.ModuleList()
        cur_resolution = resolution # 256

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
                res_in = res_out # out_channel update
                if cur_resolution in attn_resolution:
                    attnstack.append(AttnBlock(res_out))

            if i_level != self.num_resolutions - 1: # not last layer, downsample, current resolution need to adjust
                downstack.append(Downsample(res_out, resample_with_conv))
                cur_resolution //= 2 # init 256: -> 128, 64, 32

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
        x = self.conv_in(x)
        # bottom layer, downsample
        x = self.downsample_layers(x)
        # middle layer
        x = self.middle_layers(x)
        # top layer, end
        x = self.out_norm(x)
        x = nonlinear(x)
        x = self.out_conv(x)

        return x


class VectorQuantizer(nn.Module):
    """
    reference: official code and https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py version
    discretization
    encoder output: z(with dim of z_channel in latent space) -> Conv2d into some embedding dim: embed_dim -> quantize
    learn a codebook,(can be seen as a embedding table), n_embed, embed_dim, beta
    denote tensor in codebook as e (embed_dim)
    codebook loss: ||sg[z_e(x)]-e||^2
    encoder loss: beta*||z_e(x)-sg[e]||^2
    encoder output: z_e( embed_dim), z_e find nearest e_i in codebook, get q(z|x), e_i: discrete in codebook, one-hot vector
    ————————————————————
    input: embed_dim tensor (b, c, h, w) -> flatten into (b*h*w, c)
    output: z_q (discrete)
    """
    def __init__(self, n_embed, embed_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.beta = beta
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1. / self.n_embed, 1. / self.n_embed)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        # reshape: (b, c, h, w) -> (b, h, c, w) and flatten
        # c is the z_channel(in training process, encoder output z_channel, and conv(z_channel, embed_dim)
        # therefore c = embed_dim, h*w is the num of tensor
        # flatten z into (b*h*w, c) <=> (b*h*w, embed_dim)
        z_flattened = z.view(-1, self.embed_dim)
        # distances from z to embeddings e_j: (z-e)^2=z^2 + e^2 - 2*e*z
        # shape: b*h*w,embed_dim, n_embed, embed_dim, therefore z*e.T
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        print("min encoding indices shape: ", min_encoding_indices.shape())

        min_encoding = torch.zeros(min_encoding_indices.shape[0], self.n_embed).to(z)
        min_encoding.scatter_(1, min_encoding_indices, 1)
        print("min encoding shape: ", min_encoding.shape())

        # TODO: embedding part
        # quantized latent vector
        z_q = torch.matmul(min_encoding, self.embedding.weight).view(z.shape)

        # loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradient
        z_q = z + (z_q - z).detach()
        # perplexity
        e_mean = torch.mean(min_encoding, dim=0)
        perplexity = torch.exp(- torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape, b, c, h, w
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encoding, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """
        (b, h, w, c）
        for decode part
        """
        min_encoding = torch.zeros(indices.shape[0], self.n_embed).to(indices)
        min_encoding.scatter_(1, indices[:, None], 1)

        # get quantized latent vector
        z_q = torch.matmul(min_encoding.float(), self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class Decoder(nn.Module):
    """
    dual with encoder
    h_channel: hidden dim, 128 default
    z_channel: 256
    resolution: 256 (256, 256, 3)
    attn_resolution: [16]
    in_channel: input dim , out_channel: output dim
    """
    def __init__(self, out_channel, h_channel, z_channel,n_res_layers,
                 resolution, attn_resolution, ch_mult=(1, 2, 4, 8), dropout=0.0, resample_with_conv=True, eps=1e-6):
        super(Decoder, self).__init__()
        self.num_resolutions = len(ch_mult) # 4
        self.num_res_layers = n_res_layers

        # top to bottom layer, top with the lowest resolution
        # upsample, z_channel(256) -> hidden_channel, in and out channel:
        # encoder里downsample了num_resolutions -1 次（3次），last resolution: 32
        # （res_in其实应该是output resolution，这里 128 x 8 即超分放大了4倍
        # decoder里对res_in对应的其实是encoder里的res_out，一切都反过来

        res_in = h_channel * ch_mult[-1]
        cur_resolution = resolution // 2 ** (self.num_resolutions - 1) # 32
        self.z_shape = (1, z_channel, cur_resolution, cur_resolution)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # 这里先把encoder last layer几层的转换转回来
        self.conv_in = nn.Conv2d(z_channel, res_in, kernel_size=3, stride=1, padding=1)
        # middle layer
        self.middle_layers = nn.ModuleList([
            ResnetBlock(res_in, res_in, dropout),
            AttnBlock(res_in),
            ResnetBlock(res_in, res_in, dropout)
        ])
        # upsample layers
        self.upsample_layers = nn.ModuleList()
        for i_level in range(self.num_resolutions - 1, -1, -1):
            resstack = nn.ModuleList()
            attnstack = nn.ModuleList()
            upstack = nn.ModuleList()
            res_out = h_channel * ch_mult[i_level]
            for i_block in range(self.num_res_layers + 1):
                resstack.append(ResnetBlock(res_in, res_out, dropout=dropout))
                res_in = res_out
                if cur_resolution in attn_resolution:
                    attnstack.append(AttnBlock(res_in))

            if i_level != 0: # not last layer, upsample, current resolution need to adjust
                upstack.append(Upsample(res_out, resample_with_conv))
                cur_resolution *= 2

            self.upsample_layers.append(resstack)
            self.upsample_layers.append(attnstack)
            self.upsample_layers.append(upstack)

        # last layer 这里的out_conv和encoder里的conv_in对应, 不过和encoder最后很像，要先out_norm, nonlinear, 最后out_cnov
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=res_out, eps=eps, affine=True)
        self.out_conv = nn.Conv2d(res_out, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        # z back into res_in
        x = self.conv_in(z)
        # middle layer
        x = self.middle_layers(x)
        # upsample
        x = self.upsample_layers(x)
        # end
        x = self.out_norm(x)
        x = nonlinear(x)
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    encoder = Encoder(in_channel=3, z_channel=256, h_channel=128, n_res_layers=2, resolution=256, attn_resolution=[16])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print(encoder)
    print("----------")
    decoder = Decoder(out_channel=3, h_channel=128, z_channel=256, n_res_layers=2, resolution=256, attn_resolution=[16])
    device = decoder.to(device)
    print(decoder)
