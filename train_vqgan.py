import sys
import os
import argparse
# import distributed as dist
import torch
import torchvision
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from models.vqgan import VQGAN


def train():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # some basic settings
    parser.add_argument("--n_gpu", type=int, default=0, help="when n_gpu is 0, model will use cpu")
    parser.add_argument("--size", type=int, default=256, help="image size")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_path", type=str, help="training image folder's parents path")
    parser.add_argument("--name", type=str, default="fire", help="name")
    parser.add_argument("--rank", type=int, default=0, help="rank of current process")
    parser.add_argument("--word_size", type=int, default=1, help="word size = GPU number?")
    port = (2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14)
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:" + str(port))
    # setting for VQ-VAE
    parser.add_argument("--in_channel", type=int, default=3)
    parser.add_argument("--out_channel", type=int, default=3)
    parser.add_argument("--z_channel", type=int, default=256)
    parser.add_argument("--h_channel", type=int, default=128)
    parser.add_argument("--n_res_layers", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--attn_resolution", type=list, default=[16])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_embed", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.25)

    args = parser.parse_args()
    print(args)

    # dist.launch(train, args.n_gpu, 1, 0, args.dist_url, args=(args, ))
    dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=args.rank, world_size=args.word_size)

    # ————————————————————
    # load data
    # dataset
    # sampler = DistributedSampler(dataset)
    # loader = DataLoader(dataset, batch_szie=batch_size, sampler=sampler)
    # ————————————————————
    # load model
    model = VQGAN(in_channel=args.in_channel, out_channel=args.out_channel, z_channel=args.z_channel, h_channel=args.h_channel,
                  n_res_layers=args.n_res_layers, resolution=args.resolution, attn_resolution=args.attn_resolution,
                  embed_dim=args.embed_dim, n_embed=args.n_embed, beta=args.beta)
    model = model.cuda()
    model = DDP(model)
