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


def train():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()
    print(args)

    # dist.launch(train, args.n_gpu, 1, 0, args.dist_url, args=(args, ))
    dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=args.rank, world_size=args.word_size)

    ##########
    # load data
    # dataset
    # sampler = DistributedSampler(dataset)
    # loader = DataLoader(dataset, batch_szie=batch_size, sampler=sampler)
    ##########
    # load model
    # model = VQGAN()
    # model = model.cuda()
    # model = DDP(model)
