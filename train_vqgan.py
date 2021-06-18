import sys
import os
import argparse
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=0, help="when n_gpu is 0, model will use cpu")
    parser.add_argument("--size", type=int, default=256, help="image size")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_path", type=str, help="training image folder's parents path")
    parser.add_argument("--name", type=str, default="fire", help="name")
    parser.add_argument("")