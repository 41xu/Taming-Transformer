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

