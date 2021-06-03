import sys
import os
import argparse
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
