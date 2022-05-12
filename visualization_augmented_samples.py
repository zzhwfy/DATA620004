'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from ResNet18 import resnet18
from training import train_baseline, mixup_data, cutmix_data, Cutout
from utils import progress_bar, set_seed
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

class Visualization(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=3, pad_value=1)
        writer.add_image(tag, grid, epoch)        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Visualization')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--datapath', default= './dataset') #存放CIFAR100的路径
    parser.add_argument('--logpath', default='./results/visualization') #希望保存的logs路径，包括tensorboard
    #########BASIC TRAINING SETTING###############
    parser.add_argument('--seed', type=int, default=7777)
    #########SPECIFIC PARAMS###############
    parser.add_argument('--alpha', default=1.0, type=float, help='params for cutmix and mixup')
    parser.add_argument('--cut_prob', default=1, type=float, help='params for cutmix')
    parser.add_argument('--n_hole', default=1, type=int, help='params for cutout')
    parser.add_argument('--length', default=16, type=int, help='params for cutout')
    args = parser.parse_args()
    set_seed(args.seed)
    #makedir
    if os.path.exists(args.logpath) == 0:
        os.makedirs(args.logpath)

    device = torch.device("cuda:{}".format(str(args.gpu_id)) if torch.cuda.is_available() else "cpu")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR100(
            root=args.datapath, train=True, download=False, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=16, shuffle=True, num_workers=2)
    writer = SummaryWriter(os.path.join(args.logpath, 'tensorboard'))
    visualizer = Visualization()
    
    trainloaderiter = iter(trainloader)
    inputs, targets = next(trainloaderiter)
    inputs, targets = inputs.to(device), targets.to(device)
    #origin
    ori_inputs = inputs
    
    #cutmix
    inputs_copy, targets_copy = inputs.clone(), targets.clone()
    cutmix_inputs, _, _, _ = cutmix_data(inputs_copy, targets_copy, args.alpha, device = device)

    #cutout
    inputs_copy, targets_copy = inputs.clone(), targets.clone()
    cutout = Cutout(n_holes=args.n_hole, length=args.length)
    cutout_inputs = cutout(inputs_copy)
    
    #mixup
    inputs_copy, targets_copy = inputs.clone(), targets.clone()
    mixup_inputs, _, _, _ = mixup_data(inputs_copy, targets_copy, args.alpha, device = device)
    
    #visualization
    visualizer(ori_inputs[:16], 'ori_inputs', 0, writer)
    visualizer(cutmix_inputs[:16], 'cutmix_inputs', 0, writer)
    visualizer(cutout_inputs[:16], 'cutout_inputs', 0, writer)
    visualizer(mixup_inputs[:16], 'mixup_inputs', 0, writer)
    chosen_indexes = [3,8,13]
    chosen_samples = torch.cat([ori_inputs[chosen_indexes],cutmix_inputs[chosen_indexes],
                                cutout_inputs[chosen_indexes],mixup_inputs[chosen_indexes]], dim = 0)
    visualizer(chosen_samples, 'comparison', 0, writer)
    print('Done.')