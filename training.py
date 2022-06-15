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
from utils import progress_bar, set_seed
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
########################################################
###########BASELINE###############
def train_baseline(net, criterion, optimizer, trainloader, device, writer, visualizer, epoch, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch,train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
  
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)
    visualizer(inputs[:16], 'origin_input', epoch, writer)
    return {'loss': train_loss / (batch_idx + 1), 'acc': 100. * correct / total}

########################################################
###########CUTMIX###############
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, beta=1.0, device = torch.device("cuda")):
    if beta > 0:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
    else:
        lam = 1

    rand_index = torch.randperm(x.size()[0]).to(device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    # compute output
    return x, y_a, y_b, lam

def cutmix_criterion(pred, y_a, y_b, lam, criterion = nn.CrossEntropyLoss()):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_cutmix(net, criterion, optimizer, trainloader, device, writer, visualizer, epoch, args):
    cutmix_prob = args.cut_prob
    alpha = args.alpha
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        r = np.random.rand(1)
        if r < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha, device = device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = cutmix_criterion(outputs, targets_a, targets_b, lam, criterion)
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch,train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)
    visualizer(inputs[:16], 'cutmix_input', epoch, writer)
    return {'loss': train_loss / (batch_idx + 1), 'acc': 100. * correct / total}




########################################################
###########MIXUP###############
def mixup_data(x, y, alpha=1.0, device = torch.device("cuda")):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam, criterion= nn.CrossEntropyLoss()):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(net, criterion, optimizer, trainloader, device, writer, visualizer, epoch, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    mixup_prob = args.cut_prob
    alpha = args.alpha
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        r = np.random.rand(1)
        if r < mixup_prob:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, device = device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = mixup_criterion(outputs, targets_a, targets_b, lam, criterion)
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch,train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)
    visualizer(inputs[:16], 'mixup_input', epoch, writer)
    return {'loss': train_loss / (batch_idx + 1), 'acc': 100. * correct / total}


########################################################
###########CUTOUT###############
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, batch_img, device = torch.device("cuda")):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = batch_img.size(2)
        w = batch_img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask).reshape(1, 1, *mask.shape)
        mask = mask.expand_as(batch_img).to(device)
        batch_img = batch_img * mask

        return batch_img


def train_cutout(net, criterion, optimizer, trainloader, device, writer, visualizer, epoch, args):
    print('\nEpoch: %d' % epoch)
    cutout_prob = args.cut_prob
    net.train()
    n_hole = args.n_hole
    length = args.length
    cutout = Cutout(n_holes=n_hole, length=length)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        r = np.random.rand(1)
        if r < cutout_prob:
            inputs = cutout(inputs)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)
    visualizer(inputs[:16], 'cutout_input', epoch, writer)
    return {'loss': train_loss / (batch_idx + 1), 'acc': 100. * correct / total}