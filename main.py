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
from training import train_baseline, train_cutmix, train_cutout, train_mixup
from utils import progress_bar, set_seed
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

class Visualization(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)        
        

def prepare_training(args):
    '''
    return: model, dataloader,
    '''
    print('==> Preparing training..')
    print('==> Preparing model..')
    model = resnet18(nclasses = 100)
    print('==> Preparing optimizer and criterion..')
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=args.datapath, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root=args.datapath, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    writer = SummaryWriter(os.path.join(args.logpath, 'tensorboard'))
    visualizer = Visualization()
    print('==> Finish preparing.')
    return model, criterion, optimizer, scheduler, trainloader, testloader, writer, visualizer


def test(net, criterion, testloader, device, writer, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', 100.*correct/total, epoch)
    return {'acc': 100.*correct/total}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--test', '-t', action='store_true', help='test only')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--pretrain', default= None, help='resume from checkpoint')
    parser.add_argument('--datapath', default= './dataset') #存放CIFAR100的路径
    parser.add_argument('--logpath', default='./results/baseline') #希望保存的logs路径，包括tensorboard
    parser.add_argument('--model', default='baseline',choices=['baseline','cutmix','cutout','mixup'])
    #########BASIC TRAINING SETTING###############
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=4396)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    #########SPECIFIC PARAMS###############
    parser.add_argument('--alpha', default=1.0, type=float, help='params for cutmix and mixup')
    parser.add_argument('--cut_prob', default=0.5, type=float, help='probability for aguments')
    parser.add_argument('--n_hole', default=1, type=int, help='params for cutout')
    parser.add_argument('--length', default=16, type=int, help='params for cutout')
    args = parser.parse_args()
    set_seed(args.seed)
    #makedir
    if os.path.exists(args.logpath) == 0:
        os.makedirs(args.logpath)

    device = torch.device("cuda:{}".format(str(args.gpu_id)) if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer, scheduler, trainloader, testloader, writer, visualizer= prepare_training(args)
    best_acc = 0
    start_epoch = 0
    if args.pretrain is not None:
        # Load checkpoint.
        print('==> Load pretrained model from checkpoint..')
        assert os.path.exists(args.pretrain), 'Error: no checkpoint {} found!'.format(args.pretrain)
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    model.to(device)


    if args.test:
        test(model, criterion, testloader, device, writer, 0)
    else:
        trainings = {'baseline': train_baseline,
                     'cutmix': train_cutmix,
                     'cutout': train_cutout,
                     'mixup':train_mixup
                     }
        for epoch in range(start_epoch, start_epoch+args.epoch):
            train_stats = trainings[args.model](model, criterion, optimizer, trainloader, device, writer, visualizer, epoch, args)
            test_stats = test(model, criterion, testloader, device, writer, epoch)
            scheduler.step()
            acc = test_stats['acc']
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            save_name = os.path.join(args.logpath, 'latest_epoch.pth')
            torch.save(state, save_name)
            if acc > best_acc:
                print('Saving best model..')
                save_name = os.path.join(args.logpath, 'best_model.pth')
                torch.save(state, save_name)
                best_acc = acc
                
        print('Finish Training...')
        print('Best test acc is {}'.format(best_acc))
