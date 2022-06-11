import os

import torch
import torch.nn as nn

import argparse
import my_transforms as transforms
#import torchvision.transforms as transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import train_eval_utils as utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import random
def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



class Visualization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)


def create_model(num_classes:int, device, pretrain = 'none'):

    backbone = resnet50_fpn_backbone(pretrain)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    # 载入预训练模型权重
    if pretrain == 'coco':
        print('load pretrained model from COCO')
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        

    return model


def main(args):
    set_seed(args.seed)
    writer = SummaryWriter(os.path.join(args.logpath, 'tensorboard'))
    visualizer = Visualization()
    device = torch.device("cuda:{}".format(str(args.gpu_id)) if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.datapath

    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2007 -> ImageSets -> Main -> train.txt
    train_data_set = VOCDataSet(VOC_root, data_transform["train"], args.train_txt, args.json_name)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = DataLoader(train_data_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=nw,
                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_data_set = VOCDataSet(VOC_root, data_transform["val"], args.val_txt,args.json_name)
    val_data_set_loader = DataLoader(val_data_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=nw,
                                  collate_fn=train_data_set.collate_fn)

    # create models num_classes equal background + 20 classes
    # print(args.num_classes)

    model = create_model(num_classes=len(train_data_set._classes), device=device, pretrain = args.pretrain)
    # print(models)

    model.to(device)

    # print(model)
    # exit()

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['models'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_mAP = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, writer=writer,visualizer=visualizer,
                              train_loss=train_loss, train_lr=learning_rate,
                              print_freq=50, warmup=True)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, epoch, device, writer, mAP_list=val_mAP)

        # save weights
        save_files = {
            'models': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        save_name = os.path.join(args.logpath, 'last_model.pth')
        torch.save(save_files, save_name)

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        from plot_curve import plot_map
        plot_map(val_mAP)



if __name__ == "__main__":
    version = torch.version.__version__[:5]  # example: 1.6.0
    # 因为使用的官方的混合精度训练是1.6.0后才支持的，所以必须大于等于1.6.0
    #if version < "1.6.0":
    #    raise EnvironmentError("pytorch version must be 1.6.0 or above")

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--seed', type=int, default=4396)
    parser.add_argument('--gpu_id', type=int, default=0)
    #parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--datapath', default= '../dataset') #存放VOC2007的路径
    # 文件保存地址
    parser.add_argument('--logpath', default='./results/res50') #希望保存的logs路径，包括tensorboard
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 预训练方式
    parser.add_argument('--pretrain', default='none', choices = ['none', 'coco','imagenet'], type=str)
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    parser.add_argument('--json_name', default="pascal_voc_classes.json", type=str, metavar='N',
                        help='the num of classes')
    parser.add_argument('--train_txt', default="train.txt", type=str, metavar='N')
    parser.add_argument('--val_txt', default="val.txt", type=str, metavar='N')

    args = parser.parse_args()
    print(args)
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.logpath):
        os.makedirs(args.logpath)

    main(args)
