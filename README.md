## 期末作业（二）：预训练与Faster R-CNN
课程**DATA620004 神经网络和深度学习**期末作业相关代码

周杭琪 21110980019
樊可 21110980004


## Dependencies
* lxml==4.6.2
* matplotlib==3.2.1
* numpy==1.17.0
* tqdm==4.42.1
* torch==1.6.0
* torchvision==0.7.0
* pycocotools==2.0.0
* Pillow==8.0.1

## Code organization
代码框架主要如下：

* `train.py` 训练的主体文件
* `validation.py` 在验证集上测试的主体文件
* `draw_box_utils.py`, `predict.py` 用模型对输入的测试图像直接进行目标检测，并以图像的形式保存检测结果
* `my_dataset.py`, `my_transform.py` VOC2007数据集的读取与预处理
* `backbone` 训练时用到的backbone，主要是resnet系列
* `network_files` Faster R-CNN模型，结构与pytorch官方一致
* `train_utils` 训练与验证的主要代码
* `pascal_voc_classes.json` 保存VOC2007类别，无需改动


## Run experiments
### 准备数据与预训练模型
* 下载代码至本地 

* 下载[VOC2007数据集](https://pan.baidu.com/s/1EM81nuQESEak9fdD-K3MeQ)（提取码：nuis）至本地，将其解压并移动到`./dataset`（或自定义路径）文件夹中
    * 参考[FASTER RCNN](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)，解压好的文件结构应为：
```
$./dataset/VOCdevkit/                           # development kit
$./dataset/VOCdevkit/VOCcode/                   # VOC utility code
$./dataset/VOCdevkit/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```
* 下载[训练好的模型](https://pan.baidu.com/s/1XTkkwusbHk4SdW8iv05yDw)（提取码：lp4j）至本地，将其解压移动到`./pretrain`（或自定义路径）文件夹中

### 目标检测
* 选择不在VOC2007数据集中，但拥有其类别的三张测试图像： [Test Images](https://pan.baidu.com/s/1qdi8z6gTiALvh7SF-_dgqg)（提取码：ys46），将其下载至本地，解压并移动到`./dataset`（或自定义路径）文件夹中

运行下列代码可以使用训练好的模型直接在测试图像上进行目标检测
```
python predict.py --gpu_id 0 --weights ./pretrain/last_model_coco.pth --datapath ./dataset/test_images --logpath ./results/res50_coco
python predict.py --gpu_id 0 --weights ./pretrain/last_model_imagenet.pth --datapath ./dataset/test_images --logpath ./results/res50_imagenet
python predict.py --gpu_id 0 --weights ./pretrain/last_model_none.pth --datapath ./dataset/test_images --logpath ./results/res50_none
```
* --gpu_id：所使用GPU的id
* --weights：训练好的模型保存路径
* --datapath：存放测试数据的文件夹路径
* --logpath：保存输出检测结果图像的路径

### 精度测试
运行下列代码可以迅速测试训练好的模型在VOC2007验证集上的分类精度
```
python validation.py --gpu_id 0 --weights ./pretrain/last_model_coco.pth --datapath ./dataset
python validation.py --gpu_id 0 --weights ./pretrain/last_model_imagenet.pth --datapath ./dataset
python validation.py --gpu_id 0 --weights ./pretrain/last_model_none.pth --datapath ./dataset
```
* --gpu_id：所使用GPU的id
* --weights：训练好的模型保存路径
* --datapath：VOC2007数据集的根目录（存放`VOCdevkit`的文件夹路径）


### 训练模型：探究预训练的影响
* 下载在ImageNet上训练的Resnet50[Faster R-CNN ImageNet backbone](https://pan.baidu.com)（提取码：）至本地，将其移动到`./baseline`文件夹中
* 下载pytorch官方提供的[Faster R-CNN COCO训练模型](https://pan.baidu.com/s/1Z6dbTA02mODOtDyIdaa-7A)（提取码：eudr）至本地，将其移动到`./baseline`文件夹中

运行下列代码可以训练无预训练的Faster R-CNN + Resnet50
```
python train.py --gpu_id 0 --pretrain none --epochs 100 --logpath ./results/res50_none --datapath ./dataset
```
运行下列代码可以训练读取在ImageNet上预训练Resnet50作为backbone初始参数的Faster R-CNN + Resnet50
```
python train.py --gpu_id 0 --pretrain imagenet --epochs 100 --logpath ./results/res50_imagenet --datapath ./dataset
```
运行下列代码可以训练读取在COCO上预训练的Faster R-CNN作为整个模型初始参数的Faster R-CNN + Resnet50
```
python train.py --gpu_id 0 --pretrain coco --epochs 100 --logpath ./results/res50_coco --datapath ./dataset
```
基础参数：
* --pretrain：预训练的方式
* --gpu_id：所使用GPU的id
* --datapath：存放训练数据的文件夹路径
* --logpath：保存模型的文件夹路径
* --batch_size：批大小
* --seed：训练的随机数种子
* --epoch：训练的总epoch数
* --lr：初始学习率

查看Tensorboard记录的实验数据
```
tensorboard  --logdir=results
```



