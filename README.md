## 期中作业（一）：数据增强
课程**DATA620004 神经网络和深度学习**期中作业相关代码

周杭琪 21110980019
樊可 21110980004


## Dependencies
* numpy>=1.16.4
* pytorch==1.8.0
* torchvision==0.9.0
* tensorboard==2.4.1

## Code organization
代码框架主要如下：

* `main.py` 训练&测试的主体文件
* `training.py` CutMix, Cutout, Mixup三种数据增强方式的实现
* `ResNet18.py` 训练时用到的backbone
* `visualization_augmented_samples.py` 在Tensorboard上可视化数据增强的样本


## Run experiments
### 准备数据与预训练模型
* 下载代码至本地

* 下载[CIFAR100数据集](https://pan.baidu.com/s/1l-1VepJNsM4Q7ImkB28Cyg)（提取码：6pjr）至本地，将其解压并移动到`./dataset`（或自定义路径）文件夹中

* 下载[训练好的模型](https://pan.baidu.com/s/1wJR1GNcoFY2KYTDf3rk0yQ)（提取码：br3q）至本地，将其解压移动到`./pretrain`（或自定义路径）文件夹中

### 样本可视化
运行下列代码可以在Tensorboard上可视化数据增强的样本
```
python  visualization_augmented_samples.py --gpu_id 0 --datapath ./dataset
tensorboard  --logdir=results/visualization
```
* --gpu_id：所使用GPU的id
* --datapath：存放测试数据的文件夹路径

### 快速测试
运行下列代码可以迅速测试预训练的模型在CIFAR100测试集上的分类精度
```
python  main.py --gpu_id 0 --test --datapath ./dataset --pretrain ./pretrain/baseline.pth
```
* --gpu_id：所使用GPU的id
* --datapath：存放测试数据的文件夹路径
* --pretrain：预训练模型的路径


### 自定义训练
Baseline
```
python  main.py --gpu_id 0 --model baseline --datapath ./dataset --logpath ./results/baseline
```
CutMix
```
python  main.py --gpu_id 0 --model cutmix --datapath ./dataset --logpath ./results/cutmix
```
Cutout
```
python  main.py --gpu_id 0 --model cutout --datapath ./dataset --logpath ./results/cutout
```
Mixup
```
python  main.py --gpu_id 0 --model mixup --datapath ./dataset --logpath ./results/mixup
```

基础参数：
* --gpu_id：所使用GPU的id
* --model：选择的训练模型，baseline / cutmix / cutout / mixup
* --datapath：存放训练数据的文件夹路径
* --logpath：保存模型的文件夹路径
* --batch_size：批大小
* --seed：训练的随机数种子
* --epoch：训练的总epoch数
* --lr：初始学习率

更多数据增强相关的参数细节可见`main.py`

### 查看Tensorboard记录的实验数据
```
tensorboard  --logdir=results
```
* --logdir：Tensorboard路径
