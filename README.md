## 期末作业（三）：Transformer
课程**DATA620004 神经网络和深度学习**期末作业相关代码

周杭琪 21110980019
樊可 21110980004


## Dependencies
* numpy>=1.16.4
* pytorch==1.8.0
* torchvision==0.9.0
* tensorboard==2.4.1

## Code organization
代码框架主要如下：

* `main-vit.py` 训练&测试的主体文件
* `training.py` CutMix, Cutout, Mixup三种数据增强方式的实现
* `visualization_augmented_samples.py` 在Tensorboard上可视化数据增强的样本


## Run experiments
### 准备数据与预训练模型
* 下载代码至本地

* 下载[CIFAR100数据集](https://pan.baidu.com/s/1l-1VepJNsM4Q7ImkB28Cyg)（提取码：6pjr）至本地，将其解压并移动到`./dataset`（或自定义路径）文件夹中

* 下载[训练好的模型](https://pan.baidu.com/s/12C8X9LbfIolMB0NLCalIPQ?pwd=279b)（提取码：279b）至本地，将其解压移动到`./pretrain`（或自定义路径）文件夹中
* 
--来自百度网盘超级会员v6的分享
### 快速测试
运行下列代码可以迅速测试预训练的模型在CIFAR100测试集上的分类精度
```
python  main-vit.py --gpu_id 0 --test --datapath ./dataset --pretrain ./pretrain/latest_epoch.pth
```
* --gpu_id：所使用GPU的id
* --datapath：存放测试数据的文件夹路径
* --pretrain：预训练模型的路径


### 自定义训练
Baseline
```
python  main-vit.py --gpu_id 0 --model baseline --datapath ./dataset --logpath ./results/baseline --lr 1e-2 --optimizer SGD
```
Mixup
```
python  main-vit.py --gpu_id 0 --model mixup --datapath ./dataset --logpath ./results/mixup --lr 1e-2 --optimizer SGD
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
* --optimizer：选择的优化器, SGD/Adam
更多数据增强相关的参数细节可见`main-vit.py`

### 查看Tensorboard记录的实验数据
```
tensorboard  --logdir=results
```
* --logdir：Tensorboard路径
