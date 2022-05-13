## 期中作业（二）：目标检测 Faster R-CNN
课程**DATA620004 神经网络和深度学习**期中作业相关代码

周杭琪 21110980019
樊可 21110980004


## Dependencies
* matplotlib>=3.2.2
* numpy>=1.18.5
* opencv-python>=4.1.2
* Pillow>=7.1.2
* PyYAML>=5.3.1
* requests>=2.23.0
* scipy>=1.4.1
* torch>=1.7.0
* torchvision>=0.8.1
* tqdm>=4.41.0
* tensorboard>=2.4.1
* wandb
* pandas>=1.1.4
* seaborn>=0.11.0
*  albumentations>=1.0.3
*  Cython  # for pycocotools 
*  pycocotools>=2.0  # COCO mAP
* roboflow
* thop 

## Code organization
代码框架主要如下：

* `train.py` 训练的主体文件
* `val.py` 在验证集上测试的主体文件
* `detect.py` 在测试图像上做图的主体文件
* `setup.cfg` 项目层次的配置文件
* `requirements.txt` 需要的库
* `utils` 任务所需的各类模块、函数
* `models` yolo模型
* `data` 训练集、验证集合设置
* `run` 实验结果
* `tf_logs` 训练阶段tensorboard的记录
* `test_image` 用来测试的voc外的图片



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
### 训练与精度测试
### 目标检测
* 选择不在VOC2007数据集中，但拥有其类别的三张测试图像： [Test Images](https://pan.baidu.com/s/1qdi8z6gTiALvh7SF-_dgqg)（提取码：ys46），将其下载至本地，解压并移动到`./dataset`（或自定义路径）文件夹中

运行下列代码可以使用训练好的模型直接在测试图像上进行目标检测
```
python predict.py --gpu_id 0 --weights ./pretrain/last_model.pth --datapath ./dataset/test_images --logpath ./results/res50
```



查看Tensorboard记录的实验数据
```
tensorboard  --logdir=./tf_logs/finetune
```

