import os
import time
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from backbone.resnet152_fpn_model import resnet152_fpn_backbone
from network_files.rpn_function import AnchorsGenerator
# from backbone.mobilenetv2_model import MobileNetV2
from draw_box_utils import draw_box, draw_box_easy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # models = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # backbone = resnet50_fpn_backbone()
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


def main(args):
    # get devices
    device = torch.device("cuda:{}".format(str(args.gpu_id)) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create models
    model = create_model(num_classes=args.num_classes+1)

    # load train weights
    train_weights = args.weights
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["models"])
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    for img in os.listdir(args.datapath):
        if img.split(".")[1] not in ['png','jpg']:
            continue
        img_head = img.split(".")[0]
        original_img = Image.open(os.path.join(args.datapath, img))

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time.time()
            predictions = model(img.to(device))[0]
            print("inference+NMS time: {}".format(time.time() - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.8,
                     line_thickness=1)
            plt.imshow(original_img)
            plt.show()
            # 保存预测的图片结果
            save_path = os.path.join(args.logpath, "{}_test_result.jpg".format(img_head))
            original_img.save(save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--gpu_id', type=int, default=0, help='device')

    # 检测目标类别数
    parser.add_argument('--num_classes', type=int, default='20', help='number of classes')
    
    # 存放结果的地方
    parser.add_argument('--logpath', default='./results/res50') 
    # 待测图像路径
    parser.add_argument('--datapath', default= './test_images', help='test images')

    # 训练好的权重文件
    parser.add_argument('--weights', default='./results/res50/last_model.pth', type=str, help='training weights')

    args = parser.parse_args()
    if not os.path.exists(args.logpath):
        os.makedirs(args.logpath)
        
    main(args)
