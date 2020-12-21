import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa


ia.seed(1)


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)
    try:
        bndbox = root.find('object').find('bndbox')
    except:
        return None

    return bndboxlist


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str("%06d" % (str(id) + '.xml'))))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    elem.text = (id + '.JPG')
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(id + '.xml')))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 / 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def Need_Augment(path):
    num_yel = 0 #yellow的个数

    # 获取 XML 文档对象 ElementTree
    tree = ET.parse(path)
    # 获取 XML 文档对象的根结点 Element
    root = tree.getroot()

    # 递归查找所有的 neighbor 子结点
    # 遍历xml文档的第二层
    for child in root:
        for children in child:
            # 第三层节点的标签名称和属性
            if(children.tag == "name"):
                # 判断是否为Yellow，是则加1
                if(children.text == "Yellow"):
                    num_yel = num_yel + 1

    if num_yel >= 2:
        return False
    else:
        return True


if __name__ == "__main__":

    IMG_DIR = "./VOC2007/JPEGImages"
    XML_DIR = "./VOC2007/Annotations"

    AUG_XML_DIR = "./My_Dataset/Annotations"  # 存储增强后的XML文件夹路径 #todo
    AUG_IMG_DIR = "./My_Dataset/JPEGImages"  # 存储增强后的影像文件夹路径  #todo


    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)


    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 1  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 建立lambda表达式

    # 影像增强
    seq = iaa.Sequential([
        iaa.Flipud(1),  # vertically flip 20% of all images
        iaa.Fliplr(1),  # 镜像
        # iaa.Crop(percent=(0.0, 0.1)),
        iaa.SomeOf((1, 2),
                   [
                       # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 1.0)),
                           # iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                           # iaa.MedianBlur(k=(3, 11)),
                       ]),
                       # 锐化处理
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(1.0, 1.0)),
                       # 浮雕效果
                       # iaa.Emboss(alpha=(0, 1.0), strength=(1, 1.3)),
                       # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.7)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.7), direction=(0.0, 1.0)
                           ),
                       ])),

                       # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                       iaa.CoarseDropout((0.03, 0.05), size_percent=(0.01, 0.02)),
                       # 每个像素随机加减-10到10之间的数
                       # iaa.Add((-2, 2)),

                       # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                       # iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)

                   ],
                   random_order=True  # 随机的顺序把这些操作用在图像上
                   ),

        # iaa.Resize({"height": 900, "width": 1600}, interpolation='nearest'),
        iaa.Resize({"height": 450, "width": 800}, interpolation='nearest'),
    ], )

for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:
            print('name: ',name)

            if name.endswith('.csv'):
                continue

            bndbox = read_xml_annotation(XML_DIR, name)
            if bndbox == None:
                continue
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.JPG'), AUG_IMG_DIR)

            # if not Need_Augment(os.path.join(XML_DIR, name)): #判断是否需要增强
            #     print(os.path.join(XML_DIR, name))
            #     continue
            # print("Here")
            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.JPG'))
                # sp = img.size
                img = np.asarray(img)
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]

                path = os.path.join(AUG_IMG_DIR,
                                    str("%06d" % (i+100*epoch) + name[:-4] ) + '.JPG')
                image_auged = bbs.draw_on_image(image_aug, thickness=0)
                Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                           str("%06d" % (i+100*epoch) + name[:-4] ) )
                print(str("%06d"%(i+100*epoch) + name[:-4] ) + '.JPG')
                new_bndbox_list = []