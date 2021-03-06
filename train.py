# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import efficientdet_dataset_collate, EfficientdetDataset
from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import Generator, FocalLoss
from tqdm import tqdm
from PIL import Image

from functools import wraps
from datetime import datetime


train_type = 0  #  1 for train,0 for predict
Det = 2
Batch_size = 1
# tagfile = '2007_train_bike.txt'
tagfile = '2007_train.txt'
classes_path = 'model_data/voc_classes.txt'         # all tags
# classes_path = 'model_data/coco_classes.txt'  # Bike and Seat

with open('latest_model_path.txt','r') as f:
    latest_model_path = f.read()   #latest trained model path

init_model_path = None# 'logs/Epoch50-Total_Loss0.3600-Val_Loss0.3254-Det0.pth' #'logs/Epoch50-Total_Loss0.8854-Val_Loss1.4888-Det2.pth'
if not init_model_path:
    init_model_path = latest_model_path#"./weights/efficientdet-d{}.pth".format(Det)
    #print('将使用d{}作为初始权重训练'.format(Det))

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True      #确定性？  开了训练很慢
torch.cuda.empty_cache()
loss = 'F'

loss_type = {
    'F' : FocalLoss,
}

# test_loss = RepulsionLoss()

criteria = loss_type[loss]

def _curent_time():
    date = datetime.now()
    return date.strftime("%Y%m%d_%H-%M-%S")


def time_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = datetime.now()
        res = func(*args, **kwargs)
        after = datetime.now()
        print('===time cost: {} costs {}'.format(func.__name__, after - begin))
        return res

    return wrapper


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


# @time_log
def fit_one_epoch(model, optimizer, net, criteria_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_repu_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    torch.cuda.empty_cache()   # clean memory
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            _, regression, classification, anchors = net(images)

            loss, c_loss, r_loss, repu_loss = criteria_loss(classification, regression, anchors, targets, cuda=cuda)
            # rep_loss = test_loss(classification, regression, anchors, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_c_loss += c_loss.item()
            total_repu_loss += repu_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Total Loss' : total_loss / (iteration + 1),
                                'Conf Loss': total_c_loss / (iteration + 1),
                                'Regression Loss': total_r_loss / (iteration + 1),
                                'Repulsion Loss': total_repu_loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'time/s': waste_time})
            pbar.update(1)

            start_time = time.time()
    # net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in
                                   targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                _, regression, classification, anchors = net(images_val)
                loss, c_loss, r_loss, repu_loss = criteria_loss(classification, regression, anchors, targets_val, cuda=cuda)
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    # net.train()
    print('Finish Validation')
    print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    if (epoch+1)%10 == 0:
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f-Det%d.pth' % (
            (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1),Det))
        # latest_model_path: latest trained model path
        latest_model_path = 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f-Det%d.pth' % (
            (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1),Det)
        with open('latest_model_path.txt','w+') as f:
            f.write(latest_model_path)
    return val_loss / (epoch_size_val + 1)


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#


@time_log  # modified
def train():
    # -------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    # -------------------------------------------#

    phi = Det
    print("Det:", phi)
    Cuda = True
    annotation_path = tagfile


    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True
    lr = 2e-3
    Init_Epoch = 0
    Freeze_Epoch = 50
    class_names = get_classes(classes_path)
    print
    num_classes = len(class_names)
    # I have read a paper about data set augmentation, we can try on it and boast our data
    # that offer about 30 ways to modify the pictures
    # <Albumentations : fast and flexible image augmentations>
    # https://github.com/albumentations-team/albumentations
    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    input_shape = (input_sizes[phi], input_sizes[phi])  # TODO, Input picture size need adjust
    # 4000*2250  ->  512*512
    # 500 * 2250/8
    # 创建模型
    model = EfficientDetBackbone(num_classes, phi)

    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#

    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(init_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    efficient_loss = criteria()         # TODO loss: repulsive loss

    # 0.1用于验证，0.9用于训练
    val_split = 0.02  #.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(1010)
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=1e-4)  # adam  SGD  5e-4
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True,
                                                            threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7)

        if Use_Data_Loader:
            train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
            val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, shuffle=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, shuffle=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            val_loss = fit_one_epoch(model, optimizer, net, efficient_loss, epoch, epoch_size, epoch_size_val, gen,
                                     gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step(val_loss)
            # TODO every epoch: precision and recall

    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        lr = lr/10

        Unfreeze_Epoch = Freeze_Epoch * 2

        torch.cuda.empty_cache() #clean memory

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-7)

        if Use_Data_Loader:
            train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
            val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, shuffle=True,  collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, shuffle=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            val_loss = fit_one_epoch(model, optimizer, net, efficient_loss, epoch, epoch_size, epoch_size_val, gen,
                                     gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step(val_loss)



def predict(model_path, Det=0):
    from efficientdet import EfficientDet
    from PIL import Image
    efficientdet = EfficientDet(model_path, Det, tagfile, classes_path)

    #img = "./VOCdevkit/VOC2007/TestImages/bike{}.JPG"  #测试集图片
    img = "./VOCdevkit/VOC2007/JPEGImages/tagbike{}.JPG"       #训练集图片
    targets = open(tagfile, 'r').readlines()

    while True:
        I = input("Input a number:\n")
        img_file = img.format(I)
        target = [i for i in targets if 'tagbike{}.jpg'.format(I) in i]
        target = target[0].split(' ')
        target = len(target)-1
        try:
            image = Image.open(img_file)

        except:
            if I == 'q':
                break
            print('Open Error! Try again!')

        else:
            r_image = efficientdet.detect_image(image, target)
            r_image.show()
            #r_image.save('F:\project\efficientdet-pytorch - Bike\\results\\result_bike{}'.format(I) + '.jpg')


if __name__ == '__main__':
    # model_path = './logs/Epoch50-Total_Loss0.4145-Val_Loss0.3956.pth'  # hardcore, assign to Durbin,has finished
    model_path = latest_model_path
    if train_type:
        train()
    else:
        #reducing learning rate of group 0 to 6.2500e-06.
        predict(model_path, Det)
