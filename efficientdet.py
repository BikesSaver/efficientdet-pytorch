#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from nets.efficientdet import EfficientDetBackbone
from utils.utils import non_max_suppression, bbox_iou, decodebox, letterbox_image, efficientdet_correct_boxes

Det = 2   # Efficient Det version



image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

def preprocess_input(image):
    image /= 255
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image

def precision(box, pred_num):
    """
    Output num_precision and boxes_precision
    IoU > confidence
    """
    prec = min([pred_num/box*100, box/pred_num*100])
    print('Precision : {:.2f}%'.format(prec))






#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和phi都需要修改！
#--------------------------------------------#
class EfficientDet(object):
    _defaults = {
        #"model_path": 'model_data/efficientdet-d0.pth',
        # 'target_path': '2007_train_bike.txt',
        # "classes_path": 'model_data/coco_classes.txt',
        "confidence": 0.2,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Efficientdet
    #---------------------------------------------------#
    def __init__(self, model_path, det, target, classes, **kwargs):
        self.__dict__.update(self._defaults)
        self.nms_thres = 0.3
        self.target_path = target
        self.classes_path = classes
        self.model_path = model_path
        self.phi = det
        self.class_names = self._get_class()
        self.generate()



    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.net = EfficientDetBackbone(len(self.class_names), self.phi).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path)

        for name, weights in state_dict.items():
            # print(name, weights.size())  可以查看模型中的模型名字和权重维度
            if len(weights.size()) == 2:
                state_dict[name] = weights.squeeze(0)

        self.net.load_state_dict(state_dict)
        self.net = nn.DataParallel(self.net)
        if self.cuda:
            self.net = self.net.cuda()
        print('Finished!')

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, target):

        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (image_sizes[self.phi],image_sizes[self.phi])))
        photo = np.array(crop_img,dtype = np.float32)
        photo = np.transpose(preprocess_input(photo), (2, 0, 1))
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            _, regression, classification, anchors = self.net(images)
            
            regression = decodebox(regression, anchors, images)
            detection = torch.cat([regression, classification], axis=-1)
            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.nms_thres)  #default 0.3
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image
            
        top_index = batch_detections[:,4] > self.confidence
        top_conf = batch_detections[top_index,4]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)



        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(1.5e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // image_sizes[self.phi]

        total_predict = 0
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]     #confidence
            if score > self.confidence:
                total_predict += 1

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        precision(target, total_predict)
        return image

