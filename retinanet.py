import colorsys
import os
import pickle

import cv2
import keras
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont

import nets.retinanet as retinanet
from utils.anchors import get_anchors
from utils.utils import BBoxUtility, letterbox_image, retinanet_correct_boxes

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class Retinanet(object):
    _defaults = {
        "model_path"        : 'model_data/resnet50_coco_best_v2.1.0.h5',
        "classes_path"      : 'model_data/coco_classes.txt',
        "model_image_size"  : (600, 600, 3),
        "confidence"        : 0.5,
        "iou"               : 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinanet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.generate()
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.iou)
        self.prior = self._get_prior()
        
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
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_prior(self):
        data = get_anchors(self.retinanet_model)
        return data

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        #------------------#
        #   计算种类数量
        #------------------#
        self.num_classes = len(self.class_names)

        #------------------#
        #   载入模型
        #------------------#
        self.retinanet_model = retinanet.resnet_retinanet(self.num_classes, self.model_image_size)
        self.retinanet_model.load_weights(self.model_path, by_name=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

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
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = letterbox_image(image, [self.model_image_size[1],self.model_image_size[0]])
        photo = np.array(crop_img, dtype = np.float64)
        #---------------------------------------------------------#
        #   归一化并添加上batch_size维度
        #---------------------------------------------------------#
        photo = np.reshape(preprocess_input(photo),[1,self.model_image_size[0],self.model_image_size[1],self.model_image_size[2]])

        #---------------------------------------------------------#
        #   传入网络当中进行预测
        #---------------------------------------------------------#
        preds = self.retinanet_model.predict(photo)

        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results = self.bbox_util.detection_out(preds,self.prior,confidence_threshold=self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if len(results[0])<=0:
            return image
        results = np.array(results)

        det_label = results[0][:, 5]
        det_conf = results[0][:, 4]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]
        #-----------------------------------------------------------#
        #   筛选出其中得分高于confidence的框 
        #-----------------------------------------------------------#
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        #-----------------------------------------------------------#
        #   去掉灰条部分
        #-----------------------------------------------------------#
        boxes = retinanet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

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
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()
