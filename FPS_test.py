import os
import time

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image
from tqdm import tqdm

from retinanet import Retinanet
from utils.utils import BBoxUtility, letterbox_image, retinanet_correct_boxes


'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-Retinanet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''
class FPS_Retinanet(Retinanet):
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = letterbox_image(image, [self.model_image_size[0],self.model_image_size[1]])
        photo = np.array(crop_img, dtype = np.float64)
        #---------------------------------------------------------#
        #   归一化并添加上batch_size维度
        #---------------------------------------------------------#
        photo = np.reshape(preprocess_input(photo),[1,self.model_image_size[0],self.model_image_size[1],self.model_image_size[2]])

        preds = self.retinanet_model.predict(photo)
        results = self.bbox_util.detection_out(preds,self.prior,confidence_threshold=self.confidence)
        if len(results[0])>0:
            results = np.array(results)
            det_label = results[0][:, 5]
            det_conf = results[0][:, 4]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
            boxes = retinanet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        t1 = time.time()
        for _ in range(test_interval):
            preds = self.retinanet_model.predict(photo)
            results = self.bbox_util.detection_out(preds,self.prior,confidence_threshold=self.confidence)
            if len(results[0])>0:
                results = np.array(results)
                det_label = results[0][:, 5]
                det_conf = results[0][:, 4]
                det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
                boxes = retinanet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
retinanet = FPS_Retinanet()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = retinanet.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
