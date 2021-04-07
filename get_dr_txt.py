#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import os

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image

from retinanet import Retinanet
from utils.utils import BBoxUtility, letterbox_image, retinanet_correct_boxes
from tqdm import tqdm


'''
这里设置的门限值较低是因为计算map需要用到不同门限条件下的Recall和Precision值。
所以只有保留的框足够多，计算的map才会更精确，详情可以了解map的原理。
计算map时输出的Recall和Precision值指的是门限为0.5时的Recall和Precision值。

此处获得的./input/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。

这里的self.bbox_util._nms_thresh指的是非极大抑制所用到的iou，具体的可以了解非极大抑制的原理，
如果低分框与高分框的iou大于这里设定的self.bbox_util._nms_thresh，那么该低分框将会被剔除。

可能有些同学知道有0.5和0.5:0.95的mAP，这里的self.bbox_util._nms_thresh=0.5不代表mAP0.5。
如果想要设定mAP0.x，比如设定mAP0.75，可以去get_map.py设定MINOVERLAP。
'''
class mAP_Retinanet(Retinanet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        self.bbox_util._nms_thresh = 0.5
        f = open("./input/detection-results/"+image_id+".txt","w") 
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

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

retinanet = mAP_Retinanet()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # image.save("./input/images-optional/"+image_id+".jpg")
    retinanet.detect_image(image_id,image)
    # print(image_id," done!")
    

print("Conversion completed!")
