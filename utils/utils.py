import math

import keras
import numpy as np
import tensorflow as tf
from PIL import Image


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def retinanet_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result

class BBoxUtility(object):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,ignore_threshold=0.4,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variance=0.2):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))
        #---------------------------------------------------#
        #   找到处于忽略门限值范围内的先验框
        #---------------------------------------------------#
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        #---------------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #---------------------------------------------------#
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        assigned_priors = self.priors[assign_mask]
        #--------------------------------------------------#
        #   逆向编码，将真实框转化为Retinanet预测结果的格式
        #--------------------------------------------------#
        assigned_priors_w = (assigned_priors[:, 2] - assigned_priors[:, 0])
        assigned_priors_h = (assigned_priors[:, 3] - assigned_priors[:, 1])
                              
        encoded_box[:,0][assign_mask] = (box[0] - assigned_priors[:, 0])/assigned_priors_w/variance
        encoded_box[:,1][assign_mask] = (box[1] - assigned_priors[:, 1])/assigned_priors_h/variance
        encoded_box[:,2][assign_mask] = (box[2] - assigned_priors[:, 2])/assigned_priors_w/variance
        encoded_box[:,3][assign_mask] = (box[3] - assigned_priors[:, 3])/assigned_priors_h/variance

        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes):
        #---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4       的内容为当前先验框是否包含目标
        #   5:-1    的内容为先验框所对应的种类
        #   -1      的内容为当前先验框是否包含目标
        #---------------------------------------------------#
        assignment = np.zeros((self.num_priors, 4 + 1 + self.num_classes + 1))
        assignment[:, 4] = 0.0
        assignment[:, -1] = 0.0
        if len(boxes) == 0:
            return assignment

        #---------------------------------------------------#
        #   对每一个真实框都进行iou计算
        #---------------------------------------------------#
        apply_along_axis_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        #---------------------------------------------------#
        #   在reshape后，获得的ingored_boxes的shape为：
        #   [num_true_box, num_priors, 1] 其中1为iou
        #---------------------------------------------------#
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1
        assignment[:, -1][ignore_iou_mask] = -1

        #---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_priors, 4+1]
        #   4是编码后的结果，1为iou
        #---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        #---------------------------------------------------#
        #   [num_priors]求取每一个先验框重合度最大的真实框
        #---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        #----------------------------------------------------------#
        #   4和-1代表为当前先验框是否包含目标
        #----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 1
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -1][best_iou_mask] = 1

        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variance=0.2):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = mbox_loc[:,0] * prior_width * variance + mbox_priorbox[:, 0]
        decode_bbox_ymin = mbox_loc[:,1] * prior_height * variance + mbox_priorbox[:, 1]
        decode_bbox_xmax = mbox_loc[:,2] * prior_width * variance + mbox_priorbox[:, 2]
        decode_bbox_ymax = mbox_loc[:,3] * prior_height * variance + mbox_priorbox[:, 3]

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                        decode_bbox_ymin[:, None],
                                        decode_bbox_xmax[:, None],
                                        decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, mbox_priorbox, confidence_threshold=0.4):
        #---------------------------------------------------#
        #   预测结果分为两部分，0为回归预测结果
        #   1为分类预测结果
        #---------------------------------------------------#
        mbox_loc = predictions[0]
        mbox_conf = predictions[1]
        
        #------------------------#
        #   获得先验框
        #------------------------#
        mbox_priorbox = mbox_priorbox
        
        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(mbox_loc)):
            #------------------------------------------------#
            #   非极大抑制过程与Retinanet视频中不同
            #   具体过程可参考
            #   https://www.bilibili.com/video/BV1Lz411B7nQ
            #------------------------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)

            bs_class_conf = mbox_conf[i]
            
            class_conf = np.expand_dims(np.max(bs_class_conf, 1),-1)
            class_pred = np.expand_dims(np.argmax(bs_class_conf, 1),-1)
            #--------------------------------#
            #   判断置信度是否大于门限要求
            #--------------------------------#
            conf_mask = (class_conf >= confidence_threshold)[:,0]

            #--------------------------------#
            #   将预测结果进行堆叠
            #--------------------------------#
            detections = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]), 1)
            unique_class = np.unique(detections[:,-1])

            best_box = []
            if len(unique_class) == 0:
                results.append(best_box)
                continue
            #---------------------------------------------------------------#
            #   4、对种类进行循环，
            #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
            #   对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
            #---------------------------------------------------------------#
            for c in unique_class:
                cls_mask = detections[:,-1] == c
                detection = detections[cls_mask]
                scores = detection[:,4]
                #------------------------------------------#
                #   5、根据得分对该种类进行从大到小排序。
                #------------------------------------------#
                arg_sort = np.argsort(scores)[::-1]
                detection = detection[arg_sort]
                while np.shape(detection)[0]>0:
                    #-------------------------------------------------------------------------------------#
                    #   6、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                    #-------------------------------------------------------------------------------------#
                    best_box.append(detection[0])
                    if len(detection) == 1:
                        break
                    ious = iou(best_box[-1],detection[1:])
                    detection = detection[1:][ious<self._nms_thresh]
            results.append(best_box)
        #-----------------------------------------------------------------------------#
        #   获得，在所有预测结果里面，置信度比较高的框
        #   还有，利用先验框和Retinanet的预测结果，处理获得了预测框的位置
        #-----------------------------------------------------------------------------#
        return results

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou
