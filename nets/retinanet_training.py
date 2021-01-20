from random import shuffle

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from utils import backend


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, num_classes+1]
        #   y_pred [batch_size, num_anchor, num_classes]
        #---------------------------------------------------#
        labels         = y_true[:, :, :-1]
        #---------------------------------------------------#
        #   -1 是需要忽略的, 0 是背景, 1 是存在目标
        #---------------------------------------------------#
        anchor_state   = y_true[:, :, -1]  
        classification = y_pred

        # 找出存在目标的先验框
        indices_for_object        = backend.where(keras.backend.equal(anchor_state, 1))
        labels_for_object         = backend.gather_nd(labels, indices_for_object)
        classification_for_object = backend.gather_nd(classification, indices_for_object)

        # 计算每一个先验框应该有的权重
        alpha_factor_for_object = keras.backend.ones_like(labels_for_object) * alpha
        alpha_factor_for_object = backend.where(keras.backend.equal(labels_for_object, 1), alpha_factor_for_object, 1 - alpha_factor_for_object)
        focal_weight_for_object = backend.where(keras.backend.equal(labels_for_object, 1), 1 - classification_for_object, classification_for_object)
        focal_weight_for_object = alpha_factor_for_object * focal_weight_for_object ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss_for_object = focal_weight_for_object * keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back        = backend.where(keras.backend.equal(anchor_state, 0))
        labels_for_back         = backend.gather_nd(labels, indices_for_back)
        classification_for_back = backend.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        alpha_factor_for_back = keras.backend.ones_like(labels_for_back) * (1 - alpha)
        focal_weight_for_back = classification_for_back
        focal_weight_for_back = alpha_factor_for_back * focal_weight_for_back ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss_for_back = focal_weight_for_back * keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object)
        cls_loss_for_back = keras.backend.sum(cls_loss_for_back)
        loss = (cls_loss_for_object + cls_loss_for_back) / normalizer
        return loss
    return _focal

def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2
    def _smooth_l1(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 4+1]
        #   y_pred [batch_size, num_anchor, 4]
        #---------------------------------------------------#
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找出存在目标的先验框
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # 计算smooth L1损失
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 将所获得的loss除上正样本的数量
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer / 4
    return _smooth_l1

class Generator(object):
    def __init__(self, bbox_util,batch_size,
                 train_lines, val_lines, image_size,num_classes,
                 ):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.train_batches = len(train_lines)
        self.val_batches = len(val_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)

        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data

    def generate(self, train=True):
        while True:
            if train:
                # 打乱
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
            inputs = []
            target0 = []
            target1 = []
            n = len(lines)
            for i in range(len(lines)):
                img,y = self.get_random_data(lines[i], self.image_size[0:2], random=train)
                i = (i+1) % n
                
                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]
                    one_hot_label = np.eye(self.num_classes)[np.array(y[:,4],np.int32)]
                    y = np.concatenate([boxes, one_hot_label],axis=-1)

                assignment = self.bbox_util.assign_boxes(y)
                regression = assignment[:,:5]
                classification = assignment[:,5:]

                inputs.append(img)        
                target0.append(np.reshape(regression,[-1,5]))
                target1.append(np.reshape(classification,[-1,self.num_classes+1]))
                if len(target0) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = [np.array(target0,dtype=np.float32),np.array(target1,dtype=np.float32)]
                    inputs = []
                    target0 = []
                    target1 = []
                    yield preprocess_input(tmp_inp), tmp_targets