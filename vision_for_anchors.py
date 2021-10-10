import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np


def decode_boxes(mbox_loc, mbox_priorbox):
    # 获得先验框的宽与高
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    # 获得先验框的中心点
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    # 真实框距离先验框中心的xy轴偏移情况
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height
    decode_bbox_center_y += prior_center_y
    
    # 真实框的宽与高的求取
    decode_bbox_width = np.exp(mbox_loc[:, 2])
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3])
    decode_bbox_height *= prior_height

    # 获取真实框的左上角与右下角
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    # 真实框的左上角与右下角进行堆叠
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                    decode_bbox_ymin[:, None],
                                    decode_bbox_xmax[:, None],
                                    decode_bbox_ymax[:, None]), axis=-1)

    return decode_bbox

class AnchorBox:
    def __init__(self, ratios, scales):
        self.ratios  = ratios
        self.scales  = scales
        self.num_anchors = len(self.ratios) * len(self.scales)

    def generate_anchors(self, base_size = 16):
        # anchors - 9,4
        anchors         = np.zeros((self.num_anchors, 4))
        anchors[:, 2:]  = base_size * np.tile(self.scales, (2, len(self.scales))).T

        # 计算先验框的面积
        areas = anchors[:, 2] * anchors[:, 3]

        # np.repeat(ratios, len(scales))    [0.5 0.5 0.5 1.  1.  1.  2.  2.  2. ]
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = np.sqrt(areas * np.repeat(self.ratios, len(self.scales)))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T 
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors


    def shift(self, shape, stride, anchors):
        # 生成特征层的网格中心
        shift_x = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
        shift_y = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_x = np.reshape(shift_x, [-1])
        shift_y = np.reshape(shift_y, [-1])

        # 将网格中心进行堆叠
        shifts = np.stack([
            shift_x,
            shift_y,
            shift_x,
            shift_y
        ], axis=0)

        shifts            = np.transpose(shifts)
        number_of_anchors = np.shape(anchors)[0]

        k = np.shape(shifts)[0]
        # shifted_anchors   k, 9, 4 -> k * 9, 4
        shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]))
        shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
        
        #-------------------------------#
        #   可视化代码
        #-------------------------------#
        if shape[0]==4:
            fig = plt.figure()
            ax = fig.add_subplot(121)
            plt.ylim(-300,900)
            plt.xlim(-600,600)

            plt.scatter(shift_x,shift_y)
            box_widths = shifted_anchors[:,2]-shifted_anchors[:,0]
            box_heights = shifted_anchors[:,3]-shifted_anchors[:,1]
            
            for i in [108,109,110,111,112,113,114,115,116]:
                rect = plt.Rectangle([shifted_anchors[i, 0],shifted_anchors[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
                ax.add_patch(rect)
            plt.gca().invert_yaxis()

            ax = fig.add_subplot(122)
            plt.ylim(-300,900)
            plt.xlim(-600,600)
            plt.scatter(shift_x,shift_y)
            P7_num_anchors = len(shifted_anchors)
            random_inputs = np.random.uniform(0,1,[P7_num_anchors,4])/10
            after_decode = decode_boxes(random_inputs, shifted_anchors)
            
            box_widths = after_decode[:,2]-after_decode[:,0]
            box_heights = after_decode[:,3]-after_decode[:,1]

            after_decode_center_x = after_decode[:,0]/2+after_decode[:,2]/2
            after_decode_center_y = after_decode[:,1]/2+after_decode[:,3]/2
            plt.scatter(after_decode_center_x[108:116],after_decode_center_y[108:116])

            for i in [108,109,110,111,112,113,114,115,116]:
                rect = plt.Rectangle([after_decode[i, 0],after_decode[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
                ax.add_patch(rect)
            plt.gca().invert_yaxis()
            
            plt.show()
            
        return shifted_anchors

#---------------------------------------------------#
#   用于计算共享特征层的大小
#---------------------------------------------------#
def get_img_output_length(height, width):
    filter_sizes    = [7, 3, 3, 3, 3, 3, 3]
    padding         = [3, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]
    
def get_anchors(input_shape, anchors_size = [32, 64, 128, 256, 512], strides = [8, 16, 32, 64, 128], \
                ratios = [0.5, 1, 2], scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    
    all_anchors = []
    anchor_box  = AnchorBox(ratios, scales)
    for i in range(len(anchors_size)):
        #------------------------------#
        #   先生成每个特征点的9个先验框
        #   anchors     9, 4
        #------------------------------#
        anchors         = anchor_box.generate_anchors(anchors_size[i])
        shifted_anchors = anchor_box.shift([feature_heights[i], feature_widths[i]], strides[i], anchors)
        all_anchors.append(shifted_anchors)

    # 将每个特征层的先验框进行堆叠。
    all_anchors = np.concatenate(all_anchors,axis=0)
    all_anchors = all_anchors / np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    return all_anchors

get_anchors([600, 600])