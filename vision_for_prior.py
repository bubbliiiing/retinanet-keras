import keras
import matplotlib.pyplot as plt
import numpy as np


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def generate_anchors(base_size=16):
    ratios = AnchorParameters.default.ratios
    scales = AnchorParameters.default.scales
    # num_anchors = 9
    num_anchors = len(ratios) * len(scales)
    # anchors - 9,4
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # 计算先验框的面积
    areas = anchors[:, 2] * anchors[:, 3]

    # np.repeat(ratios, len(scales))    [0.5 0.5 0.5 1.  1.  1.  2.  2.  2. ]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = np.sqrt(areas * np.repeat(ratios, len(scales)))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T 
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    print(anchors)
    return anchors

def shift(shape, stride, anchors):
    # [0-74]
    # [0.5-74.5]
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
    # shifted_anchors   k, 9, 4 -> k*9, 4
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]))
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    
    #-------------------------------#
    #   可视化代码
    #-------------------------------#
    if shape[0]==5:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylim(-300,900)
        plt.xlim(-300,900)

        plt.scatter(shift_x,shift_y)
        box_widths = shifted_anchors[:,2]-shifted_anchors[:,0]
        box_heights = shifted_anchors[:,3]-shifted_anchors[:,1]
        
        for i in [108,109,110,111,112,113,114,115,116]:
            rect = plt.Rectangle([shifted_anchors[i, 0],shifted_anchors[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
            ax.add_patch(rect)
        
        plt.show()
    return shifted_anchors

border = 600
shape = [border,border]

a = [75,38,19,10,5]
# print(a)
all_anchors = []
for i in range(5):
    anchors = generate_anchors(AnchorParameters.default.sizes[i])
    shifted_anchors = shift([a[i],a[i]], AnchorParameters.default.strides[i], anchors)
    all_anchors.append(shifted_anchors)

all_anchors = np.concatenate(all_anchors,axis=0)
all_anchors = all_anchors/border
all_anchors = all_anchors.clip(0,1)

# print(all_anchors)
# with open('model_data/prior.pkl', 'wb') as f:
#     pickle.dump(all_anchors, f)

# with open('model_data/prior.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(np.shape(all_anchors))
