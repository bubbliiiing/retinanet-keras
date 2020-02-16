import numpy as np
import keras
import tensorflow as tf
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

def generate_anchors(base_size=16, ratios=None, scales=None):


    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    # print(np.shape(shifted_anchors))
    # print(shifted_anchors)
    return shifted_anchors
def get_anchors(model):
    border = model.layers[0].output_shape[1]
    features = [model.get_layer(p_name).output_shape for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    shapes = []
    for feature in features:
        shapes.append(feature[1])
    all_anchors = []
    for i in range(5):
        anchors = generate_anchors(AnchorParameters.default.sizes[i])
        shifted_anchors = shift([shapes[i],shapes[i]], AnchorParameters.default.strides[i], anchors)
        all_anchors.append(shifted_anchors)

    all_anchors = np.concatenate(all_anchors,axis=0)
    all_anchors = all_anchors/border
    all_anchors = all_anchors.clip(0,1)
    # print(np.shape(all_anchors))
    return all_anchors