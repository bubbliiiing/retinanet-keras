import tensorflow as tf
from keras import backend as K


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
        indices        = tf.where(K.not_equal(anchor_state, -1))
        labels         = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # 计算每一个先验框应该有的权重
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

        # 标准化，实际上是正样本的数量
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)
        
        # 将所获得的loss除上正样本的数量
        loss = K.sum(cls_loss) / normalizer
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
        indices           = tf.where(K.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算smooth L1损失
        regression_diff = regression - regression_target
        regression_diff = K.abs(regression_diff)
        regression_loss = tf.where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 将所获得的loss除上正样本的数量
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())
        return K.sum(regression_loss) / normalizer / 4
    return _smooth_l1
