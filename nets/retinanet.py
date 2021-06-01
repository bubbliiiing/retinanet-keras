import keras
import keras.layers
from utils.anchors import AnchorParameters
from utils.utils import PriorProbability

from nets.layers import UpsampleLike
from nets.resnet import ResNet50


#-----------------------------------------#
#   Retinahead 获得回归预测结果
#   所有特征层共用一个Retinahead
#-----------------------------------------#
def make_last_layer_loc(num_classes,num_anchors,pyramid_feature_size=256):
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size)) 
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }
    outputs = inputs
    #-----------------------------------------#
    #   进行四次卷积，通道数均为256
    #-----------------------------------------#
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=256,activation='relu',name='pyramid_regression_{}'.format(i),**options)(outputs)
    #-----------------------------------------#
    #   获得回归预测结果，并进行reshape
    #-----------------------------------------#
    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    regression = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)
    #-----------------------------------------#
    #   构建成一个模型
    #-----------------------------------------#
    regression_model = keras.models.Model(inputs=inputs, outputs=regression, name="regression_submodel")
    return regression_model

#-----------------------------------------#
#   Retinahead 获得分类预测结果
#   所有特征层共用一个Retinahead
#-----------------------------------------#
def make_last_layer_cls(num_classes, num_anchors, pyramid_feature_size=256):
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }
    outputs = inputs
    #-----------------------------------------#
    #   进行四次卷积，通道数均为256
    #-----------------------------------------#
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=256, activation='relu', name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None), bias_initializer='zeros', **options)(outputs)
    #-----------------------------------------#
    #   获得分类预测结果，并进行reshape
    #-----------------------------------------#
    outputs = keras.layers.Conv2D(filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=0.01),
        name='pyramid_classification'.format(),
        **options
    )(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    #-----------------------------------------#
    #   为了转换成概率，使用sigmoid激活函数
    #-----------------------------------------#
    classification = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    #-----------------------------------------#
    #   构建成一个模型
    #-----------------------------------------#
    classification_model = keras.models.Model(inputs=inputs, outputs=classification, name="classification_submodel")
    return classification_model


def resnet_retinanet(num_classes, input_shape=(600, 600, 3), name='retinanet'):
    inputs = keras.layers.Input(shape=input_shape)
    resnet = ResNet50(inputs)
    #-----------------------------------------#
    #   取出三个有效特征层，分别是C3、C4、C5
    #   C3     75,75,512
    #   C4     38,38,1024
    #   C5     19,19,2048
    #-----------------------------------------#
    C3, C4, C5 = resnet.outputs[1:]

    # 75,75,512 -> 75,75,256
    P3              = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    # 38,38,1024 -> 38,38,256
    P4              = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    # 19,19,2048 -> 19,19,256
    P5              = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)

    # 19,19,256 -> 38,38,256
    P5_upsampled    = UpsampleLike(name='P5_upsampled')([P5, P4])
    # 38,38,256 + 38,38,256 -> 38,38,256
    P4              = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    # 38,38,256 -> 75,75,256
    P4_upsampled    = UpsampleLike(name='P4_upsampled')([P4, P3])
    # 75,75,256 + 75,75,256 -> 75,75,256
    P3              = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])

    # 75,75,256 -> 75,75,256
    P3              = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    # 38,38,256 -> 38,38,256
    P4              = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    # 19,19,256 -> 19,19,256
    P5              = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # 19,19,2048 -> 10,10,256
    P6              = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6')(C5)
    P7              = keras.layers.Activation('relu', name='C6_relu')(P6)
    # 10,10,256 -> 5,5,256
    P7              = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    features =  [P3, P4, P5, P6, P7]

    num_anchors = AnchorParameters.default.num_anchors()
    regression_model = make_last_layer_loc(num_classes,num_anchors)
    classification_model = make_last_layer_cls(num_classes,num_anchors)

    regressions = []
    classifications = []
    
    #----------------------------------------------------------#
    #   将获取到的P3, P4, P5, P6, P7传入到
    #   Retinahead里面进行预测，获得回归预测结果和分类预测结果
    #   将所有特征层的预测结果进行堆叠
    #----------------------------------------------------------#
    for feature in features:
        regression = regression_model(feature)
        classification = classification_model(feature)
        
        regressions.append(regression)
        classifications.append(classification)

    regressions = keras.layers.Concatenate(axis=1, name="regression")(regressions)
    classifications = keras.layers.Concatenate(axis=1, name="classification")(classifications)

    model = keras.models.Model(inputs, [regressions, classifications], name=name)

    return model
