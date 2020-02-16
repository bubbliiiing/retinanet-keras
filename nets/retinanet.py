import keras
import keras.layers

from nets.resnet import ResNet50
from utils.anchors import AnchorParameters
from utils.utils import PriorProbability
from nets import layers

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
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=256,activation='relu',name='pyramid_regression_{}'.format(i),**options)(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    
    regression = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)
    
    regression_model = keras.models.Model(inputs=inputs, outputs=regression, name="regression_submodel")
    return regression_model


def make_last_layer_cls(num_classes,num_anchors,pyramid_feature_size=256):
    
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }
    classification = [] 
    outputs = inputs

    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=256,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=0.01),
        name='pyramid_classification'.format(),
        **options
    )(outputs)

    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    classification = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    classification_model = keras.models.Model(inputs=inputs, outputs=classification, name="classification_submodel")
    return classification_model


def resnet_retinanet(num_classes, inputs=None, num_anchors=None, submodels=None, name='retinanet'):
    if inputs==None:
        inputs = keras.layers.Input(shape=(600, 600, 3))
    else:
        inputs = inputs
    resnet = ResNet50(inputs)
    
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    C3, C4, C5 = resnet.outputs[1:]

    P5           = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    # 38x38x256
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5')(P5)
    
    # 38x38x256
    P4           = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])

    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # 75x75x256
    P3 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    P6 = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    features =  [P3, P4, P5, P6, P7]

    regression_model = make_last_layer_loc(num_classes,num_anchors)
    classification_model = make_last_layer_cls(num_classes,num_anchors)

    regressions = []
    classifications = []
    for feature in features:
        regression = regression_model(feature)
        classification = classification_model(feature)
        
        regressions.append(regression)
        classifications.append(classification)

    regressions = keras.layers.Concatenate(axis=1, name="regression")(regressions)
    classifications = keras.layers.Concatenate(axis=1, name="classification")(classifications)
    pyramids = [regressions,classifications]

    model = keras.models.Model(inputs=inputs, outputs=pyramids, name=name)

    return model
