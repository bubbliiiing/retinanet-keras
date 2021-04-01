import keras
import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam

import nets.retinanet as retinanet
from nets.retinanet_training import Generator, focal, smooth_l1
from utils.anchors import get_anchors
from utils.utils import BBoxUtility

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #----------------------------------------------------#
    #   训练之前一定要修改NUM_CLASSES
    #   修改成所需要区分的类的个数。
    #----------------------------------------------------#
    NUM_CLASSES = 20
    #----------------------------------------------------#
    #   input_shape有可以根据自己的需要进行修改
    #   默认为600,600,3，图像过大的话显存占用较大
    #----------------------------------------------------#
    input_shape = (600, 600, 3)

    model = retinanet.resnet_retinanet(NUM_CLASSES,input_shape)
    #----------------------------------------------------#
    #   获得网络的先验框，并且将先验框输入到BBoxUtility中
    #----------------------------------------------------#
    priors = get_anchors(model)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model_path = "model_data/resnet50_coco_best_v2.1.0.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    for i in range(174):
        model.layers[i].trainable = False

    if True:
        Init_epoch = 0
        Freeze_epoch = 50
        BATCH_SIZE = 8
        learning_rate_base = 1e-4

        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]),NUM_CLASSES)

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': focal()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base, clipnorm=1e-2)
        )
        model.fit_generator(    gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=Freeze_epoch, 
                verbose=1,
                initial_epoch=Init_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(174):
        model.layers[i].trainable = True

    if True:
        Freeze_epoch = 50
        Epoch = 100
        BATCH_SIZE = 4
        learning_rate_base = 1e-5
        
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]),NUM_CLASSES)

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': focal()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base, clipnorm=1e-2)
        )
        model.fit_generator(    gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=Epoch, 
                verbose=1,
                initial_epoch=Freeze_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
