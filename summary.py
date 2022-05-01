#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.retinanet import resnet_retinanet
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape = [600, 600, 3]
    num_classes = 80

    model = resnet_retinanet(input_shape, num_classes)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------------------#
    #   由于retinanet-keras里面某些层以Model的方式存在。
    #   无法获取详细的input_shape和output_shape。
    #   因此无法正确计算FLOPs，可参考pytorch版本
    #--------------------------------------------------------#
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
