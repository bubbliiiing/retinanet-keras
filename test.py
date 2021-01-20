#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from nets.retinanet import resnet_retinanet

if __name__ == "__main__":
    model = resnet_retinanet(80)
    model.summary()
