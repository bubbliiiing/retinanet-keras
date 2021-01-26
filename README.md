## Retinanet：目标检测模型在Keras当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| COCO-Train2017 | [resnet50_coco_best_v2.1.0.h5](https://github.com/bubbliiiing/retinanet-keras/releases/download/v1.0/resnet50_coco_best_v2.1.0.h5) | COCO-Val2017 | 1333x1333 | 32.2 | 53.2 

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 文件下载
训练所需的resnet50_coco_best_v2.1.0.h5可以在百度云下载。  
链接: https://pan.baidu.com/s/1qS7_QozWsdpjYTcnMKG50w  
提取码: i8w8  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载resnet50_coco_best_v2.1.0.h5，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```  
2、利用video.py可进行摄像头检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在retinanet.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": 'model_data/resnet50_coco_best_v2.1.0.h5',
    "classes_path": 'model_data/coco_classes.txt',
    "model_image_size" : (600, 600, 3),
    "confidence": 0.5,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```  
4. 利用video.py可进行摄像头检测。

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2frcnn.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类**，示例如下：   
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8. 将train.py的NUM_CLASSSES修改成所需要分的类的个数（无需+1），运行train.py即可开始训练。

## mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

## Reference
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/kuhung/SSD_keras  
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  

