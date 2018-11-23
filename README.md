## Action recognition for depth video using multi-view dynamic images
this is the implementation of the paper accepted by Science Information

the code is consists of there pattern: 

1. multi-view dynamic images generation. 
2. multi-view CNN training.
3. faster rcnn based human motion detection 


## requirement
#### data download
the Dataset (such as [NTU RGBD](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)) and the pretrained model [imagenet-vgg-f](http://www.vlfeat.org/matconvnet/pretrained/)

#### environmet
libnear library for matlab [LIBLEANER](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) is used to *generate the dynamic images.*\

matconvnet-1.0-bata23 [MatConvNet](http://www.vlfeat.org/matconvnet/download/) is used for the *CNN training* stage. 
## usage
For the NTU RGB+D dataset, the experiment can be implemented by the following steps
For multi-view dynamic images generation,
run MVDI_generation/dynamic_multi_general_NTU.m


## contact



Yancheng Wang    : yancheng_wang@hust.edu.cn

Yang Xiao
