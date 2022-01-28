# 运行环境

joblib==1.0.1

matplotlib==3.3.4

numpy==1.20.1

opencv_python==4.5.4.58

Pillow==9.0.0

prefetch_generator==1.0.1

pyclust==0.2.0

pyclustering==0.10.1.2

pympler==1.0.1

scipy==1.6.2

torch==1.10.1

torchvision==0.11.2

tqdm==4.59.0

# 数据集下载

[train set](https://share.weiyun.com/5WjCBWV)

[val set](https://share.weiyun.com/5ot9Qv1)

[test set](https://share.weiyun.com/5vSUomP)

[anotation](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip)

然后将下载下来的文件置于data文件夹下

# 运行方法

* make: 直接运行
* make resume: 使用上次训练获得的权重运行，需要修改makefile
* make evaluate: 获得评估文件
* make evalutation: 获得评估结果

预训练权重选择： 调节CHECKPOINT，权重位于weights文件夹中

> 如果受内存限制无法运行可以调节batch_size和workers参数

# 关于分支

* master: 添加边缘提取
* new_dataset: 使用新的数据集
* master/nms : nms部分

