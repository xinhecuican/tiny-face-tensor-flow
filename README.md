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

rsod数据集，运行程序自动下载

# 运行方法

* make: 直接运行
* make resume: 使用上次训练获得的权重运行，需要修改makefile
* make evaluate: 获得评估文件
* make evalutation: 获得评估结果

> 如果受内存限制无法运行可以调节batch_size和workers参数
