## 2021广东工业智造创新大赛
### 智能算法赛：瓷砖表面瑕疵质检
![jpeg](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/160860612915292251608606125216.jpeg)

https://tianchi.aliyun.com/competition/entrance/531846/introduction

### 系统依赖
- 硬件 联想拯救者R7000P游戏本
- 系统 Windows10
- python 3.8
### 思路概述
- 将原标注修改成yolov5格式
- 按标注位置将原图切割成320*320样本，并用yolo重头训练，epoch 100
- 因6g显存限制，yolo使用了s版本，并修改了yaml文档增加了2层SElayer实现通道注意力机制,增加后模型收敛速度明显加快
- 按标注位置将原图切割成1280*1280样本，并用320*320获得的权重重新训练yolo，epoch 100
- 应用1280*1280样本训练的yolos模型预测并生成最终结果

### 效果
- 初赛B榜 143/4432

### 文件夹说明
```
project
	|--README.md            # 解决方案及算法介绍文件
	|--requirements.txt     # 硬件介绍及Python环境依赖
	|--tcdata
	|--user_data
		|--round1_testA_sliced_all_320         # 适用于yolo的切割图像和标签文件夹（训练）
			|--images
            |--labels
        |--round1_testA_sliced_all_1280        # 适用于yolo的切割图像和标签文件夹（训练）
			|--images
            |--labels
        |--round1_testB_sliced_all_1280        # 适用于yolo的切割图像和标签文件夹（预测）
			|--images
            |--labels
	|--prediction_result
	|--code
		|--01_read_convert_tags.ipynb         # 标签转换格式
		|--02_2_try_cut_pic.ipynb             # 预测按标签切图
		|--03_run_yolo_train.ipynb			  # 训练yolo
		|--04_try_cut_target_pic.ipynb		  # 预测目标切图，其多进程版本为  04_multiproc_try_cut_target_pic.py
        |--05_run_yolo_predict.ipynb          # 预测因大量print建议在terminal运行，最终使用train/exp59/weights/last.pt 模型
        |--06_combine_target_labels.ipynb     # 汇总结果生成标签
```
### 修改的模型部分主要如下
#### 实在搞不懂开源协议，yolov5的部分就不上传了
``` python
### in yolov5/models/commom.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```
``` yaml
# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, SELayer,[256,16]], # added by ccjaread
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, SELayer,[512,16]], # added by ccjaread
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
#    [-1, 1, SELayer,[1024,16]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]
```