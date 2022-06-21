# Watermark-Tracer

一个**基于可视水印检测识别的数字媒体溯源应用系统**，是我的大作业项目，包含这个系统以及一个开源的[**大规模常见水印图像数据集（Large-scale Common Watermark Dataset, LCWD）**](#数据集准备)。

输入一个带有可视水印的图片或视频，系统会检测定位到水印所在的区域，然后将其提取出来，然后借助百度AI开放平台的OCR和logo识别以及Bing搜索引擎，溯源到这个图片或视频的源头。

## 效果

![](https://kamino-img.oss-cn-beijing.aliyuncs.com/20220621141635.png)

![](https://kamino-img.oss-cn-beijing.aliyuncs.com/20220621141720.png)

## 快速开始

### 环境准备

先安装好`Python3.8`和`PyTorch1.7+`。

```bash
# 要Clone到子模块！（可能会比较大，应为包含好几个yolo的模型，避免之后网络不好又要去下载）
git clone https://github.com/Kamino666/watermark-tracer.git --recurse-submodules
# 安装相应的库
pip install -r requirements.txt
```

### 百度开放AI平台API申请*

:warning:不申请API也可以使用除了溯源以外的功能

接着为了使用百度开放AI平台的接口，要去申请两个API（都有免费额度）：

+ [网络图片识别 图片文字识别 图片转文字-百度AI开放平台 (baidu.com)](https://ai.baidu.com/tech/ocr_others/webimage)
+ [品牌logo识别_拍照识别2万多类商品logo-百度AI开放平台 (baidu.com)](https://ai.baidu.com/tech/imagerecognition/logo)

然后在项目目录新建一个`baidu_cfg.json`的文件，将key填入

```json
{
  "ak": "Your API key",
  "sk": "Your Secret key",
}
```

### 运行

```bash
python trace.py
# 一些可选参数
--no_api          # 不使用百度API，不使用溯源功能
--no_tk           # 不使用tkinter，使用matplotlib进行显示
-m <图片视频路径>   # 在命令行指定路径，不指定也可以
```

## 训练新的模型

### 数据集准备

本项目训练的数据集是一个新构建的**大规模常见水印图像数据集（Large-scale Common Watermark Dataset, LCWD）**，可以从[Large-scale Common Watermark Dataset | Kaggle](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset)下载。

数据集的介绍可见我的[论文](https://github.com/Kamino666/watermark-tracer/blob/master/paper/Watermark_Trace.pdf)。

### 训练

根据情况修改`run_train.sh`和`run_detect.sh`中的路径

```bash
cd yolov5
./run_train.sh  # 训练
./run_detect.sh  # 测试
```

更多训练参数可见[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

若要使用新的模型，可在`trace.py`中搜索`torch.hub.load`，更改`path`参数。

## 生成新的数据集

假如对公布的数据集不满意，也可尝试自己调整参数，生成新的数据集，数据集的详细构建方法可见我的[论文](https://github.com/Kamino666/watermark-tracer/blob/master/paper/Watermark_Trace.pdf)，使用以下命令生成：

```bash
cd data
python generator.py \ 
--images_dir <背景图片目录> \ 
--logo_dir <水印logo目录> \ # 可在data/logos查看目录结构
--output_dir <输出目录> \ 
--num_workers <使用的进程数> \ 
--seed <随机种子> \ 
-n <生成的数量> \ 
--test  # 测试用，详情请看代码
```

若要调整水印生成的格式，请参考代码`WatermarkedImageGenerator`类。

此外，在`tools`目录下有连个jupyter notebook可以参考用来划分和可视化数据集。

## 引用

```latex
@misc{Liu2022watermark,
    title={Watermark-Tracer},
    url={https://github.com/Kamino666/watermark-tracer},
    journal={GitHub},
    author={Zihao, Liu},
    year={2022},
    month={6}
} 
```

## Credits

中国传媒大学 信息与通信工程学院 田佳音老师、杨成老师和和于瀛老师（无顺序）

[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

[ImageNet 1000 (mini) | Kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)

[Douban Movie Short Comments Dataset | Kaggle](https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments)

[iconfont-阿里巴巴矢量图标库](https://www.iconfont.cn/)

[百度AI开放平台-全球领先的人工智能服务平台 (baidu.com)](https://ai.baidu.com/)

[zhu733756/searchengine: 元搜索引擎 searchengine 元数据 元搜索 (github.com)](https://github.com/zhu733756/searchengine)

[X-CCS/remove-watermarks: implementation of "On the Effectiveness of Visible Watermarks" (github.com)](https://github.com/X-CCS/remove-watermarks)















