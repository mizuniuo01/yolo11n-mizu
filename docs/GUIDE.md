# 仓库使用指南

[返回 README](../README.md) · [查看工作流指南](WORKFLOW_GUIDE.md)

本文档用于说明当前仓库的设计思路和使用方式。使用者需要先准备好本仓库所需的运行环境，再从模板复制开始创建自己的工程。环境搭建和通用工程设计见工作流指南。

## 环境前提

当前仓库基于以下环境运行：

| 项目 | 环境 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.12 |
| Python 环境 | 项目 .venv |
| 训练框架 | Ultralytics YOLO11 |
| GPU 加速 | CUDA PyTorch |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| 人工标注工具 | 当前 .venv 中的 LabelImg |
| 量化工具链 | TPU-MLIR Docker |

训练和小模型训练脚本默认读取 weights/yolo11n.pt。

## 目录设计

仓库顶层目录按职责划分：

~~~text
configs/      训练、自动打标、导出和 MaixCAM 配置
datasets/     数据集和数据集模板
scripts/      数据、标注、训练、导出和量化脚本
weights/      本地预训练权重
runs/         训练和导出产物
docs/         项目文档
~~~

数据集使用以下结构：

~~~text
datasets/<project>/
├── data.yaml
├── videos/
├── images/{unlabeled,train,val}/
├── labels/{unlabeled,train,val}/
└── label_model/
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
~~~

建议在运行脚本前准备好完整的数据集目录。datasets/template/ 只提供模板；runs/ 及训练、导出分区通常由脚本自动创建；Docker 量化使用单独的工作目录。

## 数据流设计

- unlabeled：视频抽帧、自动打标和人工修正的中间区域。
- label_model：少量人工标注数据，用于训练自动打标小模型。
- train / val：人工确认后的主模型训练集和验证集。

~~~text
视频 → images/unlabeled → 少量人工标注到 label_model
     → 小模型训练 → 生成 labels/unlabeled → 人工修正
     → 划分到主模型 train/val → 主模型训练与验证
~~~

自动生成的标签必须经过人工修正后再进入主模型数据集。

## 复制模板创建工程

使用当前仓库时，复制以下模板目录：

~~~text
datasets/template/           → datasets/<project>/
configs/train/template/      → configs/train/<project>/
configs/auto_label/template/ → configs/auto_label/<project>/
configs/export/template/     → configs/export/<project>/
~~~

然后修改主数据集和 label_model 的 data.yaml、数据集路径、训练/验证路径、类别名称和类别顺序，并确认 weights/yolo11n.pt 存在。

主数据集配置的关键内容：

~~~yaml
path: ./datasets/<project>
train: images/train
val: images/val
names:
  0: classname
~~~

label_model/data.yaml 的 path 指向 ./datasets/<project>/label_model。

## 配置和实验规则

训练、自动打标和导出脚本通过顶部常量选择工程：

~~~python
PROJECT_NAME = "template"
CONFIG_NAME = "train.yaml"
~~~

默认配置文件分别为 train.yaml、export.yaml 和 maixcam.yaml。需要多套参数时，只修改 CONFIG_NAME，例如 train01.yaml。YAML 不保存工程名和实验名；实验目录由脚本内部按 exp01、exp02、exp03 顺序生成，不生成 exp01-2。

## 模块使用方法

下面每个模块都按“设计思路、修改配置、运行命令、输入和输出”说明。

### 视频抽帧

设计：将 videos/ 中的视频转换为统一编号的未标注图片，并在开始前显示预计数量。

修改 scripts/data/capture_video.py 顶部的 DATASET_NAME、FRAMES_PER_SECOND 和覆盖策略。

~~~powershell
python -u scripts/data/capture_video.py
~~~

输入为 datasets/<project>/videos/，输出为 datasets/<project>/images/unlabeled/。

### 人工标注

设计：统一启动 LabelImg，根据目标选择未标注数据、小模型数据或主模型数据；类别从对应 data.yaml 读取，classes.txt 自动写入标签目录。

修改 scripts/label/start_labeling.py 顶部的 DATASET_NAME、LABEL_TARGET、LABEL_MODEL_SPLIT 和 MAIN_MODEL_SPLIT。LABEL_TARGET 支持 unlabeled、label_model、main_model。

~~~powershell
python -u scripts/label/start_labeling.py
~~~

输出为对应 labels/ 目录中的 YOLO 标签和 classes.txt。LabelImg 的 PyQt5 兼容补丁由脚本自动处理。

### 自动打标小模型训练

设计：使用少量人工标注数据训练辅助模型，不作为最终部署模型；训练前检查 train/val 图片与标签配对。

修改 scripts/train/train_auto_label.py 顶部的 PROJECT_NAME 和 CONFIG_NAME，并确认配置包含 data: datasets/<project>/label_model/data.yaml 和 weights: weights/yolo11n.pt。

~~~powershell
python -u scripts/train/train_auto_label.py
~~~

输出为 runs/auto_label/<project>/expXX/weights/best.pt。

### 自动打标和人工修正

设计：自动选择同一工程下最新可用的小模型，批量生成 unlabeled 标签，之后重新使用 LabelImg 人工修正。

修改 scripts/label/auto_label.py 顶部的 PROJECT_NAME 和 CONFIG_NAME。

~~~powershell
python -u scripts/label/auto_label.py
~~~

输入为小模型 best.pt 和 images/unlabeled/，输出为 labels/unlabeled/。脚本显示生成、跳过和空标签数量，不生成预览图。

### 主模型数据集划分

设计：只复制图片和标签均存在的人工确认样本到 train/val，保留 unlabeled 源文件。

修改 scripts/data/split_dataset.py 顶部的 DATASET_NAME、TRAIN_RATIO 和 RANDOM_SEED。

~~~powershell
python -u scripts/data/split_dataset.py
~~~

输出为主数据集的 images/train、images/val、labels/train 和 labels/val。

### 主模型训练与验证

设计：使用 Ultralytics 训练，并在训练过程中自动使用 val 集验证；实验目录由脚本自动选择。

修改 scripts/train/train.py 顶部的 PROJECT_NAME 和 CONFIG_NAME，并确认 YAML 中的 data、weights、device 和训练参数。

~~~powershell
python -u scripts/train/train.py
~~~

输出为 runs/detect/<project>/expXX/。

### 标准模型导出

设计：保留 Ultralytics 标准模型输出，支持 ONNX 和 TensorRT；按训练实验顺序寻找存在 best.pt 且尚未导出的实验。

修改 scripts/export/export.py 顶部的 PROJECT_NAME 和 CONFIG_NAME，在 export.yaml 中设置 format、imgsz、dynamic 和 simplify。

~~~powershell
python -u scripts/export/export.py
~~~

输出为同名实验目录下的 export_standard.onnx 或 export_standard.engine。

### MaixCAM 专用导出

设计：先生成标准 YOLO11 ONNX，再根据 YOLO11 官方结构截断 Detect 头。保留 DFL box 和 Sigmoid class 两个输出，重命名为 out_box、out_cls。

修改 scripts/export/export_maixcam.py 顶部的 PROJECT_NAME 和 CONFIG_NAME，并确认 maixcam.yaml 中的输入尺寸、官方源节点和输出名称。

~~~powershell
python -u scripts/export/export_maixcam.py
~~~

输出为 runs/export/<project>/expXX/export_maixcam.onnx。

### TPU-MLIR Docker 量化

设计：在 Docker 中隔离 TPU-MLIR 工具链，按 ONNX → MLIR → BF16 cvimodel → 校准表 → INT8 cvimodel 的链路处理。当前方案使用 320 × 224 输入、cv181x 处理器和 out_box,out_cls 输出。

Docker 工作目录需要放置：

~~~text
convert_yolo11_to_cvimodel.sh
export_maixcam.onnx
test.jpg
images/
~~~

容器内运行：

~~~bash
chmod +x convert_yolo11_to_cvimodel.sh
./convert_yolo11_to_cvimodel.sh
~~~

输入为 MaixCAM ONNX、测试图片和实际部署场景校准图片，最终输出 INT8 .cvimodel。

### .mud 和 .cvimodel 文件准备

设计：.mud 描述模型类型、输入格式、类别名称和对应 .cvimodel 文件名。

模板位于 configs/maixcam/template/yolo11n.mud。确认其中的模型名和类别名，将量化生成的 .cvimodel 改成模板指定的名称，再将两个文件放入 MaixCAM 模型目录。本指南到模型文件准备结束，不展开设备端程序开发和上传。
