<div align="center">

# YOLO11n Object Detection Workflow

<p>
  从数据准备、自动标注到 MaixCAM 部署的 YOLO11n 目标检测工作流
</p>

<p>
  <a href="docs/GUIDE.md">📖 使用指南</a>
  ·
  <a href="docs/WORKFLOW_GUIDE.md">🧭 工作流指南</a>
</p>

<p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch CUDA">
  </a>
  <a href="https://github.com/ultralytics/ultralytics">
    <img src="https://img.shields.io/badge/Ultralytics-YOLO11-111F68" alt="Ultralytics YOLO11">
  </a>
</p>

</div>

## 项目简介

这是一个基于 Ultralytics YOLO11n 的目标检测工作流仓库，覆盖从原始视频、数据集准备、人工标注、自动打标，到模型训练、导出和 MaixCAM 部署的完整流程。

仓库重点提供一套可以直接参考的脚本、配置和目录设计，适合记录个人开发过程，也适合用作其他 YOLO 项目的工程起点。

本仓库面向当前 Windows 开发环境和本机 GPU，不承诺复制后即可在任意电脑上直接运行。具体环境准备、配置修改和脚本运行方式请查看[仓库使用指南](docs/GUIDE.md)。

## 核心能力

| 环节 | 能力 | 脚本或目录 | 主要产物 |
| --- | --- | --- | --- |
| 数据准备 | 视频批量抽帧、数量预估、统一编号 | `scripts/data/capture_video.py` | `images/unlabeled/` |
| 人工标注 | LabelImg 启动、类别读取、YOLO 标签保存 | `scripts/label/start_labeling.py` | `labels/unlabeled/` |
| 小模型数据 | 少量人工标注数据准备 | `datasets/<project>/label_model/` | 小模型训练集 |
| 小模型训练 | 图片标签配对检查和快速训练 | `scripts/train/train_auto_label.py` | `runs/auto_label/` |
| 自动打标 | 批量预测、标签生成、进度统计 | `scripts/label/auto_label.py` | `labels/unlabeled/*.txt` |
| 数据集整理 | 随机复制配对样本到 train/val | `scripts/data/split_dataset.py` | `images/train/`、`images/val/` |
| 主模型训练 | YOLO11n 训练和自动验证 | `scripts/train/train.py` | `runs/detect/` |
| 标准导出 | ONNX、TensorRT | `scripts/export/export.py` | `export_standard.*` |
| MaixCAM 导出 | YOLO11 Detect 头截断和双输出重命名 | `scripts/export/export_maixcam.py` | `export_maixcam.onnx` |
| MaixCAM 量化 | TPU-MLIR Docker 转换和 INT8 量化 | `scripts/quantize/convert_yolo11_to_cvimodel.sh` | `.cvimodel` |

## 工作流

```text
视频
  ↓
视频抽帧
  ↓
unlabeled 图片
  ↓
少量人工标注
  ↓
label_model 小模型训练
  ↓
自动打标
  ↓
人工修正标签
  ↓
随机划分 train / val
  ↓
主模型训练与验证
  ↓
标准 ONNX / TensorRT 导出
  ↓
MaixCAM ONNX 导出
  ↓
TPU-MLIR Docker 量化
  ↓
MaixCAM 部署
```

其中，`label_model` 是用于辅助准备数据集的小模型，不是最终部署模型。自动生成的标签需要经过人工检查和修正，确认后再划分到主模型的训练集和验证集。

## 核心目录

```text
.
├── configs/          # 各阶段配置
├── datasets/
│   └── template/     # 数据集模板
├── scripts/          # 数据、标注、训练、导出和量化脚本
├── weights/          # 本地预训练权重
├── runs/             # 本地训练和导出产物
├── docs/
│   ├── GUIDE.md
│   └── WORKFLOW_GUIDE.md
├── requirements.txt
└── README.md
```

- `datasets/template/` 用于复制出新的数据集目录。
- 实际数据、权重和运行结果属于本地内容，默认不会提交到 Git。
- 配置默认使用 `train.yaml`、`export.yaml` 和 `maixcam.yaml`。
- 需要多套参数时，可以在脚本顶部修改 `CONFIG_NAME`，例如 `train01.yaml`。
- 训练和导出实验目录由脚本自动按 `exp01`、`exp02`、`exp03` 顺序生成。

## 当前环境

| 项目 | 当前方案 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.12 |
| 训练框架 | Ultralytics YOLO11 |
| GPU 加速 | PyTorch CUDA |
| 当前 GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| MaixCAM 转换 | TPU-MLIR Docker |
| 部署设备 | MaixCAM |

## 文档

- 📖 [仓库使用指南](docs/GUIDE.md)：环境准备、配置修改和脚本运行方法。
- 🧭 [工作流指南](docs/WORKFLOW_GUIDE.md)：参考本仓库搭建自己的 YOLO 项目。

## 许可证与依赖

本仓库使用 Ultralytics YOLO 相关组件。Ultralytics 开源项目采用 AGPL-3.0，同时提供商业许可证方案。使用、修改或再发布相关组件前，请阅读其[官方许可证说明](https://github.com/ultralytics/ultralytics)和适用条款。

本仓库当前未单独声明自身许可证。
