# YOLO11n Object Detection Workflow

<div align="center">

<p>从数据准备、自动标注到 MaixCAM 部署的 YOLO11n 目标检测工作流</p>

<p><a href="docs/GUIDE.md">📖 使用指南</a> · <a href="docs/WORKFLOW_GUIDE.md">🧭 工作流指南</a></p>

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8">
<img src="https://img.shields.io/badge/Ultralytics-YOLO11-111F68" alt="Ultralytics YOLO11">
<img src="https://img.shields.io/badge/YOLO11n-object%20detection-5B4BDB" alt="YOLO11n">
</p>
<p>
<img src="https://img.shields.io/badge/ONNX-export-005CED?logo=onnx&logoColor=white" alt="ONNX">
<img src="https://img.shields.io/badge/Docker-TPU--MLIR-2496ED?logo=docker&logoColor=white" alt="Docker TPU-MLIR">
<img src="https://img.shields.io/badge/MaixCAM-deployment-00A98F" alt="MaixCAM">
</p>

</div>

## 项目简介

这是一个基于 Ultralytics YOLO11n 的目标检测工作流仓库，覆盖数据准备、人工标注、自动打标、模型训练、模型导出和 MaixCAM 部署。

仓库面向当前 Windows + CUDA 开发环境，主要用于记录和复用个人工作流。脚本、配置和模板可以作为其他 YOLO 项目的参考，但不承诺复制后即可在任意电脑上直接运行。

## 核心能力

<table><tr><td width="50%" valign="top">

### 🎞️ 数据准备
视频抽帧、数量预估和数据集划分。<br>
`scripts/data/`

### 🏷️ 人工标注
启动 LabelImg、同步类别并保存 YOLO 标签。<br>
`scripts/label/start_labeling.py`

### 🤖 自动打标
训练辅助小模型，批量生成 YOLO 标签。<br>
`scripts/train/train_auto_label.py` · `scripts/label/auto_label.py`

</td><td width="50%" valign="top">

### 🎯 主模型训练
训练 YOLO11n，并在训练过程中自动验证。<br>
`scripts/train/train.py`

### 📦 模型导出
导出标准 ONNX、TensorRT 和 MaixCAM ONNX。<br>
`scripts/export/`

### 🚀 边缘部署
使用 TPU-MLIR Docker 完成 INT8 量化并部署到 MaixCAM。<br>
`scripts/quantize/`

</td></tr></table>

## 工作流

<table><tr>
<td align="center" width="20%"><b>🎞️ 数据</b><br>抽帧<br>整理数据集</td>
<td align="center" width="20%"><b>🏷️ 标注</b><br>人工标注<br>自动打标<br>人工修正</td>
<td align="center" width="20%"><b>🎯 训练</b><br>小模型<br>主模型<br>自动验证</td>
<td align="center" width="20%"><b>📦 导出</b><br>ONNX<br>TensorRT<br>MaixCAM ONNX</td>
<td align="center" width="20%"><b>🚀 部署</b><br>TPU-MLIR<br>INT8<br>MaixCAM</td>
</tr></table>

少量人工标注数据先用于训练自动打标小模型，再由人工修正自动生成的标签，最后组成主模型训练集和验证集。

## 仓库结构

<table><tr><td width="50%" valign="top">

**代码与配置**

```text
configs/       各阶段配置
scripts/       数据、标注、训练、导出和量化脚本
docs/          使用指南和工作流指南
```

</td><td width="50%" valign="top">

**本地内容**

```text
datasets/template/  数据集模板
weights/            本地预训练权重
runs/               训练和导出产物
```

</td></tr></table>

- `datasets/template/` 用于复制出新的数据集目录。
- 实际数据、权重和运行产物默认不提交到 Git。
- 配置默认使用 `train.yaml`、`export.yaml` 和 `maixcam.yaml`。
- 需要多套参数时，在脚本顶部修改 `CONFIG_NAME`，例如 `train01.yaml`。
- 实验目录由脚本自动按 `exp01`、`exp02`、`exp03` 顺序生成。

## 当前环境

| 项目 | 当前方案 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.12 |
| 训练框架 | Ultralytics YOLO11 |
| GPU 加速 | PyTorch CUDA 12.8 |
| 当前 GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| MaixCAM 转换 | TPU-MLIR Docker |

## 文档

<table><tr><td width="50%" valign="top">

### 📖 仓库使用指南
环境准备、配置修改和脚本运行方法。<br>
[打开 GUIDE.md](docs/GUIDE.md)

</td><td width="50%" valign="top">

### 🧭 工作流指南
参考本仓库搭建自己的 YOLO 工程。<br>
[打开 WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)

</td></tr></table>

## 许可证与依赖

本仓库使用 Ultralytics YOLO 相关组件。Ultralytics 开源项目采用 AGPL-3.0，同时提供商业许可证方案。使用、修改或再发布相关组件前，请阅读其[官方许可证说明](https://github.com/ultralytics/ultralytics)和适用条款。

本仓库当前未单独声明自身许可证。
