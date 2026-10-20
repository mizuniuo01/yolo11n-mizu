# YOLO 工作流参考设计指南

[返回 README](../README.md) · [查看仓库使用指南](GUIDE.md)

本文档说明如何理解和参考本仓库的 YOLO 工作流，包括环境准备、目录设计、脚本组织、数据流、训练、导出和 Docker 量化。文档提供参考方案和入口命令，不提供详细安装教程，也不要求复刻本仓库。

读者应了解 YOLO、Python 和 Docker 的基本概念。当前仓库的具体运行方法见[仓库使用指南](GUIDE.md)。

## 整体工作流

```text
原始数据
→ 视频抽帧 / 图片整理
→ 少量人工标注
→ 辅助小模型训练
→ 自动打标
→ 人工修正
→ 主模型 train/val
→ 主模型训练与验证
→ 标准导出或设备专用导出
→ Docker 量化
→ 边缘设备部署
```

辅助小模型服务于数据准备，主模型负责最终任务。每个阶段通过明确的目录和文件产物衔接。

## 环境分层

### 宿主机

宿主机负责数据准备、人工标注、辅助小模型和主模型训练，以及标准 ONNX/TensorRT 导出。

需要准备的组件包括：

- Python 和独立虚拟环境。
- PyTorch、CUDA 和 NVIDIA 驱动。
- Ultralytics。
- OpenCV、PyYAML、tqdm 等脚本依赖。
- LabelImg 或其他人工标注工具。
- Git。

以下命令仅作为安装入口示例。本指南不提供详细安装教程，具体版本和命令应根据使用者设备与目标平台调整：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pip install ultralytics
pip install labelImg
```

PyTorch CUDA、Docker 和 TPU-MLIR 使用对应项目的官方安装文档。

### Docker

Docker 只负责 TPU-MLIR 工具链、ONNX 转换、校准和设备模型生成。训练和普通模型导出保留在宿主机，避免量化工具链污染 Python 环境。

## 目录设计

目录按职责隔离：

```text
configs/      阶段配置
datasets/     数据集和模板
scripts/      各阶段执行脚本
weights/      预训练权重
runs/         训练和导出产物
docs/         项目文档
```

数据集建议使用：

```text
datasets/<project>/
├── data.yaml
├── images/{unlabeled,train,val}/
├── labels/{unlabeled,train,val}/
└── label_model/
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

数据、配置、脚本、权重和运行产物分开管理，便于替换模型、清理实验和迁移工程。Docker 量化工作目录与 Python 工程目录分开。

## 数据集与标注设计

### 三类数据区

- `label_model`：少量人工标注数据，用于训练辅助小模型。
- `unlabeled`：抽帧、自动打标和人工修正的工作区。
- `train/val`：人工确认后用于主模型训练和验证的数据。

### 两阶段标注闭环

```text
少量人工标注
→ 辅助小模型训练
→ 自动打标
→ 人工修正
→ 主模型 train/val
→ 主模型训练
```

自动生成的标签必须人工修正后才能进入主模型数据集。主模型图片和标签需要按文件名配对。

## 脚本架构

脚本采用单职责设计：

| 模块 | 参考脚本 | 主要职责 |
| --- | --- | --- |
| 数据抽帧 | `scripts/data/capture_video.py` | 视频转换为未标注图片 |
| 数据划分 | `scripts/data/split_dataset.py` | 配对样本复制到 train/val |
| 人工标注 | `scripts/label/start_labeling.py` | 启动标注工具和类别同步 |
| 小模型训练 | `scripts/train/train_auto_label.py` | 训练辅助打标模型 |
| 自动打标 | `scripts/label/auto_label.py` | 生成 YOLO 标签 |
| 主模型训练 | `scripts/train/train.py` | 训练和自动验证 |
| 标准导出 | `scripts/export/export.py` | ONNX/TensorRT 导出 |
| 设备导出 | `scripts/export/export_maixcam.py` | 设备专用 ONNX |
| 量化 | `scripts/quantize/convert_yolo11_to_cvimodel.sh` | TPU-MLIR 转换和量化 |

当前仓库通过脚本顶部常量选择工程和配置：

```python
PROJECT_NAME = "<project>"
CONFIG_NAME = "train.yaml"
```

这种方式适合个人工程和少量配置。项目规模扩大后，可以迁移到统一配置加载器或 CLI，但仍应保持各脚本的阶段边界。

## 配置与实验管理

配置按模块和工程分层：

```text
configs/train/<project>/
configs/auto_label/<project>/
configs/export/<project>/
```

默认文件名可以保持 `train.yaml`、`export.yaml` 和 `maixcam.yaml`；多套参数可使用 `train01.yaml` 等文件，由脚本顶部的 `CONFIG_NAME` 选择。

实验名不写入 YAML，由脚本自动生成：

```text
exp01 → exp02 → exp03
```

建议保持以下对应关系：

```text
runs/detect/<project>/expXX/
runs/auto_label/<project>/expXX/
runs/export/<project>/expXX/
```

导出脚本应根据训练权重和已有导出产物匹配实验，避免重复覆盖或使用错误权重。

## 训练设计

### 辅助小模型

- 使用少量人工标注数据。
- 目标是获得预标注能力。
- 输出用于自动打标。
- 不作为最终部署模型。

### 主模型

- 使用人工修正并划分后的完整数据集。
- 训练过程自动验证 `val` 集。
- 输出用于后续标准导出和设备导出。

参数设计只保留方向：

- `imgsz` 与任务和部署输入协调。
- `batch` 根据显存调整。
- `cache` 根据内存调整。
- `amp` 只用于训练。
- `patience` 和 `epochs` 根据验证曲线调整。

参考入口命令：

```powershell
python -u scripts/train/train_auto_label.py
python -u scripts/train/train.py
```

命令只表示脚本入口，实际工程需要修改顶部配置和对应 YAML。

## 导出设计

### 标准导出

标准 ONNX/TensorRT 用于通用推理、模型检查和通用部署，保留标准模型输出。

参考入口：

```powershell
python -u scripts/export/export.py
```

### 设备专用导出

设备导出需要：

1. 从训练权重导出标准 ONNX。
2. 根据设备解析要求提取目标节点。
3. 截断设备不需要的部分。
4. 保留设备要求的输出。
5. 重命名输出节点。

当前 MaixCAM 方案使用：

```text
out_box
out_cls
```

参考入口：

```powershell
python -u scripts/export/export_maixcam.py
```

命令仅表示入口，实际配置需要按新工程和目标设备调整。

## Docker 与 TPU-MLIR 量化

### 量化链路

```text
设备专用 ONNX
→ MLIR
→ BF16 模型
→ 校准表
→ INT8 模型
```

### 参考脚本

仓库中的：

```text
scripts/quantize/convert_yolo11_to_cvimodel.sh
```

只是参考脚本。用户需要根据自己的工程修改：

- 模型名。
- ONNX 文件名。
- 输入尺寸和输入格式。
- 处理器型号。
- 输出节点。
- 测试图片。
- 校准图片目录和数量。
- 最终模型文件名。

修改完成后，将脚本放入自己的 Docker 工作目录，与模型和校准输入一起使用。

### Docker 工作目录

参考结构：

```text
models/<project>/
├── convert_yolo11_to_cvimodel.sh
├── export_maixcam.onnx
├── test.jpg
└── images/
```

Docker 命令只作为参考形式：

```powershell
docker run --rm -it `
  -v "<host-model-directory>:/workspace" `
  <tpu-mlir-image> /bin/bash
```

容器内：

```bash
cd /workspace
chmod +x convert_yolo11_to_cvimodel.sh
./convert_yolo11_to_cvimodel.sh
```

其中 `<host-model-directory>` 和 `<tpu-mlir-image>` 都需要替换为使用者自己的实际值。量化脚本不会自动创建或复制模型、测试图片或校准图片。

### 校准图片

校准图片应来自真实部署场景，并覆盖光照、目标大小和位置、背景、遮挡、噪声及典型异常情况。数量根据模型、工具链和场景覆盖情况决定。

## `.mud` 与设备模型

- `.mud` 是设备运行时描述文件。
- `.cvimodel` 是 TPU-MLIR 生成的设备模型。
- `.mud` 中的模型文件名必须与 `.cvimodel` 实际文件名一致。
- 类别顺序必须与训练配置一致。
- 输入尺寸、输入格式和输出节点必须匹配。

不写设备端上传和启动教程。

## 关键设计边界

- 辅助数据集和主模型数据集分开。
- 自动标签必须人工修正。
- 标准导出和设备专用导出分开。
- 设备输出节点必须根据实际模型结构确认。
- 量化脚本修改后才能用于新工程。
- Docker 工作目录与 Python 工程目录分开。
- `.mud` 和 `.cvimodel` 必须匹配。

## 参考命令

以下命令只表示模块入口，具体路径、工程名和参数需要按新项目调整：

```powershell
python -u scripts/data/capture_video.py
python -u scripts/label/start_labeling.py
python -u scripts/train/train_auto_label.py
python -u scripts/label/auto_label.py
python -u scripts/data/split_dataset.py
python -u scripts/train/train.py
python -u scripts/export/export.py
python -u scripts/export/export_maixcam.py
```

Docker 内：

```bash
chmod +x convert_yolo11_to_cvimodel.sh
./convert_yolo11_to_cvimodel.sh
```

## 迁移建议

参考本仓库搭建其他 YOLO 工程时：

1. 先确定数据、训练、导出和量化的职责边界。
2. 决定是否需要辅助小模型。
3. 按实际模型结构确认设备输出。
4. 按目标芯片调整 TPU-MLIR 参数。
5. 修改参考量化脚本后再放入 Docker 工作目录。
6. 保持训练权重、导出产物和实验目录对应。
7. 重新验证 `.mud`、`.cvimodel`、输入和类别配置。

## 文档约束

- `plan.md` 后续直接删除，不再维护。
- 安装命令仅作为参考入口。
- 本指南不提供详细安装教程。
- 不使用当前仓库的绝对路径和固定工程名。
- 不将参考命令描述为固定可复制命令。
- 不出现无关历史实现比较。
- 本指南比 `GUIDE.md` 更抽象，更关注设计、环境分层和迁移。
