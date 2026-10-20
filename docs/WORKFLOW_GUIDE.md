# YOLO 工作流参考设计指南

[返回 README](../README.md) · [查看仓库使用指南](GUIDE.md)

本文档拆解本仓库的设计，用于参考搭建其他 YOLO 目标检测工程。它面向知道 YOLO、Python 和 Docker 基础的开发者，重点说明模块边界、数据流、脚本接口和部署衔接。文中的命令是参考入口，不是逐步的新手教程。

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

宿主机负责数据准备、LabelImg 标注、辅助模型和主模型训练，以及标准 ONNX/TensorRT 导出。常用组件包括：

- Python 3.12 和独立虚拟环境。
- PyTorch、CUDA 和 NVIDIA 驱动。
- Ultralytics、OpenCV、PyYAML、tqdm。
- LabelImg。

安装入口可以参考：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu128
pip install -U ultralytics
pip install opencv-python PyYAML tqdm matplotlib pillow
pip install labelImg onnx onnxruntime-gpu onnxslim
```

PyTorch CUDA 使用 PyTorch 官方页面提供的对应命令。

实际安装的依赖及版本，取决于使用者的设备配置与环境，不可照抄本仓库依赖，本仓库 `requirements.txt` 仅供参考。

### Docker

Docker 只负责 TPU-MLIR 工具链、模型转换、校准和设备模型生成。训练和普通 ONNX 导出保留在宿主机，避免量化工具链污染 Python 环境。

## 目录设计

目录按职责隔离：

```text
configs/      可修改的阶段配置
datasets/     数据集和数据集模板
scripts/      各阶段执行脚本
weights/      预训练权重入口
runs/         训练和导出产物
docs/         项目文档
```

数据集建议使用：

```text
datasets/<project>/
├── data.yaml
├── videos/
├── images/{unlabeled,train,val}/
├── labels/{unlabeled,train,val}/
└── label_model/
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

数据、配置、代码、权重和运行产物分开，便于替换模型、清理实验和迁移工程。

## 数据集与标注设计

### 三类数据区

- `label_model`：少量人工标注数据，用于训练辅助小模型。
- `unlabeled`：抽帧、自动打标和人工修正的工作区。
- `train/val`：人工确认后用于主模型训练和验证的数据。

### 两阶段标注闭环

```text
少量人工标注
→ 辅助小模型训练
→ 批量自动打标
→ 人工修正
→ 主模型数据集划分
```

自动生成的标签不能直接作为最终训练标签。主模型划分脚本只处理图片和标签均存在的配对样本，并保留 `unlabeled` 源文件。

## 脚本架构

脚本采用单职责设计：

| 模块 | 参考脚本 | 主要职责 |
| --- | --- | --- |
| 数据抽帧 | `scripts/data/capture_video.py` | 视频转换为未标注图片 |
| 数据划分 | `scripts/data/split_dataset.py` | 配对样本复制到 train/val |
| 人工标注 | `scripts/label/start_labeling.py` | 启动 LabelImg 和类别同步 |
| 小模型训练 | `scripts/train/train_auto_label.py` | 训练辅助打标模型 |
| 自动打标 | `scripts/label/auto_label.py` | 生成 YOLO 标签 |
| 主模型训练 | `scripts/train/train.py` | 训练和自动验证 |
| 标准导出 | `scripts/export/export.py` | ONNX/TensorRT 导出 |
| 设备导出 | `scripts/export/export_maixcam.py` | MaixCAM 专用 ONNX |
| 量化 | `scripts/quantize/convert_yolo11_to_cvimodel.sh` | TPU-MLIR 转换和量化 |

当前仓库通过脚本顶部常量选择工程和配置：

```python
PROJECT_NAME = "template"
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

辅助小模型只使用少量人工数据，用于快速获得预标注能力，不承担最终部署和主模型评估。主模型使用人工修正并划分后的完整数据集，训练时自动验证 `val` 集。

常见参数设计方向：

- `imgsz` 与目标尺寸和部署输入尺寸协调。
- `batch` 根据显存调整。
- `cache` 根据内存容量选择。
- `amp` 用于训练加速，不参与模型导出。
- `patience` 和 `epochs` 根据验证曲线调整。

参考入口：

```powershell
python -u scripts/train/train_auto_label.py
python -u scripts/train/train.py
```

## 导出设计

### 标准导出

标准 ONNX/TensorRT 导出保留通用模型结构，用于通用推理、模型检查或服务器端部署：

```powershell
python -u scripts/export/export.py
```

### 设备专用导出

MaixCAM 导出是独立路径：

1. 从训练权重导出标准 YOLO11 ONNX。
2. 根据 YOLO11 官方 Detect 结构提取目标节点。
3. 截断设备不需要的部分。
4. 保留 DFL box 和 Sigmoid class 两个输出。
5. 重命名为 `out_box` 和 `out_cls`。

```powershell
python -u scripts/export/export_maixcam.py
```

## Docker 与 TPU-MLIR 量化

### 转换链路

```text
export_maixcam.onnx
→ MLIR
→ BF16 cvimodel
→ 校准表
→ INT8 cvimodel
```

`model_transform.py` 负责输入、输出和 MLIR 转换；`run_calibration.py` 使用真实场景图片生成校准表；`model_deploy.py` 生成设备模型。

### 工作目录

建议把量化输入集中到独立目录：

```text
models/<project>/
├── export_maixcam.onnx
├── test.jpg
├── images/
└── convert_yolo11_to_cvimodel.sh
```

挂载和启动命令只作为参考形式：

```powershell
docker run --rm -it `
  -v "${PWD}\models\<project>:/workspace" `
  <tpu-mlir-image> /bin/bash
```

容器内：

```bash
cd /workspace
chmod +x convert_yolo11_to_cvimodel.sh
./convert_yolo11_to_cvimodel.sh
```

实际镜像、TPU-MLIR 版本、处理器型号和输入尺寸必须根据目标芯片及官方工具链调整。

### 校准图片

校准图片应来自真实部署场景，覆盖光照、目标大小、背景、位置、遮挡和噪声变化。数量根据模型规模、工具链要求和场景覆盖情况调整，不建议只使用单一场景图片。

## MaixCAM 模型文件

`.mud` 是设备运行时描述文件，负责模型类型、输入格式、类别名称、归一化参数和对应 `.cvimodel` 文件名。两者必须匹配：

```ini
model = yolo11n.cvimodel
```

`.cvimodel` 由 TPU-MLIR 生成；`.mud` 模板应随工程或设备模型一起管理。类别顺序、输入尺寸、输入格式和输出节点必须与训练、导出和量化结果一致。

## 关键设计边界

- `label_model` 与主模型数据集分开。
- 自动标签必须经过人工修正。
- 辅助小模型和主模型使用不同的运行输出目录。
- 标准 ONNX 与设备专用 ONNX 不混用。
- TPU-MLIR 与普通 Python 环境分开。
- Docker 挂载目录中的文件名必须与 shell 配置一致。
- `.mud` 和 `.cvimodel` 必须匹配模型名、类别和输入配置。

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

1. 先确定数据区、训练区、导出区和量化区的职责。
2. 决定是否需要辅助小模型和自动打标闭环。
3. 根据模型结构重新确认设备导出节点。
4. 根据目标芯片确定 TPU-MLIR processor、输入尺寸和输出约定。
5. 修改量化脚本中的模型名、输出节点和校准配置。
6. 保持训练权重、导出产物和实验目录一一对应。
7. 将 `.mud` 模板与最终 `.cvimodel` 一起验证。
8. 复用流程和边界，不假设不同 YOLO 版本具有相同输出结构。
