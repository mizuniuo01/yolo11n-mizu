# yolo11n 仓库构建计划

## 阶段性流程

1. 确定目录结构
2. 整理数据集并编写 data.yaml
3. 视频抽帧脚本
4. 随机抽取图片脚本
5. 准备打标工具
6. 编写训练配置和训练脚本
7. 编写自动打标小模型脚本
8. 编写自动打标工具脚本
9. 编写通用导出脚本
10. 编写 MaixCam 专用导出脚本
11. 接入现有 Docker 量化流程
12. 全流程测试并整理仓库
13. 配置 gitignore
14. 编写 README
15. 编写 GUIDE

## 特别说明

1. 参考 `D:\Study\Python\yolov5` 工程的目录组织和工作流程。
2. MaixCam 导出脚本参考 YOLOv5 的导出方式，但根据 YOLO11 的模型结构重新实现。
3. 沿用 YOLOv5 工程中的 Docker 量化方案，Python 工程只负责生成 ONNX 和准备量化输入。
4. 训练过程使用 Ultralytics API，训练时自动进行验证。
5. 训练、导出和量化流程保持相互独立，方便单独执行。
6. YOLOv5 工程仅作为参考，不直接复制其完整源码。
7. 数据集保持标准 YOLO 格式，优先复用现有数据集和 `data.yaml`。
8. 所有脚本尽可能做好兼容性，以便直接在不同设备上复用。
9. GUIDE.md 以面向新手的程度进行编写。
10. 基于Python语言的需严格遵循PEP-8规范进行编写。

## 阶段性日志

### 确定目录结构

本阶段设计：

- 项目按数据集、配置、脚本、权重和运行结果划分顶层职责。
- 数据集模板采用标准 YOLO 检测结构，包含 `images/train`、`images/val`、`images/unlabeled` 及对应的 `labels` 目录。
- 每个数据集单独保留 `label_model/`，其中的少量人工标注数据专门用于训练自动打标小模型。
- 自动打标生成的标签写入数据集的 `labels/unlabeled/`，人工修正后再与图片一起移动到 `train` 或 `val`。
- 主模型训练、自动打标小模型训练、自动打标工具和导出工具分别预留独立的配置与脚本位置。
- `runs/` 暂时沿用 Ultralytics 的默认内部输出结构，不在本阶段强行规定训练和导出子目录。
- 实际数据集、模型权重和运行产物作为本地内容管理，模板、脚本、配置和文档作为项目内容管理。

本阶段产出：

- 目录结构模板。
- 数据集模板和模板 `data.yaml`。
- 自动打标小模型数据入口约定。
- 自动打标标签流转约定。

后续阶段再根据实际脚本行为确定具体配置字段、运行参数和输出文件命名。

### 整理数据集并编写 data.yaml

本阶段设计：

- 数据集模板使用标准 YOLO 检测数据格式，正式数据分为 `train` 和 `val`，待自动打标数据使用 `unlabeled`。
- 每个数据集独立保存自己的 `data.yaml`，路径以该数据集目录为基准，并使用相对路径引用图片目录。
- `label_model/` 保存人工准备的少量标注数据，并拥有独立的 `label_model/data.yaml`，不复用主模型的数据路径。
- 类别 ID 从 `0` 开始连续编号，`names` 的顺序与标签文件中的类别 ID 保持一致。
- 图片和标签按相同文件名主干对应，标签采用归一化的 YOLO 检测格式。
- 自动生成和人工修正中的标签保存在 `labels/unlabeled/`，确认后与图片一起移动到 `train` 或 `val`。
- 本阶段只准备模板和配置约定，不重新划分实际数据，也不实现抽帧、随机抽图或自动打标脚本。

本阶段产出：

- 主模型 `data.yaml` 模板。
- 自动打标小模型独立的 `label_model/data.yaml` 模板。
- 数据集目录、图片标签对应关系和类别配置规则说明。

验证方式：

- 检查两个 YAML 文件能够解析。
- 检查 `train` 和 `val` 路径分别对应预留的图片目录。
- 检查主模型配置和小模型配置不会引用错误的数据目录。

### 视频抽帧脚本

本阶段设计：

- 视频按数据集和项目名保存于 `datasets/<dataset_name>/videos/<project_name>/`。
- 抽帧脚本处理项目目录中的全部支持格式视频。
- 输出图片写入对应数据集的 `images/unlabeled/`。
- 所有视频共用六位数字连续编号，避免不同视频产生重名。
- 抽帧参数集中写在脚本顶部，脚本使用 `pathlib`、OpenCV 和 `tqdm`，遵循 PEP 8。

### 随机抽取图片和标签

本阶段设计：

- 随机抽取脚本由人工在自动打标和人工修正完成后手动运行。
- 脚本只处理 `images/unlabeled/` 与 `labels/unlabeled/` 中能够按文件名配对的样本。
- 配对样本按固定随机种子划分并复制到主数据集的 `train` 和 `val`。
- `unlabeled` 中的原始图片和标签保留，不执行移动或删除。
- 缺少标签的图片不参与划分，并输出统计信息。
- 路径、比例、随机种子和覆盖策略集中写在脚本顶部，代码按 PEP 8 组织。

本阶段产出：

- `scripts/data/capture_video.py`
- `scripts/data/split_dataset.py`
- 视频目录、抽帧输出和随机划分规则说明。

### 准备打标工具

本阶段设计：

- 使用当前 `.venv` 中的 LabelImg 作为人工标注工具。
- 通过 `scripts/label/start_labeling.py` 启动，不直接复制 YOLOv5 的旧入口。
- 脚本顶部配置数据集名称和标注目标，可在 `unlabeled` 与 `label_model` 之间切换。
- `unlabeled` 模式打开 `images/unlabeled/`，标签保存到 `labels/unlabeled/`。
- `label_model` 模式根据 `train` 或 `val` 配置打开小模型数据目录，并保存到对应的 `label_model/labels/` 目录。
- 类别名称从对应的 `data.yaml` 读取，生成 LabelImg 使用的临时类别文件，避免重复维护类别列表。
- 工具只负责启动人工标注界面和保存标签，不执行自动预测、数据移动或训练集划分。
- 脚本使用顶部配置、`pathlib`、类型提示和函数拆分，遵循 PEP 8。

本阶段产出：

- `scripts/label/start_labeling.py`
- LabelImg 的数据目录映射和类别配置约定。

验证方式：

- 检查脚本可以解析对应 `data.yaml` 的类别名称。
- 检查 `unlabeled` 与 `label_model` 的图片、标签目录映射正确。
- 检查标签目录不存在时能够自动创建。
- 在安装 LabelImg 后启动脚本，确认 GUI 使用正确图片和标签目录。

### 编写训练配置和训练脚本

本阶段设计：

- 主模型训练使用 Ultralytics API，训练过程中自动使用 `val` 集验证。
- YOLO11n 预训练权重提前下载并保存为 `weights/yolo11n.pt`；训练脚本只加载本地权重，缺失时直接提示错误。
- 训练配置按工程分目录保存：`configs/train/<project_name>/train.yaml`。
- 配置文件集中保存工程名、实验名、数据集配置路径、权重路径和训练参数。
- `scripts/train/train.py` 通过顶部的 `PROJECT_NAME` 选择工程配置。
- 配置中的相对路径统一相对于项目根目录解析，不依赖当前终端工作目录。
- 训练结果使用配置中的 `project` 和 `name` 区分，目录形式为 `runs/detect/<project_name>/<experiment_name>/`。
- 本阶段只实现主模型训练，不包含自动打标小模型训练。
- 训练脚本使用类型提示、函数拆分、文档字符串和 PEP 8 风格。

本阶段产出：

- `configs/train/template/train.yaml`
- `scripts/train/train.py`
- 预训练权重下载和本地加载约定。

运行方式：

```powershell
.\\.venv\\Scripts\\Activate.ps1
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python scripts/train/train.py
```

验证方式：

- 检查训练配置和脚本可以解析。
- 检查数据集配置和本地权重路径不存在时有明确错误。
- 检查 `PROJECT_NAME` 可以选择对应工程配置。
- 检查训练结果按工程名和实验名区分。
- 通过一次短训练确认 CUDA GPU 可用并生成 `best.pt`。

补充设计：

- 本仓库面向当前开发电脑，不作为跨电脑复现工程。
- 训练使用本机 `.venv`、PyTorch CUDA 和 RTX 5070 Ti Laptop GPU。
- 配置固定使用 `device: 0`，CUDA 不可用时脚本直接报错，不切换 CPU。
- `imgsz` 使用 `[height, width]` 格式，模板默认 `[640, 640]`。
- 训练脚本在启动前打印工程、数据、权重、GPU 和结果目录，并使用 `flush=True` 保证信息及时显示。
- Ultralytics 负责显示 epoch、batch、loss 和验证进度，训练入口不重写其进度条。
- 直接使用本地终端运行脚本；需要无缓冲输出时使用 `python -u scripts/train/train.py`。
- Docker 不用于训练，仅在后续阶段保留给 MaixCam 量化流程。

### 编写自动打标小模型脚本

本阶段设计：

- 自动打标小模型使用少量人工准备的 `label_model/` 数据训练。
- 小模型使用独立配置目录 `configs/auto_label/<project_name>/train.yaml` 和独立入口 `scripts/train/train_auto_label.py`。
- 初始权重继续使用本地 `weights/yolo11n.pt`，不在训练脚本中自动联网下载。
- 训练前检查 train/val 图片与同名标签的配对情况，并报告缺失标签。
- 小模型采用独立的较小输入尺寸和训练参数，以便快速获得预标注模型。
- 训练结果与主模型保持相同的工程和实验层级，保存到 `runs/auto_label/<project_name>/<experiment_name>/`。
- 训练完成后的 `best.pt` 作为下一阶段自动打标工具的输入。
- 本阶段仍使用本机 `.venv`、CUDA device 0 和 Ultralytics 的终端进度显示。
- 脚本使用类型提示、函数拆分和文档字符串，遵循 PEP 8。

本阶段产出：

- `configs/auto_label/template/train.yaml`
- `scripts/train/train_auto_label.py`

验证方式：

- 检查配置和脚本可以解析。
- 检查小模型数据路径和权重路径。
- 检查图片标签配对统计和缺失标签提示。
- 使用少量人工标注数据进行 GPU 短训练，确认生成 `runs/auto_label/<project>/<experiment>/weights/best.pt`。

### 编写自动打标工具脚本

本阶段设计：

- 自动打标工具复用第七阶段的 `configs/auto_label/<project_name>/train.yaml`。
- 根据配置中的 `project` 和 `name` 自动定位 `runs/auto_label/<project>/<experiment>/weights/best.pt`。
- 数据集根目录由配置中的 `label_model/data.yaml` 自动推导。
- 输入为 `datasets/<dataset_name>/images/unlabeled/`，输出为对应的 `labels/unlabeled/`。
- 置信度阈值、IoU 阈值和标签覆盖策略放在脚本顶部，便于频繁调整。
- 默认覆盖已有标签，将 `OVERWRITE_LABELS` 改为 `False` 时跳过已有标签。
- 使用 Ultralytics 批量预测和进度显示，不生成预览图，不移动图片，不写入 train/val。
- 无检测结果时生成空的同名 `.txt` 文件。
- 脚本使用类型提示、函数拆分和文档字符串，遵循 PEP 8。

本阶段产出：

- `scripts/label/auto_label.py`
- 小模型权重定位、预测参数和标签输出约定。

验证方式：

- 检查脚本可以解析第七阶段配置并定位小模型权重。
- 检查输入和输出目录推导正确。
- 使用小样本确认生成标准 YOLO 标签和空标签文件。
- 确认覆盖开关和 CUDA 设备检查行为。
