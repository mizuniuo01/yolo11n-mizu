# 数据集模板

每个数据集使用以下目录结构：

- `data.yaml`：YOLO11 主模型训练配置
- `images/train/` 和 `labels/train/`：主模型训练样本
- `images/val/` 和 `labels/val/`：主模型验证样本
- `images/unlabeled/`：等待自动打标的图片
- `labels/unlabeled/`：自动生成并人工修正后的标签
- `label_model/`：用于训练自动打标模型的少量人工标注数据
- `label_model/data.yaml`：自动打标模型的独立配置

图片和标签文件使用相同的相对路径和文件名主干。YOLO 标签文件包含归一化的 `class_id center_x center_y width height` 值。类别 ID 从 `0` 开始，并与 `names` 中的顺序一致。
