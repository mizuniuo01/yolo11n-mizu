# yolo11n 仓库构建计划

## 阶段性流程

1. 确定目录结构
2. 整理数据集并编写 data.yaml
3. 编写训练配置和训练脚本
4. 编写通用导出脚本
5. 编写 MaixCam 专用导出脚本
6. 接入现有 Docker 量化流程
7. 配置 gitignore
8. 编写 README
9. 编写 GUIDE

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