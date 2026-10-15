"""Export YOLO11 detection outputs for the MaixCAM conversion workflow."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# Edit this value to select a project under configs/export/.
PROJECT_NAME = "template"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "export"


def config_path() -> Path:
    """Return the selected MaixCAM configuration path."""
    return CONFIG_ROOT / PROJECT_NAME / "maixcam.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a MaixCAM YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"MaixCAM 配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML 映射: {path}")
    return config


def checkpoint_path(config: dict[str, Any]) -> Path:
    """Return the trained checkpoint selected by project and experiment."""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("experiment", "exp01"))
    return PROJECT_ROOT / "runs" / "detect" / project / experiment / "weights" / "best.pt"


def output_path(config: dict[str, Any]) -> Path:
    """Return the MaixCAM ONNX output path."""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("experiment", "exp01"))
    directory = PROJECT_ROOT / "runs" / "export" / project / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "export_maixcam.onnx"


def maixcam_forward(self: Detect, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return YOLO11 DFL box and sigmoid class outputs without NMS."""
    batch_size = features[0].shape[0]
    box_outputs = [
        self.cv2[index](features[index]).view(batch_size, 4 * self.reg_max, -1)
        for index in range(self.nl)
    ]
    class_outputs = [
        self.cv3[index](features[index]).view(batch_size, self.nc, -1)
        for index in range(self.nl)
    ]
    boxes = self.dfl(torch.cat(box_outputs, dim=2))
    classes = torch.cat(class_outputs, dim=2).sigmoid()
    return boxes, classes


def replace_detect_forward(model: nn.Module) -> None:
    """Replace the YOLO11 Detect forward method with MaixCAM outputs."""
    detect_modules = [module for module in model.modules() if isinstance(module, Detect)]
    if len(detect_modules) != 1:
        raise RuntimeError(f"期望找到一个 Detect 模块，实际找到 {len(detect_modules)} 个")
    detect = detect_modules[0]
    detect.forward = types.MethodType(maixcam_forward, detect)


def validate_config(config: dict[str, Any]) -> tuple[Path, tuple[int, int], list[str]]:
    """Validate paths, input size and output names."""
    checkpoint = checkpoint_path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"训练权重不存在: {checkpoint}")
    imgsz = config.get("imgsz", [160, 256])
    if not isinstance(imgsz, list) or len(imgsz) != 2:
        raise ValueError("imgsz 必须是 [height, width]")
    output_names = config.get("output_names", ["out_box", "out_cls"])
    if not isinstance(output_names, list) or len(output_names) != 2:
        raise ValueError("output_names 必须包含两个输出名称")
    return checkpoint, (int(imgsz[0]), int(imgsz[1])), [str(name) for name in output_names]


def main() -> None:
    """Export the configured YOLO11 checkpoint for MaixCAM."""
    config = load_config(config_path())
    checkpoint, (height, width), output_names = validate_config(config)
    target = output_path(config)
    model = YOLO(str(checkpoint))
    model.model.eval()
    replace_detect_forward(model.model)

    dummy_input = torch.zeros(1, 3, height, width)
    print(f"输入权重: {checkpoint}", flush=True)
    print(f"输入形状: [1, 3, {height}, {width}]", flush=True)
    print(f"输出名称: {output_names}", flush=True)
    print(f"输出文件: {target}", flush=True)

    torch.onnx.export(
        model.model,
        dummy_input,
        str(target),
        opset_version=int(config.get("opset", 17)),
        input_names=[str(config.get("input_name", "images"))],
        output_names=output_names,
        dynamo=False,
    )
    print(f"MaixCAM 导出完成: {target}", flush=True)


if __name__ == "__main__":
    main()
