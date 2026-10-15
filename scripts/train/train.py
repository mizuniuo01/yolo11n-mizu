"""Train a YOLO11 model using a project-specific YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import torch
from ultralytics import YOLO

# Edit this value to select a project under configs/train/.
PROJECT_NAME = "template"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "train"


def config_path() -> Path:
    """Return the selected project's training configuration path."""
    return CONFIG_ROOT / PROJECT_NAME / "train.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a training YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"训练配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"训练配置必须是 YAML 映射: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_config(config: dict[str, Any]) -> tuple[Path, Path]:
    """Validate required paths and return the dataset and weight paths."""
    for key in ("data", "weights"):
        if key not in config:
            raise KeyError(f"训练配置缺少字段: {key}")

    data_path = resolve_path(config["data"])
    weights_path = resolve_path(config["weights"])
    if not data_path.is_file():
        raise FileNotFoundError(f"数据集配置不存在: {data_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"预训练权重不存在: {weights_path}\n"
            "请先下载 yolo11n.pt 并放入 weights/ 目录。"
        )
    return data_path, weights_path


def training_arguments(config: dict[str, Any], data_path: Path) -> dict[str, Any]:
    """Build Ultralytics training arguments from the project configuration."""
    arguments = dict(config)
    arguments.pop("weights", None)
    arguments["data"] = str(data_path)
    return arguments


def validate_device(arguments: dict[str, Any]) -> None:
    """Require the configured CUDA device on this GPU training machine."""
    device = arguments.get("device", 0)
    if device != 0:
        raise ValueError("本机训练配置必须使用 device: 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查本机 NVIDIA 驱动和 PyTorch CUDA 环境。")


def print_training_info(
    data_path: Path,
    weights_path: Path,
    project: str,
    experiment: str,
) -> None:
    """Print training details before Ultralytics starts its progress display."""
    print(f"工程: {PROJECT_NAME}", flush=True)
    print(f"数据配置: {data_path}", flush=True)
    print(f"预训练权重: {weights_path}", flush=True)
    print(f"训练设备: CUDA:0 ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"训练结果: runs/detect/{project}/{experiment}", flush=True)


def main() -> None:
    """Load the selected project and start training."""
    path = config_path()
    config = load_config(path)
    data_path, weights_path = validate_config(config)
    arguments = training_arguments(config, data_path)
    validate_device(arguments)

    project = arguments.get("project", PROJECT_NAME)
    experiment = arguments.get("name", "train")
    print_training_info(data_path, weights_path, project, experiment)

    model = YOLO(str(weights_path))
    model.train(**arguments)


if __name__ == "__main__":
    main()
