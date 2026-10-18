"""Train a small YOLO11 model for automatic image labeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

# Edit this value to select a project under configs/auto_label/.
PROJECT_NAME = "auto_aim"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "auto_label"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def config_path() -> Path:
    """Return the selected project's automatic-labeling config."""
    return CONFIG_ROOT / PROJECT_NAME / "train.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML training configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"自动打标小模型配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"训练配置必须是 YAML 映射: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """Validate the dataset config and local model weight paths."""
    for key in ("data", "weights"):
        if key not in config:
            raise KeyError(f"训练配置缺少字段: {key}")

    data_path = resolve_path(config["data"])
    weights_path = resolve_path(config["weights"])
    if not data_path.is_file():
        raise FileNotFoundError(f"小模型数据配置不存在: {data_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"预训练权重不存在: {weights_path}\n"
            "请先下载 yolo11n.pt 并放入 weights/ 目录。"
        )
    return data_path, weights_path


def dataset_root(data_path: Path) -> Path:
    """Return the label-model dataset root from its data YAML path."""
    return data_path.parent


def count_pairs(image_dir: Path, label_dir: Path) -> tuple[int, int]:
    """Return image count and image count missing matching labels."""
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"小模型数据目录不完整: {image_dir.parent}")

    image_count = 0
    missing_count = 0
    for image_path in image_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_count += 1
        if not (label_dir / f"{image_path.stem}.txt").is_file():
            missing_count += 1
    return image_count, missing_count


def validate_dataset(data_path: Path) -> tuple[int, int, int, int]:
    """Check train and val image-label pairing and return their counts."""
    root = dataset_root(data_path)
    train_images, train_missing = count_pairs(
        root / "images" / "train", root / "labels" / "train"
    )
    val_images, val_missing = count_pairs(
        root / "images" / "val", root / "labels" / "val"
    )
    if train_images == train_missing or val_images == val_missing:
        raise RuntimeError("小模型数据集中没有可用的图片标签配对样本。")
    return train_images, train_missing, val_images, val_missing


def training_arguments(config: dict[str, Any], data_path: Path) -> dict[str, Any]:
    """Build Ultralytics arguments and an absolute output project path."""
    arguments = dict(config)
    arguments.pop("weights", None)
    project = str(arguments.get("project", PROJECT_NAME))
    arguments["project"] = str((PROJECT_ROOT / "runs" / "auto_label" / project).resolve())
    arguments["data"] = str(data_path)
    return arguments


def main() -> None:
    """Train the automatic-labeling model."""
    config = load_config(config_path())
    data_path, weights_path = validate_paths(config)
    train_count, train_missing, val_count, val_missing = validate_dataset(data_path)
    arguments = training_arguments(config, data_path)
    if arguments.get("device", 0) != 0 or not torch.cuda.is_available():
        raise RuntimeError("本机自动打标小模型训练需要 CUDA device=0。")

    project = config.get("project", PROJECT_NAME)
    experiment = config.get("name", "train")
    print(f"工程: {PROJECT_NAME}", flush=True)
    print(f"数据配置: {data_path}", flush=True)
    print(f"预训练权重: {weights_path}", flush=True)
    print(f"训练设备: CUDA:0 ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"训练集图片: {train_count}，缺少标签: {train_missing}", flush=True)
    print(f"验证集图片: {val_count}，缺少标签: {val_missing}", flush=True)
    print(f"训练结果: runs/auto_label/{project}/{experiment}", flush=True)

    model = YOLO(str(weights_path))
    model.train(**arguments)


if __name__ == "__main__":
    main()
