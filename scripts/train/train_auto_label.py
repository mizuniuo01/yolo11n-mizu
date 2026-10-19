"""训练用于自动打标的 YOLO11 小模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

# 选择 configs/auto_label/ 下的工程配置。
PROJECT_NAME = "template"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "auto_label"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def config_path() -> Path:
    """返回选定工程的自动打标配置。"""
    return CONFIG_ROOT / PROJECT_NAME / "train.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """加载并检查 YAML 训练配置。"""
    if not path.is_file():
        raise FileNotFoundError(f"自动打标小模型配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"训练配置必须是 YAML 映射: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """将工程相对路径解析为仓库根目录下的路径。"""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """检查数据配置和本地模型权重路径。"""
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
    """根据数据配置返回小模型数据集根目录。"""
    return data_path.parent


def count_pairs(image_dir: Path, label_dir: Path) -> tuple[int, int]:
    """返回图片数量和缺少对应标签的图片数量。"""
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"小模型数据目录不完整: {image_dir.parent}")

    image_count = 0
    missing_count = 0
    for image_path in image_dir.iterdir():
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue
        image_count += 1
        if not (label_dir / f"{image_path.stem}.txt").is_file():
            missing_count += 1
    return image_count, missing_count


def validate_dataset(data_path: Path) -> tuple[int, int, int, int]:
    """检查 train、val 图片标签配对并返回统计数量。"""
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


def training_arguments(
    config: dict[str, Any], data_path: Path
) -> dict[str, Any]:
    """构造 Ultralytics 参数和绝对输出路径。"""
    arguments = dict(config)
    arguments.pop("weights", None)
    project = str(arguments.get("project", PROJECT_NAME))
    arguments["project"] = str(
        (PROJECT_ROOT / "runs" / "auto_label" / project).resolve()
    )
    arguments["data"] = str(data_path)
    return arguments


def main() -> None:
    """训练自动打标模型。"""
    config = load_config(config_path())
    data_path, weights_path = validate_paths(config)
    train_count, train_missing, val_count, val_missing = validate_dataset(
        data_path
    )
    arguments = training_arguments(config, data_path)
    if arguments.get("device", 0) != 0 or not torch.cuda.is_available():
        raise RuntimeError("本机自动打标小模型训练需要 CUDA device=0。")

    project = config.get("project", PROJECT_NAME)
    experiment = config.get("name", "train")
    print("[准备] 自动打标小模型训练配置", flush=True)
    print(f"工程: {PROJECT_NAME}", flush=True)
    print(f"数据配置: {data_path}", flush=True)
    print(f"预训练权重: {weights_path}", flush=True)
    print(f"训练设备: CUDA:0 ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"训练集图片: {train_count}，缺少标签: {train_missing}", flush=True)
    print(f"验证集图片: {val_count}，缺少标签: {val_missing}", flush=True)
    print(f"训练结果: runs/auto_label/{project}/{experiment}", flush=True)

    print("[处理] 开始训练，以下为 Ultralytics 实时进度", flush=True)
    model = YOLO(str(weights_path))
    model.train(**arguments)


if __name__ == "__main__":
    main()
