"""将已标注样本从 unlabeled 复制到 train 和 val。"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

# 运行前修改本区域配置。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "template"
TRAIN_RATIO = 0.8
RANDOM_SEED = 0
OVERWRITE_EXISTING = False
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    """一张图片及其对应的 YOLO 标签文件。"""

    image: Path
    label: Path


def dataset_directory() -> Path:
    """返回配置的数据集目录。"""
    return PROJECT_ROOT / "datasets" / DATASET_NAME


def find_samples(dataset: Path) -> tuple[list[Sample], int]:
    """在未标注目录中查找图片和标签配对。"""
    image_directory = dataset / "images" / "unlabeled"
    label_directory = dataset / "labels" / "unlabeled"
    missing_labels = 0
    samples = []

    for image_path in sorted(image_directory.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_directory / f"{image_path.stem}.txt"
        if label_path.is_file():
            samples.append(Sample(image_path, label_path))
        else:
            missing_labels += 1
    return samples, missing_labels


def copy_sample(sample: Sample, dataset: Path, split: str) -> None:
    """将一组配对样本复制到指定数据集分区。"""
    image_target = dataset / "images" / split / sample.image.name
    label_target = dataset / "labels" / split / sample.label.name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)

    if not OVERWRITE_EXISTING and (image_target.exists() or label_target.exists()):
        raise FileExistsError(f"目标文件已存在: {image_target} 或 {label_target}")
    shutil.copy2(sample.image, image_target)
    shutil.copy2(sample.label, label_target)


def main() -> None:
    """随机复制配对样本到 train 和 val。"""
    if not 0 < TRAIN_RATIO < 1:
        raise ValueError("TRAIN_RATIO 必须在 0 和 1 之间")

    dataset = dataset_directory()
    print("[准备] 检查数据集和样本配对", flush=True)
    samples, missing_labels = find_samples(dataset)
    if not samples:
        raise RuntimeError("没有找到同时存在图片和标签的样本")

    print(f"[检查] 找到 {len(samples)} 个配对样本，缺少标签 {missing_labels} 张", flush=True)
    random.Random(RANDOM_SEED).shuffle(samples)
    train_count = round(len(samples) * TRAIN_RATIO)
    train_samples = samples[:train_count]
    val_samples = samples[train_count:]

    print("[处理] 开始复制 train/val 样本", flush=True)
    for sample in train_samples:
        copy_sample(sample, dataset, "train")
    for sample in val_samples:
        copy_sample(sample, dataset, "val")

    print("[完成] 数据集划分完成", flush=True)
    print(f"可用配对样本: {len(samples)}", flush=True)
    print(f"缺少标签的图片: {missing_labels}")
    print(f"复制到 train: {len(train_samples)}")
    print(f"复制到 val: {len(val_samples)}")
    print("unlabeled 源文件已保留。")


if __name__ == "__main__":
    main()
