"""Generate YOLO labels for unlabeled images with a trained small model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from tqdm import tqdm

# Edit these values before running the script.
PROJECT_NAME = "auto_aim"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
OVERWRITE_LABELS = False
BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "auto_label"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def config_path() -> Path:
    """Return the automatic-labeling training configuration path."""
    return CONFIG_ROOT / PROJECT_NAME / "train.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load an automatic-labeling YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"自动打标配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML 映射: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def dataset_root(data_path: Path) -> Path:
    """Return the main dataset root from the label-model data YAML path."""
    label_model_root = data_path.parent
    if label_model_root.name != "label_model":
        raise ValueError(f"自动打标数据配置必须位于 label_model 目录: {data_path}")
    return label_model_root.parent


def model_path(config: dict[str, Any]) -> Path:
    """Locate the trained automatic-labeling model checkpoint."""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("name", "train"))
    return PROJECT_ROOT / "runs" / "auto_label" / project / experiment / "weights" / "best.pt"


def write_labels(result: Any, label_path: Path) -> bool:
    """Write normalized YOLO detections and return whether detections exist."""
    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        coordinates = boxes.xywhn.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        detections = [
            f"{int(class_id)} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
            for class_id, (x, y, width, height) in zip(classes, coordinates)
        ]

    label_path.write_text(
        "\n".join(detections) + ("\n" if detections else ""), encoding="utf-8"
    )
    return bool(detections)


def main() -> None:
    """Predict labels for all unlabeled images."""
    if not 0 <= CONFIDENCE_THRESHOLD <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD 必须在 0 和 1 之间")
    if not 0 <= IOU_THRESHOLD <= 1:
        raise ValueError("IOU_THRESHOLD 必须在 0 和 1 之间")
    if not torch.cuda.is_available():
        raise RuntimeError("本机自动打标需要 CUDA device=0。")

    config = load_config(config_path())
    data_path = resolve_path(config["data"])
    checkpoint = model_path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"自动打标小模型不存在: {checkpoint}")

    dataset = dataset_root(data_path)
    image_dir = dataset / "images" / "unlabeled"
    label_dir = dataset / "labels" / "unlabeled"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"待标注图片目录不存在: {image_dir}")
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"待标注图片目录为空: {image_dir}", flush=True)
        return

    model = YOLO(str(checkpoint))
    print(f"模型: {checkpoint}", flush=True)
    print(f"输入目录: {image_dir}", flush=True)
    print(f"输出目录: {label_dir}", flush=True)
    print(f"设备: CUDA:0 ({torch.cuda.get_device_name(0)})", flush=True)

    generated = 0
    skipped = 0
    empty = 0
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=config.get("imgsz", [320, 320]),
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=config.get("device", 0),
        batch=BATCH_SIZE,
        save=False,
        verbose=False,
        stream=True,
    )
    with tqdm(total=len(images), desc="自动打标", unit="image") as progress:
        for image_path, result in zip(images, results):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists() and not OVERWRITE_LABELS:
                skipped += 1
            else:
                has_detections = write_labels(result, label_path)
                generated += 1
                empty += int(not has_detections)
            progress.update(1)
            progress.set_postfix(
                generated=generated, skipped=skipped, empty=empty
            )

    print(f"图片总数: {len(images)}", flush=True)
    print(f"生成标签: {generated}", flush=True)
    print(f"跳过已有标签: {skipped}", flush=True)
    print(f"无检测结果: {empty}", flush=True)


if __name__ == "__main__":
    main()
