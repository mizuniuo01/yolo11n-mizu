"""Start LabelImg for manual YOLO annotation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Edit this section before running the script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "template"
LABEL_TARGET = "unlabeled"  # "unlabeled" or "label_model"
LABEL_MODEL_SPLIT = "train"  # Used when LABEL_TARGET is "label_model".
LABELIMG_COMMAND = PROJECT_ROOT / ".venv" / "Scripts" / "labelImg.exe"


def dataset_root() -> Path:
    """Return the configured dataset directory."""
    return PROJECT_ROOT / "datasets" / DATASET_NAME


def configuration_path() -> Path:
    """Return the data configuration matching the selected target."""
    root = dataset_root()
    if LABEL_TARGET == "unlabeled":
        return root / "data.yaml"
    if LABEL_TARGET == "label_model":
        return root / "label_model" / "data.yaml"
    raise ValueError("LABEL_TARGET 必须是 'unlabeled' 或 'label_model'")


def annotation_directories() -> tuple[Path, Path]:
    """Return image and label directories for the selected target."""
    root = dataset_root()
    if LABEL_TARGET == "unlabeled":
        return root / "images" / "unlabeled", root / "labels" / "unlabeled"
    if LABEL_MODEL_SPLIT not in {"train", "val"}:
        raise ValueError("LABEL_MODEL_SPLIT 必须是 'train' 或 'val'")
    label_model = root / "label_model"
    return (
        label_model / "images" / LABEL_MODEL_SPLIT,
        label_model / "labels" / LABEL_MODEL_SPLIT,
    )


def load_class_names(path: Path) -> list[str]:
    """Load ordered class names from a dataset YAML file."""
    with path.open(encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream) or {}
    names = config.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda item: int(item))]
    raise ValueError(f"配置中缺少有效的 names 字段: {path}")


def write_class_file(names: list[str]) -> Path:
    """Write a temporary LabelImg class list and return its path."""
    class_file = PROJECT_ROOT / ".labelimg-classes.txt"
    class_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    return class_file


def validate_paths(image_dir: Path, label_dir: Path, data_file: Path) -> None:
    """Validate configured paths before starting LabelImg."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not data_file.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_file}")
    label_dir.mkdir(parents=True, exist_ok=True)


def patch_labelimg_compatibility() -> None:
    """Patch LabelImg's PyQt5 float scroll-bar compatibility issue."""
    patches = {
        "labelImg.labelImg": {
            "bar.setValue(bar.value() + bar.singleStep() * units)": (
                "bar.setValue(int(bar.value() + bar.singleStep() * units))"
            )
        },
        "libs.canvas": {
            "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)": (
                "p.drawRect(\n"
                "                int(left_top.x()), int(left_top.y()), "
                "int(rect_width), int(rect_height)\n"
                "            )"
            ),
            "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())": (
                "p.drawLine(\n"
                "                int(self.prev_point.x()), 0, int(self.prev_point.x()), "
                "self.pixmap.height()\n"
                "            )"
            ),
            "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())": (
                "p.drawLine(\n"
                "                0, int(self.prev_point.y()), self.pixmap.width(), "
                "int(self.prev_point.y())\n"
                "            )"
            ),
        },
    }
    for module_name, replacements in patches.items():
        module = importlib.util.find_spec(module_name)
        if module is None or module.origin is None:
            raise FileNotFoundError(f"无法定位 LabelImg 源文件: {module_name}")
        source_path = Path(module.origin)
        source = source_path.read_text(encoding="utf-8")
        patched_source = source
        for old_code, new_code in replacements.items():
            patched_source = patched_source.replace(old_code, new_code)
        if patched_source != source:
            source_path.write_text(patched_source, encoding="utf-8")
            print(f"已应用 LabelImg PyQt5 兼容补丁: {source_path}")


def main() -> None:
    """Start LabelImg with the selected dataset directories."""
    image_dir, label_dir = annotation_directories()
    data_file = configuration_path()
    validate_paths(image_dir, label_dir, data_file)
    patch_labelimg_compatibility()
    names = load_class_names(data_file)
    class_file = write_class_file(names)

    command = [
        str(LABELIMG_COMMAND),
        str(image_dir),
        str(class_file),
        str(label_dir),
    ]
    print(f"图片目录: {image_dir}")
    print(f"标签目录: {label_dir}")
    print(f"类别配置: {data_file}")
    if not LABELIMG_COMMAND.is_file():
        raise FileNotFoundError(f"LabelImg executable not found: {LABELIMG_COMMAND}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
