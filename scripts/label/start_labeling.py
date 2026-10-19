"""启动 LabelImg 进行人工 YOLO 标注。"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import yaml

# 运行前修改本区域配置。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "template"
LABEL_TARGET = "unlabeled"  # "unlabeled" or "label_model"
LABEL_MODEL_SPLIT = "train"  # Used when LABEL_TARGET is "label_model".
LABELIMG_COMMAND = PROJECT_ROOT / ".venv" / "Scripts" / "labelImg.exe"


def dataset_root() -> Path:
    """返回配置的数据集目录。"""
    return PROJECT_ROOT / "datasets" / DATASET_NAME


def configuration_path() -> Path:
    """返回与标注目标匹配的数据配置。"""
    root = dataset_root()
    if LABEL_TARGET == "unlabeled":
        return root / "data.yaml"
    if LABEL_TARGET == "label_model":
        return root / "label_model" / "data.yaml"
    raise ValueError("LABEL_TARGET 必须是 'unlabeled' 或 'label_model'")


def annotation_directories() -> tuple[Path, Path]:
    """返回选定目标的图片和标签目录。"""
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
    """从数据集 YAML 文件加载有序类别名称。"""
    with path.open(encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream) or {}
    names = config.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [
            str(names[index])
            for index in sorted(names, key=lambda item: int(item))
        ]
    raise ValueError(f"配置中缺少有效的 names 字段: {path}")


def write_class_file(names: list[str]) -> Path:
    """写入临时 LabelImg 类别文件并返回路径。"""
    class_file = PROJECT_ROOT / ".labelimg-classes.txt"
    class_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    return class_file


def validate_paths(image_dir: Path, label_dir: Path, data_file: Path) -> None:
    """启动 LabelImg 前检查配置路径。"""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not data_file.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_file}")
    label_dir.mkdir(parents=True, exist_ok=True)


def patch_labelimg_compatibility() -> None:
    """修补 LabelImg 的 PyQt5 浮点滚动条兼容问题。"""
    patches = {
        "labelImg.labelImg": {
            "bar.setValue(bar.value() + bar.singleStep() * units)": (
                "bar.setValue(int(bar.value() + bar.singleStep() * units))"
            )
        },
        "libs.canvas": {
            (
                "p.drawRect(left_top.x(), left_top.y(), rect_width, "
                "rect_height)"
            ): (
                "p.drawRect(\n"
                "                int(left_top.x()), int(left_top.y()), "
                "int(rect_width), int(rect_height)\n"
                "            )"
            ),
            (
                "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), "
                "self.pixmap.height())"
            ): (
                "p.drawLine(\n"
                "                int(self.prev_point.x()), 0, "
                "int(self.prev_point.x()), "
                "self.pixmap.height()\n"
                "            )"
            ),
            (
                "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), "
                "self.prev_point.y())"
            ): (
                "p.drawLine(\n"
                "                0, int(self.prev_point.y()), "
                "self.pixmap.width(), "
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
    """使用选定的数据集目录启动 LabelImg。"""
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
    print("[准备] 启动人工标注工具", flush=True)
    print(f"图片目录: {image_dir}", flush=True)
    print(f"标签目录: {label_dir}")
    print(f"类别配置: {data_file}")
    if not LABELIMG_COMMAND.is_file():
        raise FileNotFoundError(
            f"LabelImg executable not found: {LABELIMG_COMMAND}"
        )
    subprocess.run(command, check=True)
    print("[完成] LabelImg 已退出", flush=True)


if __name__ == "__main__":
    main()
