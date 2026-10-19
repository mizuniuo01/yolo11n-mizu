"""将训练好的 YOLO11 模型导出为 ONNX 或 TensorRT。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

# 选择 configs/export/ 下的工程配置。
PROJECT_NAME = "template"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "export"


def config_path() -> Path:
    """返回选定工程的导出配置路径。"""
    return CONFIG_ROOT / PROJECT_NAME / "export.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """加载并检查导出 YAML 配置。"""
    if not path.is_file():
        raise FileNotFoundError(f"导出配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"导出配置必须是 YAML 映射: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """将工程相对路径解析为仓库根目录下的路径。"""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def checkpoint_path(config: dict[str, Any]) -> Path:
    """返回由工程和实验确定的训练权重。"""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("experiment", "exp01"))
    return (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / project
        / experiment
        / "weights"
        / "best.pt"
    )


def output_directory(config: dict[str, Any]) -> Path:
    """返回并创建通用导出目录。"""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("experiment", "exp01"))
    directory = PROJECT_ROOT / "runs" / "export" / project / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_config(config: dict[str, Any]) -> tuple[Path, str]:
    """检查导出格式、训练权重和 CUDA 要求。"""
    checkpoint = checkpoint_path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"训练权重不存在: {checkpoint}")

    export_format = str(config.get("format", "onnx")).lower()
    if export_format not in {"onnx", "engine"}:
        raise ValueError("format 只能是 'onnx' 或 'engine'")
    if not torch.cuda.is_available():
        raise RuntimeError("本机通用导出需要 CUDA device=0。")
    return checkpoint, export_format


def export_arguments(
    config: dict[str, Any], export_format: str
) -> dict[str, Any]:
    """根据配置构造 Ultralytics 导出参数。"""
    arguments = {
        "format": export_format,
        "imgsz": config.get("imgsz", [640, 640]),
        "dynamic": bool(config.get("dynamic", False)),
        "simplify": bool(config.get("simplify", True)),
        "opset": config.get("opset"),
        "device": config.get("device", 0),
    }
    if export_format == "engine":
        arguments["half"] = bool(config.get("half", True))
    if export_format == "engine" and arguments["device"] != 0:
        raise ValueError("TensorRT 导出必须使用 device: 0")
    return arguments


def artifact_path(exported: str | Path, export_format: str) -> Path:
    """解析 Ultralytics 返回的导出产物路径。"""
    path = Path(exported)
    if path.is_file():
        return path
    suffix = ".engine" if export_format == "engine" else ".onnx"
    candidate = path.with_suffix(suffix)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"导出产物不存在: {exported}")


def main() -> None:
    """导出选定模型并将产物复制到 runs/export。"""
    config = load_config(config_path())
    checkpoint, export_format = validate_config(config)
    arguments = export_arguments(config, export_format)
    output = output_directory(config)

    print("[准备] 通用模型导出配置", flush=True)
    print(f"输入权重: {checkpoint}", flush=True)
    print(f"导出格式: {export_format}", flush=True)
    print(f"导出参数: {arguments}", flush=True)
    print(f"输出目录: {output}", flush=True)

    print("[处理] 开始导出模型", flush=True)
    model = YOLO(str(checkpoint))
    exported = model.export(**arguments)
    source = artifact_path(exported, export_format)
    target = output / (
        "export_standard.onnx"
        if export_format == "onnx"
        else "export_standard.engine"
    )
    shutil.copy2(source, target)
    print(f"[完成] 导出完成: {target}", flush=True)


if __name__ == "__main__":
    main()
