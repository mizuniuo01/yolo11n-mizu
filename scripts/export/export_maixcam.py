"""导出并截断适用于 MaixCAM 的 YOLO11 ONNX 模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import torch
import yaml
from ultralytics import YOLO

# 选择 configs/export/ 下的工程配置。
PROJECT_NAME = "template"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "export"


def config_path() -> Path:
    """返回选定工程的 MaixCAM 配置路径。"""
    return CONFIG_ROOT / PROJECT_NAME / "maixcam.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """加载并检查 MaixCAM 导出配置。"""
    if not path.is_file():
        raise FileNotFoundError(f"MaixCAM 配置不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML 映射: {path}")
    return config


def checkpoint_path(config: dict[str, Any]) -> Path:
    """返回由工程和实验确定的训练权重。"""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("experiment", "exp01"))
    return PROJECT_ROOT / "runs" / "detect" / project / experiment / "weights" / "best.pt"


def output_path(config: dict[str, Any]) -> Path:
    """返回截断后的 MaixCAM ONNX 路径。"""
    project = str(config.get("project", PROJECT_NAME))
    experiment = str(config.get("experiment", "exp01"))
    directory = PROJECT_ROOT / "runs" / "export" / project / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "export_maixcam.onnx"


def truncate_outputs(
    source: Path,
    target: Path,
    input_name: str,
    source_outputs: list[str],
    output_names: list[str],
) -> None:
    """提取官方 YOLO11 Detect 输出并重命名。"""
    if len(source_outputs) != 2 or len(output_names) != 2:
        raise ValueError("YOLO11 MaixCAM 必须配置两个输入输出节点")

    model = onnx.load(str(source))
    graph_values = {
        value.name
        for value in (*model.graph.value_info, *model.graph.output)
    }
    graph_values.update(
        output
        for node in model.graph.node
        for output in node.output
    )
    missing = [name for name in source_outputs if name not in graph_values]
    if missing:
        raise RuntimeError(
            "YOLO11 Detect 输出节点不存在，请用 Netron 检查导出图: "
            + ", ".join(missing)
        )

    extracted = target.with_suffix(".tmp.onnx")
    onnx.utils.extract_model(
        str(source),
        str(extracted),
        [input_name],
        source_outputs,
    )
    truncated = onnx.load(str(extracted))
    if len(truncated.graph.output) != 2:
        raise RuntimeError("YOLO11 MaixCAM 截断结果不是两个输出")
    for value, name in zip(truncated.graph.output, output_names):
        old_name = value.name
        for node in truncated.graph.node:
            node.output[:] = [
                name if output == old_name else output
                for output in node.output
            ]
        value.name = name
    onnx.checker.check_model(truncated)
    onnx.save(truncated, str(target))
    extracted.unlink()


def main() -> None:
    """导出并截断配置的 YOLO11 权重。"""
    config = load_config(config_path())
    checkpoint = checkpoint_path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"训练权重不存在: {checkpoint}")

    imgsz = config.get("imgsz", [224, 320])
    if not isinstance(imgsz, list) or len(imgsz) != 2:
        raise ValueError("imgsz 必须是 [height, width]")
    if any(int(size) <= 0 or int(size) % 32 != 0 for size in imgsz):
        raise ValueError("imgsz 的 height 和 width 必须是 32 的倍数")

    target = output_path(config)
    source_outputs = config.get("source_output_names")
    output_names = config.get("output_names")
    input_name = str(config.get("input_name", "images"))
    if not isinstance(source_outputs, list) or not isinstance(output_names, list):
        raise ValueError("必须配置 source_output_names 和 output_names")
    source_outputs = [str(name) for name in source_outputs]
    output_names = [str(name) for name in output_names]
    export_args = {
        "format": "onnx",
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "opset": int(config.get("opset", 17)),
        "simplify": bool(config.get("simplify", True)),
        "dynamic": bool(config.get("dynamic", False)),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("本机 MaixCAM 导出需要 CUDA device=0。")

    print("[准备] MaixCAM 模型导出配置", flush=True)
    print(f"输入权重: {checkpoint}", flush=True)
    print(f"导出参数: {export_args}", flush=True)
    print(f"输出文件: {target}", flush=True)
    print("[处理] 导出标准 ONNX 并截断 YOLO11 Detect 头", flush=True)
    exported = YOLO(str(checkpoint)).export(**export_args)
    source = Path(exported)
    if not source.is_file():
        raise FileNotFoundError(f"标准 ONNX 导出产物不存在: {source}")
    truncate_outputs(
        source,
        target,
        input_name,
        source_outputs,
        output_names,
    )
    print(f"[完成] 导出完成: {target}", flush=True)


if __name__ == "__main__":
    main()
