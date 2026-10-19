#!/bin/bash

# 使用 TPU-MLIR 将 YOLO11 MaixCAM ONNX 模型转换为 INT8 cvimodel。
# 请在 TPU-MLIR Docker 容器的工作目录中运行本脚本。

set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
RESET='\033[0m'
START_TIME=$(date +%s)
CURRENT_STAGE="初始化"

stage() {
    CURRENT_STAGE="$1"
    printf "\\n${CYAN}[%s] %s${RESET}\\n" "$(date '+%H:%M:%S')" "$1"
}

finish() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        printf "${RED}失败: 阶段「%s」，退出码 %d${RESET}\\n" "${CURRENT_STAGE}" "${exit_code}" >&2
    fi
    exit "${exit_code}"
}

trap finish EXIT

net_name="template"
model_def="export_maixcam.onnx"
input_w=320
input_h=224
processor="cv181x"
output_names="out_box,out_cls"
test_input="test.jpg"
calibration_dir="images"
calibration_count=200

mean="0,0,0"
scale="0.00392156862745098,0.00392156862745098,0.00392156862745098"

required_files=("${model_def}" "${test_input}")
stage "检查工具和输入文件"
for command in model_transform.py model_deploy.py run_calibration.py; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "缺少 TPU-MLIR 命令: ${command}" >&2
        exit 1
    fi
done
for file in "${required_files[@]}"; do
    if [[ ! -f "${file}" ]]; then
        echo "缺少必需文件: ${file}" >&2
        exit 1
    fi
done

if [[ ! -d "${calibration_dir}" ]]; then
    echo "缺少校准图片目录: ${calibration_dir}" >&2
    exit 1
fi

calibration_images=$(find "${calibration_dir}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
    | wc -l)
if (( calibration_images == 0 )); then
    echo "校准图片目录为空: ${calibration_dir}" >&2
    exit 1
fi
if (( calibration_count > calibration_images )); then
    calibration_count=${calibration_images}
fi

stage "转换 ONNX 为 MLIR"
echo "模型: ${model_def}"
echo "输入: [1, 3, ${input_h}, ${input_w}]"
echo "输出节点: ${output_names}"
echo "处理器: ${processor}"
echo "校准图片: ${calibration_count} 张"

model_transform.py \
    --model_name "${net_name}" \
    --model_def "${model_def}" \
    --input_shapes "[[1,3,${input_h},${input_w}]]" \
    --mean "${mean}" \
    --scale "${scale}" \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --channel_format nchw \
    --output_names "${output_names}" \
    --test_input "${test_input}" \
    --test_result "${net_name}_top_outputs.npz" \
    --tolerance 0.99,0.99 \
    --mlir "${net_name}.mlir"

stage "生成 BF16 cvimodel"
model_deploy.py \
    --mlir "${net_name}.mlir" \
    --quantize BF16 \
    --processor "${processor}" \
    --test_input "${net_name}_in_f32.npz" \
    --test_reference "${net_name}_top_outputs.npz" \
    --model "${net_name}_bf16.cvimodel"

stage "生成 INT8 校准表"
run_calibration.py "${net_name}.mlir" \
    --dataset "${calibration_dir}" \
    --input_num "${calibration_count}" \
    -o "${net_name}_cali_table"

stage "生成 INT8 cvimodel"
model_deploy.py \
    --mlir "${net_name}.mlir" \
    --quantize INT8 \
    --quant_input \
    --calibration_table "${net_name}_cali_table" \
    --processor "${processor}" \
    --test_input "${net_name}_in_f32.npz" \
    --test_reference "${net_name}_top_outputs.npz" \
    --tolerance 0.9,0.6 \
    --model "${net_name}_int8.cvimodel"

cp "${net_name}_int8.cvimodel" "${net_name}.cvimodel"
elapsed=$(( $(date +%s) - START_TIME ))
printf "${GREEN}完成: %s（耗时 %ss）${RESET}\\n" "${net_name}.cvimodel" "${elapsed}"
