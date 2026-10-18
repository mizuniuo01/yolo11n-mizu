#!/bin/bash

# Convert a YOLO11 MaixCAM ONNX model with TPU-MLIR.
# Run this script inside the TPU-MLIR Docker container from /workspace.

set -euo pipefail

net_name="auto_aim"
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
for file in "${required_files[@]}"; do
    if [[ ! -f "${file}" ]]; then
        echo "missing required file: ${file}" >&2
        exit 1
    fi
done

if [[ ! -d "${calibration_dir}" ]]; then
    echo "missing calibration directory: ${calibration_dir}" >&2
    exit 1
fi

calibration_images=$(find "${calibration_dir}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
    | wc -l)
if (( calibration_images == 0 )); then
    echo "no calibration images found in ${calibration_dir}" >&2
    exit 1
fi
if (( calibration_count > calibration_images )); then
    calibration_count=${calibration_images}
fi

echo "model: ${model_def}"
echo "input: [1, 3, ${input_h}, ${input_w}]"
echo "outputs: ${output_names}"
echo "processor: ${processor}"
echo "calibration images: ${calibration_count}"

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

model_deploy.py \
    --mlir "${net_name}.mlir" \
    --quantize BF16 \
    --processor "${processor}" \
    --test_input "${net_name}_in_f32.npz" \
    --test_reference "${net_name}_top_outputs.npz" \
    --model "${net_name}_bf16.cvimodel"

echo "running INT8 calibration"
run_calibration.py "${net_name}.mlir" \
    --dataset "${calibration_dir}" \
    --input_num "${calibration_count}" \
    -o "${net_name}_cali_table"

echo "building INT8 cvimodel"
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
echo "导出完成: ${net_name}.cvimodel"
