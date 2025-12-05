#!/bin/bash

PROMPTS="benchmark/coco_30k_10k.csv"
OUTPUT_DIR="benchmark/results"
GPU="0"

# 평가할 이미지 디렉토리 목록 (원하면 추가/삭제 가능)
IMAGE_DIRS=(
    "benchmark/generations/openjourney_coco_small_with_seed/gs1.5"
    "benchmark/generations/openjourney_coco_small_with_seed/gs2"
    "benchmark/generations/openjourney_coco_small_with_seed/gs3"
    "benchmark/generations/openjourney_coco_small_with_seed/gs4"
    "benchmark/generations/openjourney_coco_small_with_seed/gs5"
    #"benchmark/generations/sd2-1_coco_small_with_seed/gs1.5"
    #"benchmark/generations/sd2-1_coco_small_with_seed/gs2"
    #"benchmark/generations/sd2-1_coco_small_with_seed/gs3"
    #"benchmark/generations/sd2-1_coco_small_with_seed/gs4"
    #"benchmark/generations/sd2-1_coco_small_with_seed/gs5"
    #"benchmark/generations/sd2-1_coco"
    #"benchmark/generations/openjourney_coco"
    #"benchmark/generations/sdxl_coco"
    #"benchmark/generations/vd_coco"
)

for IMG_DIR in "${IMAGE_DIRS[@]}"; do
    echo "============================================"
    echo " Running COCO test for: $IMG_DIR"
    echo "============================================"

    python3 test_coco.py \
        --prompts_path "$PROMPTS" \
        --images_dir "$IMG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id "$GPU"

    echo ""
    echo " Finished: $IMG_DIR"
    echo
done
