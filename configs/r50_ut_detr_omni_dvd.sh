#!/usr/bin/env bash

set -x

EXP_DIR=results/sdugamod/aerial/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'aerial/aligned_ids/aerial_train_aligned_ids_w_indicator.json' \
    --annotation_json_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
    --annotation_json_aerial_label 'aerial/aligned_ids/aerial_train_aligned_ids_w_perspective.json' \
    --annotation_json_aerial_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
    --annotation_json_ground_label 'ground/aligned_ids/ground_train_aligned_ids_w_perspective.json' \
    --annotation_json_ground_unlabel 'ground_8605_scaled_labels_w_indicator_aligned_ids.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path '../../../data/jlorray1/SDU-GAMODv4/' \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --pixels 600 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'dvd' \
#    --dataset_file '/data/jlorray1/SDU-GAMODv4' \
#    --dataset_file 'sdugamod_aerial' \
#    --data_path './dvd' \
