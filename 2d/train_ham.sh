#!/bin/bash
BASE_DIR=DIRECTORY/0_dataset/Fairness/HAM10000
LIST_DIR=DIRECTORY/0_dataset/Fairness/HAM10000/split
BATCH_SIZE=42
CENTER_CROP_SIZE=1024
NUM_CLASS=5
MAX_EPOCHS=100
STOP_EPOCH=101
ATTRIBUTE=age
LEARNING_RATE=0.0175

CUDA_VISIBLE_DEVICES=0 python train.py \
	--output DIRECTORY/3_output/18.1_dMOE/dmoe/HAM_${ATTRIBUTE} \
	--root_path ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${MAX_EPOCHS} \
	--batch_size ${BATCH_SIZE} \
	--attribute ${ATTRIBUTE} \
	--dmoe True