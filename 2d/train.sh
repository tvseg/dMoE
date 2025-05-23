#!/bin/bash
BASE_DIR=DIRECTORY/0_dataset/Fairness/Harvard-FairSeg/Training
LIST_DIR=DIRECTORY/1_code/dMoE/2d/lists/FairSeg_final
BATCH_SIZE=42
CENTER_CROP_SIZE=512
NUM_CLASS=3
MAX_EPOCHS=300
STOP_EPOCH=130
ATTRIBUTE=race

CUDA_VISIBLE_DEVICES=0 python train.py \
	--output DIRECTORY/3_output/18.1_dMOE/dmoe/OCT_${ATTRIBUTE} \
	--root_path ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${STOP_EPOCH} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--attribute ${ATTRIBUTE} \
	--dmoe True \
