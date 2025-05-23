#!/bin/bash
BASE_DIR=DIRECTORY/0_dataset/Fairness/HAM10000
LIST_DIR=DIRECTORY/0_dataset/Fairness/HAM10000/split
BATCH_SIZE=42
CENTER_CROP_SIZE=1024
IMAGE_SIZE=224
NUM_CLASS=2
MAX_EPOCHS=100 
STOP_EPOCH=100
ATTRIBUTE=age

# DMOE
CUDA_VISIBLE_DEVICES=0 python test_ham.py \
	--output DIRECTORY/3_output/18.1_dMOE/dmoe/HAM_${ATTRIBUTE} \
	--datadir ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} --img_size ${IMAGE_SIZE} \
	--num_classes ${NUM_CLASS} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${STOP_EPOCH} \
	--batch_size ${BATCH_SIZE} \
	--attribute ${ATTRIBUTE} \
	--dmoe True 
