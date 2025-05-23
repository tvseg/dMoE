#!/bin/bash
BASE_DIR=DIRECTORY/0_dataset/Fairness/Harvard-FairSeg/Training
LIST_DIR=DIRECTORY/1_code/dMoE/2d/lists/FairSeg_final
BATCH_SIZE=42
CENTER_CROP_SIZE=512
STOP_EPOCH=100
NUM_CLASS=3
ATTRIBUTE=race


# DMOE
CUDA_VISIBLE_DEVICES=0 python test.py \
	--output DIRECTORY/3_output/18.1_dMOE/dmoe/OCT_${ATTRIBUTE} \
	--datadir ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--stop_epoch ${STOP_EPOCH} \
	--batch_size ${BATCH_SIZE} \
	--dmoe True\
	--attribute ${ATTRIBUTE} \
	