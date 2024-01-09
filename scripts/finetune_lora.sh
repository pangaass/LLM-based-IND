#! /usr/bin/env bash
NVIDIA_VISIBLE_DEVICES=1,2,3,4
set -ex

LR=1e-5
NUM_GPUS=1
LORA_RANK=8
LORA_ALPHA=32
LORA_DROUPOUT=0.1

MAX_SOURCE_LEN=8000
MAX_TARGET_LEN=16
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=2
MAX_STEP=10000
SAVE_INTERVAL=5000

RUN_NAME=text
BASE_MODEL_PATH=/workspace/models/ZhipuAI/chatglm3-6b
AUTHOR_PATH=/workspace/dataset/whoiswho/train_pub.json
PUB_PATH=/workspace/dataset/whoiswho/train_pub.json
TRAIN_PATH=/workspace/source_code/finetune_basemodel_demo/dataset/train_data.json
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --author_data $AUTHOR_PATH \
    --pub_data $PUB_PATH \
    --train_data $TRAIN_PATH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROUPOUT \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR  2>&1 | tee ${OUTPUT_DIR}/train.log

