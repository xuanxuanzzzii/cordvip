#!/bin/bash

TASK_NAME="flip_cap"
CKPT_DIR=./checkpoints/flip_cap
POLICY_CLASS="ACT"
KL_WEIGHT=100
CHUNK_SIZE=30
HIDDEN_DIM=512
BATCH_SIZE=128
DIM_FEEDFORWARD=3200
NUM_EPOCHS=6000
LR=10e-5
SEED=0


python imitate_episodes.py \
    --task_name $TASK_NAME \
    --ckpt_dir $CKPT_DIR \
    --policy_class $POLICY_CLASS \
    --kl_weight $KL_WEIGHT \
    --chunk_size $CHUNK_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE \
    --dim_feedforward $DIM_FEEDFORWARD \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --seed $SEED
