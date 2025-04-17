#!/bin/bash

TASK_NAME="Assembly"
CKPT_DIR=./checkpoints/Assembly
POLICY_CLASS="ACT"
KL_WEIGHT=100
CHUNK_SIZE=30
HIDDEN_DIM=512
BATCH_SIZE=128
DIM_FEEDFORWARD=3200
NUM_EPOCHS=6000
LR=10e-5
SEED=0
IF_TRAIN=true
PRETRAIN_CKPT_DIR="/home/fqx/RoboTwin/policy/CoDP/CoDP/checkpoints/CoDP_Pretrain_pointnet_all_tasks_5hz_4/2000.ckpt"
CONFIG_PATH="/home/fqx/code_release/policy/act_3d_ours/detr/models/encoder.yaml"

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
    --seed $SEED \
    --if_train $IF_TRAIN \
    --pretrain_ckpt_dir $PRETRAIN_CKPT_DIR \
    --config_path $CONFIG_PATH \

