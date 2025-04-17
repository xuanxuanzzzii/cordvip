#!/bin/bash
# bash train.sh mug_hanging 10 0 0

task_name=${1}
expert_data_num=${2}
seed=${3}
gpu_id=${4}
num_epochs=${5}
checkpoint_every=${6}
horizon=${7}
n_obs_steps=${8}
n_action_steps=${9}
encoder=${10}
freeze_encoder=${11}
load_checkpoint_number=${12}
load_checkpoint_name=${13}

bash scripts/load_pretrain_policy.sh CordViP_train ${task_name}_${expert_data_num} train ${seed} ${gpu_id} ${num_epochs} ${checkpoint_every} ${expert_data_num} ${horizon} ${n_obs_steps} ${n_action_steps} ${encoder} ${freeze_encoder} ${load_checkpoint_number} ${load_checkpoint_name}

