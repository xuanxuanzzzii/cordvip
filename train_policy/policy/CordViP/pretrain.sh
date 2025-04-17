#!/bin/bash
# bash train.sh mug_hanging 10 0 0

task_name=${1}
expert_data_num=${2}
seed=${3}
gpu_id=${4}
num_epochs=${5}
checkpoint_every=${6}
n_obs_steps=${7}
encoder=${8}

bash scripts/pretrain_policy.sh CordViP_pretrain ${task_name}_${expert_data_num} train ${seed} ${gpu_id} ${num_epochs} ${checkpoint_every} ${expert_data_num} ${n_obs_steps} ${encoder}
