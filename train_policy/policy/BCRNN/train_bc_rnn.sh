# Examples:
# bash train.sh robot_dp dual_bottles_pick_easy 0 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer_pointcloud 0112 0 0


task_name=${1}
expert_data_num=${2}
seed=${3}
gpu_id=${4}
epochs=${5}
checkpoint_every=${6}

horizon=${7}
n_obs_steps=${8}
n_action_steps=${9}

crop_h=${10}
crop_w=${11}

DEBUG=False
save_ckpt=True

alg_name=robot_bc_rnn
# task choices: See TASK.md
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_bc_rnn-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="/home/fqx/CordViP_code/code_release/policy/Diffusion-Policy/data/${task_name}_${expert_data_num}.zarr" \
                            expert_data_num=${expert_data_num} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.num_epochs=${epochs} \
                            training.checkpoint_every=${checkpoint_every} \
                            training.device="cuda:0" \
                            horizon=${horizon} \
                            n_obs_steps=${n_obs_steps} \
                            n_action_steps=${n_action_steps} \
                            crop_h=${crop_h} \
                            crop_w=${crop_w} \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode}