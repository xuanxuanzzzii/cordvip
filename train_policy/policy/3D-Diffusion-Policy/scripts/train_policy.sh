# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer_pointcloud 0112 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer_pointcloud 0112 0 0


DEBUG=False
save_ckpt=True

alg_name=${1}
# task choices: See TASK.md
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
num_epochs=${6}
checkpoint_every=${7}
expert_data_num=${8}
horizon=${9}
n_obs_steps=${10}
n_action_steps=${11}

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

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.num_epochs=${num_epochs} \
                            training.checkpoint_every=${checkpoint_every} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            horizon=${horizon} \
                            n_obs_steps=${n_obs_steps} \
                            n_action_steps=${n_action_steps} \
