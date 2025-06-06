
task_name=${1}
expert_data_num=${2}
checkpoint_num=${3}
seed=${4}
gpu_id=${5}
encoder=${6}
alg_name=${7}
config_name=${alg_name}

addition_info=eval
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./policy/CordViP/CordViP/CordViP/data/outputs/${exp_name}_seed${seed}"

DEBUG=False
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..
python script/eval_policy_CordViP.py --config-name=${config_name}.yaml \
                            task=${task_name}_${expert_data_num} \
                            raw_task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint_num=${checkpoint_num} \
                            expert_data_num=${expert_data_num} \
                            policy.pointnet_type=${encoder}