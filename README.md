# CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[🌐**Project Page**](https://aureleopku.github.io/CordViP/) | [✍️**Paper(Arxiv)**](https://arxiv.org/abs/2502.08449) 

[Yankai Fu](https://github.com/AureleoPKU)\*, [Qiuxuan Feng](https://github.com/xuanxuanzzzii)\*, [Ning Chen](https://github.com/ccdcs)*, [Zichen Zhou](https://bochiman.github.io/), [Mengzhen Liu](/lmz), [Mingdong Wu](https://aaronanima.github.io/), [Tianxing Chen](https://tianxingchen.github.io/), [Shanyu Rong](https://rainfallsdown.github.io/),  [Jiaming Liu](https://liujiaming1996.github.io/), [Hao Dong](https://zsdonghao.github.io/), [Shanghang Zhang](https://www.shanghangzhang.com)



We propose **CordViP**🤖, a correspondence-based visuomotor policy for dexterous manipulation in the real world.

## ✨ News ✨

- [2025/4/11] CordViP has been accepted to RSS 2025! 🎉
- [2025/2/13] CordViP is now live on arXiv! 🚀 



## 📢 Repository Contents

- **Code For CordViP 📈**: Code for data collection, training, and evaluation **in Real-World**. 
- **CordViP Model** 🎯:  We provide a [pretrained checkpoint](https://huggingface.co/FengQiuxuan/CordViP) that is trained on data from four tasks: pickplace, flipcup, assembly, and artimanip.
- **Real-World Dataset** 🎯:  We provide real-world data for the task of  [flipcap](https://huggingface.co/datasets/FengQiuxuan/CordViP) , which can be used as a reference for the data format.
- **2D and 3D Baseline Methods 🛠️**: Provides standard 2D and 3D baseline methods for comparison:
  - **2D Baseline**: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [act](https://github.com/tonyzhaozh/act/tree/main), [BCRNN](https://robomimic.github.io/)
  - **3D Baseline**: [3D Diffusion Policy](https://3d-diffusion-policy.github.io/), [act 3D](https://github.com/tonyzhaozh/act/tree/main), [BCRNN 3D](https://robomimic.github.io/)
  - **state-base Baseline**: state-base Diffusion Policy, State-base CNNMLP



## 📦 Installation

#### Environment for data generation


```bash
# create conda environment
conda create -n posetrack python=3.9

# activate conda environment
conda activate posetrack

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
cd generate_pc
python -m pip install -r requirements.txt

# Install NVDiffRast
cd FoundationPose
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

pip install pytorch_kinematics==0.7.4
```



#### Environment for training and evaluation

<details>
<summary>1. Prepare a conda environment</summary>


```bash
conda create -n CordViP python=3.8
conda activate CordViP
cd CordViP_code/train_policy
```

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic openai huggingface_hub==0.25.0 zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor robomimic easydict

cd act/detr && pip install -e .
```

</details>

<details>
<summary>2. Install pytorch3d</summary>


```bash
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

</details>



## 🤖️ Real-world setting

In our real-world setup, our system consists of a Leap Hand and a UR5 Arm, with a fixed Realsense L515 camera employed to capture visual observation. The Realsense D435 camera is only used for data collection during teleoperation and is not involved in the policy learning.



## 💡Usage

### Point Cloud Generation.

Our method use FoundationPose and D(R,O) Grasp to track pose. 

**1. Use TripoSR to generate object mesh.**  

refer to: https://github.com/VAST-AI-Research/TripoSR

**2. Use FoundationPose and D(R,O) Grasp to track pose..**  

1. Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `generate_pc/FoundationPose/weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

2. [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `generate_pc/FoundationPose/demo_data/`

3. In FoundationPose, the mask of the tracked object is obtained by clicking on the object in the initial frame, and press the `q` key to start tracking..

After the preparations above , simply run:

```bash
cd CordViP_code/generate_pc/
conda activate posetrack
bash preprocess.sh ${demo_path} ${idx} ${mesh_path}
# As example: bash preprocess.sh /home/alan/project/fyk/cordvip/expert_dataset/flipcup 0 assets/mesh/flipcup/combine_2.obj
```

to generate point cloud dataset.



### Steps to train a policy on an existing environment.

```bash
conda activate CordViP
# cd CordViP_code/train_policy
```

To go through the whole process, there are 3️⃣ necessary steps:

**1. Generate dataset.**  

```bash
# step1 
python contact.py ${load_dir} ${num} ${current_ep} ${num_files}
# As example: python contact.py /load_dir 50 0 380
# step2
python script/pkl2zarr_cordvip.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_cordvip.py assembly 100
```

**2. Train Policy.**  

**step1:  pretrain the encoder.**  

```bash
# step1: pretrain
cd policy/CordViP
sh pretrain.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${num_epochs} ${checkpoint_every} ${n_obs_steps} ${encoder}
# As example: sh pretrain.sh all_tasks 160 42 1 10000 1000 4 pointnet
```

**step2:  load the pretrained encoder and fine-tune on specific task**  

🪄**Diffusion-based**

```bash
cd policy/CordViP
sh load_pretrain.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${num_epochs} ${checkpoint_every} ${horizon} ${n_obs_steps} ${n_action_steps} ${encoder} ${freeze_encoder}${load_checkpoint_number} ${load_checkpoint_name}
# As example: sh load_pretrain.sh Assembly 50 42 0 20000 1000 12 4 6 pointnet false 2000 CordViP_pretrain
# the path to load the pretrain checkpoint: ./CordViP/checkpoints/{cfg.load_checkpoint_name}/{cfg.load_checkpoint_number}.ckpt
# We provide a pretrained checkpoint that is trained on data from four tasks: pickplace, flipcup, assembly, and artimanip.
```

🪄**Transformer-based**

```bash
cd policy/act_3d_ours
# step1
define task config in aloha_script/constants.py
As example: 'Articulated_manip':{
        'dataset_dir': '/data/Articulated_manip_50.zarr',
        'num_episodes': 50,
        'episode_len': 150,
        'camera_names': ['cam0']
    }
# step2
sh train.sh
```

**3. Evaluate in Real-World**

🪄**Diffusion-based**

```bash
cd policy/CordViP
sh eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id} ${encoder} ${alg_name}
# As example: sh eval.sh Assembly 50 8000 42 0 pointnet CordViP_train
```

🪄**Transformer-based**

```bash
python eval_policy_act_3d_ours.py --ckpt_dir ${ckpt_dir} --policy_class ${policy_class} --task_name ${task_name} --seed ${seed} --num_epochs ${num_epochs}
# As example: python eval_policy_act_3d_ours.py --ckpt_dir ../policy/act_3d_ours/checkpoints/assembly --policy_class ACT --task_name assembly --seed 0 --num_epochs 2000
```





# 🚴‍♂️ Baselines

## 1. Diffusion Policy

The DP code can be found in `policy/Diffusion-Policy`.

```bash
python script/pkl2zarr_dp.py ${task_name} ${expert_data_num} 
# As example: python script/pkl2zarr_dp.py assembly 100 
```

Then, move to `policy/Diffusion-Policy` first, and run the following code to train DP:

```bash
bash train.sh Assembly ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${epochs} ${checkpoint_every} ${horizon} ${n_obs_steps} ${n_action_steps} ${crop_h} ${crop_w}
# As example: bash train.sh Assembly 50 42 3 2000 100 12 4 8 480 640
```

Run the following code to evaluate DP for a specific task:

```bash
cd policy/Diffusion-Policy
sh eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${gpu_id} 
# As example: sh eval.sh Assembly 50 600 0
```

## 2. 3D Diffusion Policy

The DP3 code can be found in `policy/3D-Diffusion-Policy`.

```bash
python script/pkl2zarr_dp3.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_dp3.py assembly 100 
```

**If you want to use the point cloud data generated by CordViP and train it with DP3**

```bash
python script/pkl2zarr_dp3_ourpc.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_dp3_ourpc.py assembly 100 
```

Then, move to `policy/3D-Diffusion-Policy` first, and run the following code to train DP3:

```bash
bash train.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${num_epochs} ${checkpoint_every} ${horizon} ${n_obs_steps} ${n_action_steps}
# As example: bash train.sh Articulated_manip 50 42 2 20000 1000 12 4 8
```

Run the following code to evaluate DP3 for a specific task:

```bash
cd policy/3D-Diffusion-Policy
sh eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id} 
# As example: sh eval.sh Assembly 50 8000 42 0 
```

**If you want to use the point cloud data generated by CordViP and train it with DP3**

```bash
cd policy/3D-Diffusion-Policy
sh eval_our_pc.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id} 
# As example: sh eval_our_pc.sh Assembly 50 8000 42 0 
```

## 3. BCRNN

The BCRNN code can be found in `policy/BCRNN`.

```bash
python script/pkl2zarr_dp.py ${task_name} ${expert_data_num} 
# As example: python script/pkl2zarr_dp.py assembly 100 
```

Then, move to `policy/BCRNN` first, and run the following code to train BCRNN:

```bash
sh train_bc_rnn.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${epochs} ${checkpoint_every} ${horizon} ${n_obs_steps} ${n_action_steps} ${crop_h} ${crop_w}
# As example: sh train_bc_rnn.sh assembly 50 42 2 3000 100 10 1 1 480 640
```

Run the following code to evaluate BCRNN for a specific task:

```bash
cd policy/BCRNN
sh eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${gpu_id} 
# As example: sh eval.sh Assembly 50 600 0
```

## 4. BCRNN 3D

The BCRNN 3D code can be found in `policy/BCRNN_3D`.

```bash
python script/pkl2zarr_dp3.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_dp3.py assembly 100 
```

Then, move to `policy/BCRNN_3D` first, and run the following code to train BCRNN 3D:

```bash
sh train_bc_rnn_pc.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${epochs} ${checkpoint_every} ${horizon} ${n_obs_steps} ${n_action_steps}
# As example: sh train_bc_rnn_pc.sh flip_bottle 50 42 2 3000 100 10 1 1
```

Run the following code to evaluate BCRNN 3D for a specific task:

```bash
cd policy/BCRNN_3D
sh eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${gpu_id} 
# As example: sh eval.sh Assembly 50 8000 0
```

## 5. ACT

The ACT code can be found in `policy/act`.

```bash
python script/pkl2zarr_dp.py ${task_name} ${expert_data_num} 
# As example: python script/pkl2zarr_dp.py assembly 100 
```

Then, move to `policy/act` first, and run the following code to train ACT:

```bash
# step1
define task config in aloha_script/constants.py
As example: 'Articulated_manip':{
        'dataset_dir': '/data/Articulated_manip_50.zarr',
        'num_episodes': 50,
        'episode_len': 150,
        'camera_names': ['cam0']
    }
# step2
sh train.sh
```

Run the following code to evaluate ACT for a specific task:

```bash
python eval_policy_act.py --ckpt_dir ${ckpt_dir} --policy_class ${policy_class} --task_name ${task_name} --seed ${seed} --num_epochs ${num_epochs}
# As example: python eval_policy_act.py --ckpt_dir ../policy/act/checkpoints/assembly --policy_class ACT --task_name assembly --seed 0 --num_epochs 2000
```

## 6. ACT 3D

The ACT 3D code can be found in `policy/act+3d`.

```bash
python script/pkl2zarr_dp3.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_dp3.py assembly 100 
```

Then, move to `policy/act+3d` first, and run the following code to train ACT 3D:

```bash
# step1
define task config in aloha_script/constants.py
As example: 'Articulated_manip':{
        'dataset_dir': '/data/Articulated_manip_50.zarr',
        'num_episodes': 50,
        'episode_len': 150,
        'camera_names': ['cam0']
    }
# step2
sh train.sh
```

Run the following code to evaluate ACT 3D for a specific task:

```bash
python eval_policy_act_3d.py --ckpt_dir ${ckpt_dir} --policy_class ${policy_class} --task_name ${task_name} --seed ${seed} --num_epochs ${num_epochs}
# As example: python eval_policy_act_3d.py --ckpt_dir ../policy/act+3d/checkpoints/assembly --policy_class ACT --task_name assembly --seed 0 --num_epochs 2000
```

## 7. State-base Diffusion Policy

The state-base dp code can be found in `policy/statebase_Diffusion-Policy`.

```bash
python script/pkl2zarr_statebase.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_statebase.py flip_cup 50
```

Then, move to `policy/statebase_Diffusion-Policy` first, and run the following code to train state-base dp:

```bash
bash train.sh Assembly ${task_name} ${expert_data_num} ${seed} ${gpu_id} ${epochs} ${checkpoint_every} ${horizon} ${n_obs_steps} ${n_action_steps} ${crop_h} ${crop_w}
# As example: bash train.sh Assembly 50 42 3 2000 100 12 4 8 480 640
```

Run the following code to evaluate state-base dp for a specific task:

```bash
cd policy/statebase_Diffusion-Policy
sh eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${gpu_id} 
# As example: sh eval.sh Assembly 50 600 0
```

## 6. State-base CNNMLP

The state-base cnnmlp code can be found in `policy/statebase_CNNMLP`.

```bash
python script/pkl2zarr_statebase.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_statebase.py flip_cup 50
```

Then, move to `policy/statebase_CNNMLP` first, and run the following code to train state-base cnnmlp:

```bash
# step1
define task config in aloha_script/constants.py
As example: 'Articulated_manip':{
        'dataset_dir': '/data/Articulated_manip_50.zarr',
        'num_episodes': 50,
        'episode_len': 150,
        'camera_names': ['cam0']
    }
# step2
sh train.sh
```

Run the following code to evaluate state-base cnnmlp for a specific task:

```bash
python eval_policy_mlp.py --ckpt_dir ${ckpt_dir} --policy_class ${policy_class} --task_name ${task_name} --seed ${seed} --num_epochs ${num_epochs}
# As example: python eval_policy_mlp.py --ckpt ../policy/statebase_CNNMLP/checkpoints/assembly --policy_class CNNMLP --task_name flip_cap --seed 0 --num_epoch 900
```

## 🙏 Acknowledgement
Our code is built upon [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy), [DRO-Grasp](https://github.com/zhenyuwei2003/DRO-Grasp), [FoundationPose](https://github.com/NVlabs/FoundationPose). We thank all these authors for their nicely open sourced code and their great contributions to the community.


## 📜️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 BibTeX 

```bibtex
@article{fu2025cordvip,
  title={CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World},
  author={Fu, Yankai and Feng, Qiuxuan and Chen, Ning and Zhou, Zichen and Liu, Mengzhen and Wu, Mingdong and Chen, Tianxing and Rong, Shanyu and Liu, Jiaming and Dong, Hao and others},
  journal={arXiv preprint arXiv:2502.08449},
  year={2025}
}
```



