#!/bin/bash

demonstration_id=$1  
source ~/anaconda3/etc/profile.d/conda.sh

demonstration_folder_path="./expert_data/pick_chicken/demonstration_${demonstration_id}"
save_foundation_folder="./FoundationPose/expert_data/Assembly/demonstration_${demonstration_id}"
output_cloud_dir="./DRO-Grasp/expert_data/Assembly/demonstration_${demonstration_id}"
output_mesh_dir="./FoundationPose/object_mesh"

obj_path="${output_mesh_dir}/0/mesh.obj" 
texture_image_path="${output_mesh_dir}/0/texture.png" 
output_dir="${output_mesh_dir}/1" 


# echo "Running TripoSR to generate texture and mesh"
# conda activate tripo
# cd ./tripoSr/TripoSR
# python run.py examples/objects/color_cup.jpeg --output-dir "$output_mesh_dir" --bake-texture --texture-resolution 2048
# sleep 1

# echo "Running Blender to apply texture using get_texture.py"
# conda activate tripo
# blender --background --python ./FoundationPose/get_texture.py -- --obj_path "$obj_path"  --texture_image_path "$texture_image_path" --output_dir "$output_dir"
# sleep 1

echo "Running get_foundation.py in $CONDA_ENV_FOUNDATIONPOSE environment"
conda activate foundationpose
python ./FoundationPose/get_foundation.py  --data_folder "$demonstration_folder_path" --save_folder "$save_foundation_folder" --mesh_file "$output_mesh_dir/1/mesh_with_texture.obj"
sleep 1

echo "Running generate_pc.py in $CONDA_ENV_DRO environment"
conda activate foundationpose
python ./DRO-Grasp/data_utils/generate_pc.py  --output_cloud_dir "$output_cloud_dir" --demonstration_folder_path "$demonstration_folder_path" 
sleep 1

echo "Running pc_visualization.py in $CONDA_ENV_DRO environment"
conda activate foundationpose
python ./DRO-Grasp/data_utils/pc_visualization.py --pt_folder_path "$output_cloud_dir" --pose_folder_path "$save_foundation_folder" --save_folder "$demonstration_folder_path" 

echo "All scripts finished."
