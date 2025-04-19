#!/bin/bash
demo_dir=$1  
demonstration_id=$2  
mesh_path=$3
source ~/anaconda3/etc/profile.d/conda.sh

demonstration_folder_path="${demo_dir}/demonstration_${demonstration_id}"
save_foundation_folder="./temp/pose/demonstration_${demonstration_id}"
output_cloud_dir="./temp/pc/demonstration_${demonstration_id}"

echo "Running get_foundation.py in posetrack environment"
conda activate foundationpose
python ./FoundationPose/get_foundation.py  --data_folder "$demonstration_folder_path" --save_folder "$save_foundation_folder" --mesh_file "$mesh_path"
sleep 1

echo "Running generate_pc.py in posetrack environment"
conda activate foundationpose
python ./data_utils/generate_pc.py  --output_cloud_dir "$output_cloud_dir" --demonstration_folder_path "$demonstration_folder_path" 
sleep 1

echo "Running pc_get.py in posetrack environment"
conda activate foundationpose
python ./data_utils/pc_get.py --mesh_path "$mesh_path" --output_path "./temp/ply/pc_1024.ply" --num_points 1024


echo "Running pc_visualization.py in posetrack environment"
conda activate foundationpose
python ./data_utils/pc_visualization.py --pt_folder_path "$output_cloud_dir" --pose_folder_path "$save_foundation_folder" --ply_file_path "./temp/ply/pc_1024.ply" --save_folder "$demonstration_folder_path" 

rm -rf "./temp"
echo "All scripts finished."
