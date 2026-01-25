#!/bin/bash

# ================= CONFIGURATION =================
# Set the base absolute path to your project root
PROJECT_ROOT="/nfs/lschnaitl/projects/svraster"
DATA_ROOT="/nfs/lschnaitl/projects/nerf_synthetic"

# Define the list of directories you want to train on
TARGET_DIRS=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# =================================================

# Loop through each directory
for dir_name in "${TARGET_DIRS[@]}"; do
    # Construct the specific paths based on the directory name
    SOURCE_PATH="${DATA_ROOT}/${dir_name}/"
    MODEL_PATH="output/nerf_synthetic/${dir_name}"

    echo "------------------------------------------------"
    echo "Starting training for: $dir_name"
    echo "Source Path: $SOURCE_PATH"
    echo "Model Output: $MODEL_PATH"

    # Check if the processed data actually exists before trying to train
    if [ -d "$SOURCE_PATH" ]; then
        python train.py \
            --eval \
            --cfg_files cfg/synthetic_nerf.yaml \
            --source_path $SOURCE_PATH \
            --model_path $MODEL_PATH \
            --lambda_normal_dmean 0.001 \
            --lambda_normal_dmed 0.001 \

            
        echo "Finished training for $dir_name"

        python render_fly_through.py $MODEL_PATH
    else
        echo "Error: Source path '$SOURCE_PATH' does not exist."
        echo "Has this dataset been processed successfully yet? Skipping..."
    fi
done

echo "------------------------------------------------"
echo "All training runs complete."
