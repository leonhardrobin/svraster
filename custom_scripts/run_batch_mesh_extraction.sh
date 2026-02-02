#!/bin/bash

# ================= CONFIGURATION =================
# Set the base absolute path to your project root
PROJECT_ROOT="/nfs/lschnaitl/projects/svraster"

# Define the list of directories you want to train on
TARGET_DIRS=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# =================================================

# Loop through each directory
for dir_name in "${TARGET_DIRS[@]}"; do
    # Construct the specific paths based on the directory name
    MODEL_PATH="output/nerf_synthetic/${dir_name}"

    echo "------------------------------------------------"
    echo "Starting mesh extraction for: $dir_name"
    echo "Model Output: $MODEL_PATH"

    # Check if the processed data actually exists before trying to train
    if [ -d "$MODEL_PATH" ]; then
        python extract_mesh.py $MODEL_PATH \
            --use_vert_color \
            --mesh_fname mesh_svraster_${dir_name}_v3

            
        echo "Finished mesh extraction for $dir_name"
    else
        echo "Error: Model path '$MODEL_PATH' does not exist."
        echo "Has this dataset been processed successfully yet? Skipping..."
    fi
done

echo "------------------------------------------------"
echo "All mesh extraction runs complete."