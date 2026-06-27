#!/bin/bash

PROJECT_ROOT="/nfs/lschnaitl/projects/svraster"

TARGET_DIRS=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for dir_name in "${TARGET_DIRS[@]}"; do
    MODEL_PATH="output/nerf_synthetic/${dir_name}"

    echo "------------------------------------------------"
    echo "Starting mesh extraction for: $dir_name"
    echo "Model Output: $MODEL_PATH"

    if [ -d "$MODEL_PATH" ]; then
        python extract_mesh.py $MODEL_PATH \
            --use_vert_color \
            --mesh_fname mesh_svraster_${dir_name}_v6 \
            --final_lv 9

            
        echo "Finished mesh extraction for $dir_name"
    else
        echo "Error: Model path '$MODEL_PATH' does not exist."
        echo "Has this dataset been processed successfully yet? Skipping..."
    fi
done

echo "------------------------------------------------"
echo "All mesh extraction runs complete."