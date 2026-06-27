#!/bin/bash

PROJECT_ROOT="/nfs/lschnaitl/projects/svraster"
DATA_ROOT="/nfs/lschnaitl/projects/nerf_synthetic"

TARGET_DIRS=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for dir_name in "${TARGET_DIRS[@]}"; do
    SOURCE_PATH="${DATA_ROOT}/${dir_name}/"
    MODEL_PATH="output/nerf_synthetic/${dir_name}"

    echo "------------------------------------------------"
    echo "Starting training for: $dir_name"
    echo "Source Path: $SOURCE_PATH"
    echo "Model Output: $MODEL_PATH"

    if [ -d "$SOURCE_PATH" ]; then
        python train.py \
            --eval \
            --cfg_files $PROJECT_ROOT/cfg/synthetic_nerf.yaml \
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
