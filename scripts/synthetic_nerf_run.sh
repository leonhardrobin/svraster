# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

DATA_ROOT=/nfs/lschnaitl/projects/nerf_synthetic

lanuch_exp() {
    local scene_name="$1"
    shift
    local output_dir="$1"
    shift
    local exp_args="$*"

    python train.py --cfg_files cfg/synthetic_nerf.yaml --source_path $DATA_ROOT/$scene_name --model_path $output_dir/$scene_name --bound_mode camera_median $exp_args
    python render.py $output_dir/$scene_name --skip_train --eval_fps
    python render.py $output_dir/$scene_name --skip_train
    python eval.py $output_dir/$scene_name/
    python render_fly_through.py $output_dir/$scene_name/
    rm -r $output_dir/$scene_name/checkpoints/
}


for scene in chair drums ficus hotdog lego materials mic ship
do
   echo "============ start " $scene " ============"
   lanuch_exp $scene $1 "${@:2}"
   echo "============ end " $scene " ============"
done
