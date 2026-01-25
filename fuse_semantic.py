import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import trimesh

# Add project root to path
sys.path.insert(0, os.path.abspath('./'))

from src.config import cfg, update_config
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.octree_utils import level_2_vox_size
from src.utils.fuser_utils import Fuser
import svraster_cuda

def get_args():
    parser = argparse.ArgumentParser(description='Fuse Semantic Fields using Segformer')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model output directory')
    parser.add_argument('--source_path', type=str, default=None, help='Path to the dataset (overrides config)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    # Segformer settings
    parser.add_argument('--segformer_model', type=str, default="nvidia/segformer-b5-finetuned-ade-640-640", help='HuggingFace model string')
    parser.add_argument('--logit_sharpening', type=float, default=1.0, help='Sharpening factor for logits')
    
    # Fusion settings
    parser.add_argument('--bandwidth_mult', type=float, default=50.0, help='Bandwidth multiplier for voxel size')
    
    return parser.parse_known_args()[0]

def get_ade20k_palette():
    return [
        [120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],[120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],
        [224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],[255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],
        [0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],[255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
        [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],[7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7],
        [255,122,8],[0,255,20],[255,8,41],[255,5,153],[6,51,255],[235,12,255],[160,150,20],[0,163,255],[140,140,140],[250,10,15],
        [20,255,0],[31,255,0],[255,31,0],[255,224,0],[153,255,0],[0,0,255],[255,71,0],[0,235,255],[0,173,255],[31,0,255],
        [11,200,200],[255,82,0],[0,255,245],[0,61,255],[0,255,112],[0,255,133],[255,0,0],[255,163,0],[255,102,0],[194,255,0],
        [0,143,255],[51,255,0],[0,82,255],[0,255,41],[0,255,173],[10,0,255],[173,255,0],[0,255,153],[255,92,0],[255,0,255],
        [255,0,245],[255,0,102],[255,173,0],[255,0,20],[255,184,184],[0,31,255],[0,255,61],[0,71,255],[255,0,204],[0,255,194],
        [0,255,82],[0,10,255],[0,112,255],[51,0,255],[0,194,255],[0,122,255],[0,255,163],[255,153,0],[0,255,10],[255,112,0],
        [143,255,0],[82,0,255],[163,255,0],[255,235,0],[8,184,170],[133,0,255],[0,255,92],[184,0,255],[255,0,31],[0,184,255],
        [0,214,255],[255,0,112],[92,255,0],[0,224,255],[112,224,255],[70,184,160],[163,0,255],[153,0,255],[71,255,0],[255,0,163],
        [255,204,0],[255,0,143],[0,255,235],[133,255,0],[255,0,235],[245,0,255],[255,0,122],[255,245,0],[10,190,212],[214,255,0],
        [0,204,255],[20,0,255],[255,255,0],[0,153,255],[0,41,255],[0,255,204],[41,0,255],[41,255,0],[173,0,255],[0,245,255],
        [71,0,255],[122,0,255],[0,255,184],[0,92,255],[184,255,0],[0,133,255],[255,214,0],[25,194,194],[102,255,0],[92,0,255]
    ]

def load_segformer(model_name, device='cuda'):
    print(f"Loading Segformer: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    image_mean = torch.tensor(image_processor.image_mean, dtype=torch.float32, device=device)
    image_std = torch.tensor(image_processor.image_std, dtype=torch.float32, device=device)
    
    # Get Label Mapping
    id2label = model.config.id2label
    
    # Create Palette
    num_classes = len(id2label)
    if num_classes == 150:
        palette = get_ade20k_palette()
    else:
        # Fallback random palette
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(num_classes, 3)).tolist()
        
    return model, image_mean, image_std, id2label, palette

def preprocess_image(x, image_mean, image_std):
    H, W = x.shape[-2:]
    if H > W:
        tH = 640 * H / W
        tW = 640
    else:
        tW = 640 * W / H
        tH = 640
    tH = round(tH / 32) * 32
    tW = round(tW / 32) * 32
    
    x = torch.nn.functional.interpolate(
        x[None],
        size=(int(tH), int(tW)),
        mode='bilinear',
        align_corners=False,
    )
    x = (x - image_mean.view(1, 3, 1, 1)) / image_std.view(1, 3, 1, 1)
    return x

def main():
    args = get_args()
    
    if args.config is None:
        potential_cfg = os.path.join(args.model_path, 'config.yaml')
        if not os.path.exists(potential_cfg):
             print("Warning: No config file specified.")
    else:
        update_config(args.config)

    print(f"Output Root: {args.model_path}")
    
    # 1. Load Sparse Voxel Model
    voxel_model = SparseVoxelModel()
    voxel_model.load_iteration(args.model_path)
    
    # 2. Load Dataset
    data_path = args.source_path if args.source_path else cfg.source_path
    print(f"Loading Data from: {data_path}")
    data_pack = DataPack(data_path)
    tr_cams = data_pack.get_train_cameras()
    
    # 3. Initialize Segformer & Labels
    seg_model, img_mean, img_std, id2label, palette = load_segformer(args.segformer_model)
    num_classes = len(id2label)
    print(f"Number of classes: {num_classes}")

    # 4. Save Label Metadata
    label_info = {}
    for i, name in id2label.items():
        # Ensure palette has enough colors
        color = palette[i] if i < len(palette) else [0,0,0]
        label_info[i] = {"name": name, "color": color}
    
    json_path = os.path.join(args.model_path, 'labels.json')
    with open(json_path, 'w') as f:
        json.dump(label_info, f, indent=4)
    print(f"Saved label definitions to {json_path}")

    # 5. Setup Fuser
    finest_vox_size = level_2_vox_size(voxel_model.scene_extent, voxel_model.octlevel.max()).item()
    
    feat_volume = Fuser(
        xyz=voxel_model.vox_center,
        bandwidth=args.bandwidth_mult * finest_vox_size,
        use_trunc=False,
        fuse_tsdf=False,
        feat_dim=num_classes,
        crop_border=0.,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=True
    )

    # 6. Fusion Loop
    print("Starting Semantic Fusion...")
    with torch.no_grad():
        for cam in tqdm(tr_cams, desc="Fusing Images"):
            img_tensor = cam.image.cuda()
            pixel_values = preprocess_image(img_tensor, img_mean, img_std)
            
            probs = seg_model(pixel_values).logits
            probs = probs.mul(args.logit_sharpening).softmax(dim=1).squeeze(0)
            
            render_pkg = voxel_model.render(cam, output_depth=True)
            depth = render_pkg['depth'][2]
            
            feat_volume.integrate(cam=cam, feat=probs, depth=depth)

    # 7. Finalize and Save Features
    print("Finalizing features...")
    feature = feat_volume.feature.nan_to_num_()
    
    save_path = os.path.join(args.model_path, 'semantic_features.pt')
    print(f"Saving logits to {save_path}")
    torch.save(feature, save_path)
    
    # 8. Generate Colored Point Cloud for Visualization
    print("Generating colored point cloud...")
    
    # Get most likely class for each voxel
    voxel_classes = feature.argmax(dim=1).cpu().numpy()
    
    # Map classes to colors
    voxel_colors = np.array([label_info[c]["color"] for c in voxel_classes], dtype=np.uint8)
    voxel_coords = voxel_model.vox_center.cpu().numpy()
    
    # Create Point Cloud
    pcd = trimesh.PointCloud(vertices=voxel_coords, colors=voxel_colors)
    ply_path = os.path.join(args.model_path, 'semantic_points.ply')
    pcd.export(ply_path)
    
    print(f"Saved visualization to {ply_path}")
    print("Done.")

if __name__ == "__main__":
    main()