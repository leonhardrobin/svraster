# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os
import math
import time
import numpy as np
from tqdm import tqdm
import trimesh
import torch
import torch.nn.functional as F

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import svraster_cuda
from src.config import cfg, update_config
from src.utils import octree_utils
from src.utils import activation_utils
from src.utils.marching_cubes_utils import torch_marching_cubes_grid
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.fuser_utils import Fuser
import cv2

MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).cuda()
    seg_model.eval()
    
    seg_mean = torch.tensor(image_processor.image_mean, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
    seg_std = torch.tensor(image_processor.image_std, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
except Exception as e:
    print(f"Error loading Segformer: {e}")
    print("Ensure you have transformers installed: pip install transformers")
    exit(1)

def run_segformer(image_tensor):
    """
    Runs Segformer on a single image tensor (C, H, W) normalized [0,1].
    Returns integer class mask (H, W).
    """
    H, W = image_tensor.shape[-2:]
    
    target_dim = 640
    scale = target_dim / max(H, W)
    tH, tW = int(round(H * scale / 32) * 32), int(round(W * scale / 32) * 32)

    img_resized = F.interpolate(image_tensor.unsqueeze(0), size=(tH, tW), mode='bilinear', align_corners=False)
    
    pixel_values = (img_resized - seg_mean) / seg_std
    
    with torch.no_grad():
        outputs = seg_model(pixel_values)
        logits = outputs.logits # [1, 150, H/4, W/4]
        
    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    
    pred_seg = logits.argmax(dim=1).squeeze(0) # [H, W]
    return pred_seg

def filter_grid_by_semantics(args, voxel_model, grid_pts_xyz, data_pack, target_id):
    print(f"Filtering SDF for Semantic ID: {target_id}...")
    print(f"  > Threshold: {args.sem_conf*100}% consensus required")
    print(f"  > Erosion:   {args.sem_erode} pixels")
    
    semantic_votes = torch.zeros(grid_pts_xyz.shape[0], dtype=torch.int32, device="cuda")
    total_valid_views = torch.zeros(grid_pts_xyz.shape[0], dtype=torch.int32, device="cuda")
    
    cam_lst = data_pack.get_train_cameras()
    
    for cam in tqdm(cam_lst, desc="Semantic Projection"):
        render_pkg = voxel_model.render(cam, output_depth=True) 
        frame_depth = render_pkg['raw_depth'][[0]] 
        H, W = frame_depth.shape[-2:] 

        img_tensor = cam.image.cuda() 
        if img_tensor.ndim == 4: img_tensor = img_tensor.squeeze(0)
        
        frame_semantic = run_segformer(img_tensor) # [H_orig, W_orig]

        if args.sem_erode > 0:
            mask_np = frame_semantic.cpu().numpy().astype(np.uint8) 
            kernel = np.ones((args.sem_erode, args.sem_erode), np.uint8)
            
            binary_mask = (mask_np == target_id).astype(np.uint8)
            eroded_binary = cv2.erode(binary_mask, kernel, iterations=1)
            
            diff = binary_mask - eroded_binary
            mask_np[diff == 1] = 0 # clear the edge pixels
            
            frame_semantic = torch.from_numpy(mask_np).cuda().long()

        if frame_semantic.shape[-2:] != (H, W):
            frame_semantic = frame_semantic.float().unsqueeze(0).unsqueeze(0)
            frame_semantic = F.interpolate(frame_semantic, size=(H, W), mode='nearest')
            frame_semantic = frame_semantic.squeeze(0).squeeze(0).long()

        pts_uv = cam.project(grid_pts_xyz)

        inside_mask = (pts_uv.abs() <= 1).all(-1)
        valid_indices = torch.where(inside_mask)[0]
        if len(valid_indices) == 0: continue

        valid_uv = pts_uv[valid_indices]
        valid_pts = grid_pts_xyz[valid_indices]

        pts_frame_depth = F.grid_sample(
            frame_depth.view(1, 1, H, W), valid_uv.view(1, 1, -1, 2),
            mode='nearest', align_corners=False).flatten()

        pts_dist_to_cam = ((valid_pts - cam.position) @ cam.lookat)
        depth_margin = args.bandwidth_vox * 2.0 
        depth_mask = (pts_frame_depth - pts_dist_to_cam).abs() < depth_margin
        
        valid_indices = valid_indices[depth_mask]
        valid_uv = valid_uv[depth_mask]
        if len(valid_indices) == 0: continue
            
        pts_semantic_val = F.grid_sample(
            frame_semantic.float().view(1, 1, H, W), valid_uv.view(1, 1, -1, 2),
            mode='nearest', align_corners=False).flatten().long()

        match_mask = (pts_semantic_val == target_id)
        semantic_votes[valid_indices[match_mask]] += 1
        total_valid_views[valid_indices] += 1

    has_views = total_valid_views > 0
    ratio = semantic_votes.float() / (total_valid_views.float() + 1e-8)
    
    final_mask = has_views & (ratio >= args.sem_conf) 
    
    print(f"Retained {final_mask.sum()} / {len(final_mask)} voxels for label {target_id}")
    return final_mask

def tsdf_fusion(cam_lst, depth_lst, alpha_lst, grid_pts_xyz, trunc_dist, crop_border, alpha_thres):
    assert len(cam_lst) == len(depth_lst)
    fuser = Fuser(
        xyz=grid_pts_xyz, bandwidth=trunc_dist, use_trunc=True, fuse_tsdf=True,
        feat_dim=0, alpha_thres=alpha_thres, crop_border=crop_border,
        normal_weight=False, depth_weight=False, border_weight=False, use_half=False)

    for cam, frame_depth, frame_alpha in zip(tqdm(cam_lst, desc="TSDF Fusion"), depth_lst, alpha_lst):
        frame_depth = frame_depth.cuda()
        frame_alpha = frame_alpha.cuda()
        fuser.integrate(cam, frame_depth, alpha=frame_alpha)

    tsdf = fuser.tsdf.squeeze(1).contiguous()
    return tsdf

def extract_mesh_progressive(args, data_pack, voxel_model, init_lv, final_lv, crop_bbox):
    cam_lst = data_pack.get_train_cameras()
    depth_lst = []
    alpha_lst = []
    for cam in tqdm(cam_lst, desc="Render training views"):
        render_pkg = voxel_model.render(cam, output_depth=True, output_T=True)
        frame_depth = render_pkg['raw_depth'][[0]] if args.use_mean else render_pkg['raw_depth'][[2]]
        frame_alpha = 1 - render_pkg['raw_T']
        if args.save_gpu:
            frame_depth = frame_depth.cpu()
            frame_alpha = frame_alpha.cpu()
        depth_lst.append(frame_depth)
        alpha_lst.append(frame_alpha)

    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")

    vol = SparseVoxelModel(sh_degree=0)
    vol.model_init(bounding=torch.stack([inside_min, inside_max]), outside_level=0, init_n_level=init_lv)

    for lv in range(init_lv, final_lv+1):
        now_voxel_size = vol.vox_size[0].item()
        bandwidth = args.bandwidth_vox * now_voxel_size
        print(f"Running lv={lv:2d}: #voxels={vol.num_voxels:9d}; band={bandwidth}")

        grid_tsdf = tsdf_fusion(
            cam_lst, depth_lst, alpha_lst, vol.grid_pts_xyz, bandwidth, args.crop_border, args.alpha_thres)

        if lv < final_lv:
            vox_tsdf = grid_tsdf[vol.vox_key]
            thickness = min(2 / args.bandwidth_vox, 0.99)
            prune_mask = vox_tsdf.isnan().any(-1) | (vox_tsdf.amax(1) < -thickness) | (vox_tsdf.amin(1) > thickness)
            vol.pruning(prune_mask)
            vol.subdividing(torch.ones([vol.num_voxels], dtype=torch.bool))

    if args.semantic_id is not None:
        semantic_mask = filter_grid_by_semantics(args, voxel_model, vol.grid_pts_xyz, data_pack, args.semantic_id)
        grid_tsdf[~semantic_mask] = 100.0 

    verts, faces = torch_marching_cubes_grid(grid_pts_val=grid_tsdf, grid_pts_xyz=vol.grid_pts_xyz, vox_key=vol.vox_key, iso=0)
    return trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())

def extract_mesh(args, data_pack, voxel_model, final_lv, crop_bbox, iso=0):
    cam_lst = data_pack.get_train_cameras()
    depth_lst = []
    alpha_lst = []
    for cam in tqdm(cam_lst, desc="Render training views"):
        render_pkg = voxel_model.render(cam, output_depth=True, output_T=True)
        frame_depth = render_pkg['raw_depth'][[0]] if args.use_mean else render_pkg['raw_depth'][[2]]
        frame_alpha = 1 - render_pkg['raw_T']
        if args.save_gpu:
            frame_depth = frame_depth.cpu()
            frame_alpha = frame_alpha.cpu()
        depth_lst.append(frame_depth)
        alpha_lst.append(frame_alpha)

    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")

    target_lv = voxel_model.outside_level + final_lv
    octpath, octlevel = octree_utils.clamp_level(voxel_model.octpath, voxel_model.octlevel, target_lv)

    vol = SparseVoxelModel(sh_degree=0)
    vol.octpath_init(voxel_model.scene_center, voxel_model.scene_extent, octpath, octlevel)

    gridpts_outside = ((vol.grid_pts_xyz < inside_min) | (vol.grid_pts_xyz > inside_max)).any(-1)
    corners_outside = gridpts_outside[vol.vox_key]
    prune_mask = corners_outside.all(-1)
    vol.pruning(prune_mask)

    bandwidth = args.bandwidth_vox * vol.vox_size.min().item()
    print(f"Running adaptive: #voxels={vol.num_voxels:9d} / band={bandwidth}")
    
    grid_tsdf = tsdf_fusion(cam_lst, depth_lst, alpha_lst, vol.grid_pts_xyz, bandwidth, args.crop_border, args.alpha_thres)

    if args.semantic_id is not None:
        semantic_mask = filter_grid_by_semantics(args, voxel_model, vol.grid_pts_xyz, data_pack, args.semantic_id)
        grid_tsdf[~semantic_mask] = 100.0

    verts, faces = torch_marching_cubes_grid(grid_pts_val=grid_tsdf, grid_pts_xyz=vol.grid_pts_xyz, vox_key=vol.vox_key, iso=iso)
    return trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())

def colorize_pts(args, pts, data_pack, voxel_model):
    cloest_color = torch.full([len(pts), 3], 0.5, dtype=torch.float32, device="cuda")
    cloest_dist = torch.full([len(pts)], np.inf, dtype=torch.float32, device="cuda")
    cam_lst = data_pack.get_train_cameras()
    for cam in tqdm(cam_lst):
        render_pkg = voxel_model.render(cam, color_mode="sh0", output_depth=True, output_T=True)
        frame_color = render_pkg['color']
        frame_depth = render_pkg['raw_depth'][[0]] if args.use_mean else render_pkg['raw_depth'][[2]]
        frame_alpha = 1 - render_pkg['raw_T']
        H, W = frame_depth.shape[-2:]
        pts_uv = cam.project(pts)
        filter_idx = torch.where((pts_uv.abs() <= 1).all(-1))[0]
        valid_pts_idx = filter_idx
        valid_pts = pts[filter_idx]
        pts_uv = pts_uv[filter_idx]
        pts_frame_alpha = F.grid_sample(frame_alpha.view(1,1,H,W), pts_uv.view(1,1,-1,2), mode='bilinear', align_corners=False).flatten()
        filter_idx = torch.where(pts_frame_alpha > args.alpha_thres)[0]
        valid_pts_idx = valid_pts_idx[filter_idx]
        valid_pts = valid_pts[filter_idx]
        pts_uv = pts_uv[filter_idx]
        pts_frame_depth = F.grid_sample(frame_depth.view(1,1,H,W), pts_uv.view(1,1,-1,2), mode='bilinear', align_corners=False).flatten()
        pts_depth = ((valid_pts - cam.position) @ cam.lookat)
        pts_dist = (pts_frame_depth - pts_depth).abs()
        filter_idx = torch.where(pts_dist < cloest_dist[valid_pts_idx])[0]
        valid_pts_idx = valid_pts_idx[filter_idx]
        pts_uv = pts_uv[filter_idx]
        pts_dist = pts_dist[filter_idx]
        pts_color = F.grid_sample(frame_color[None], pts_uv.view(1,1,-1,2), mode='bilinear', align_corners=False).squeeze().T
        cloest_dist[valid_pts_idx] = pts_dist
        cloest_color[valid_pts_idx] = pts_color
    return cloest_color

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sparse voxels raster extract mesh.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_gpu", action='store_true')
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_n_samp_per_vox", default=None, type=str)
    parser.add_argument("--mesh_fname", default=None, type=str)
    parser.add_argument("--bbox_path", default=None)
    parser.add_argument("--bbox_scale", default=1.0, type=float)
    parser.add_argument("--direct", action='store_true')
    parser.add_argument("--progressive", action='store_true')
    parser.add_argument("--init_lv", default=8, type=int)
    parser.add_argument("--final_lv", default=10, type=int)
    parser.add_argument("--bandwidth_vox", default=5.0, type=float)
    parser.add_argument("--crop_border", default=0.01, type=float)
    parser.add_argument("--alpha_thres", default=0.5, type=float)
    parser.add_argument("--semantic_id", default=None, type=int)
    parser.add_argument("--sem_conf", default=0.6, type=float, help="Confidence threshold (0.0 to 1.0). Higher = stricter.")
    parser.add_argument("--sem_erode", default=0, type=int, help="Pixels to erode from 2D mask before projection. Good for separating floor.")
    parser.add_argument("--use_mean", action='store_true')
    parser.add_argument("--use_vert_color", action='store_true')
    parser.add_argument("--use_clean", action='store_true')
    parser.add_argument("--use_remesh", action='store_true')
    parser.add_argument("--remesh_len", default=-1, type=float)

    args = parser.parse_args()
    print("Rendering " + args.model_path)
    update_config(os.path.join(args.model_path, 'config.yaml'))

    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
        camera_params_only=False,
    )

    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    voxel_model.load_iteration(args.model_path, args.iteration)
    voxel_model.freeze_vox_geo()

    if args.overwrite_ss is not None: voxel_model.ss = args.overwrite_ss
    if args.overwrite_n_samp_per_vox is not None: voxel_model.n_samp_per_vox = args.overwrite_n_samp_per_vox

    outdir = os.path.join(args.model_path, "mesh", f"iter{voxel_model.loaded_iter:06d}" if voxel_model.loaded_iter > 0 else "latest")
    os.makedirs(outdir, exist_ok=True)

    if args.bbox_path: crop_bbox = np.loadtxt(args.bbox_path)
    else: crop_bbox = None

    fname = 'mesh'
    if args.semantic_id is not None: fname += f'_sem{args.semantic_id}'

    with torch.no_grad():
        if args.progressive:
            if args.semantic_id is not None: fname += f'_lv{args.init_lv}-{args.final_lv}'
            mesh = extract_mesh_progressive(args, data_pack, voxel_model, args.init_lv, args.final_lv, crop_bbox)
        else:
            mesh = extract_mesh(args, data_pack, voxel_model, args.final_lv, crop_bbox)
            fname += f'_lv{args.final_lv}_adaptive'

    if args.use_mean: fname += '_dmean'

    if args.use_clean:
        fname += '_clean'
        try:
            labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
            cc, cc_cnt = np.unique(labels, return_counts=True)
            cc_maxid = cc[cc_cnt.argmax()]
            mesh.update_faces(labels==cc_maxid)
            vmask = np.zeros([len(mesh.vertices)], dtype=bool)
            vmask[mesh.faces] = 1
            mesh.update_vertices(vmask)
        except: print("Failed to segment largest cc")

    if args.use_remesh:
        from gpytoolbox import remesh_botsch
        avg_edge_len = mesh.edges_unique_length.mean()
        target_edge_len = args.remesh_len if args.remesh_len >= 0 else min(avg_edge_len, voxel_model.inside_extent.item() / 1024)
        print(f"Remeshing to {target_edge_len}")
        try:
            v, f = remesh_botsch(mesh.vertices, mesh.faces, i=5, h=target_edge_len)
            mesh = trimesh.Trimesh(vertices=v, faces=f)
        except: print(f"Remesh failed.")

    if args.use_vert_color:
        print("Colorizing vertices")
        with torch.no_grad():
            pts = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
            verts_color = colorize_pts(args, pts, data_pack, voxel_model)
            verts_color = verts_color.cpu().numpy()
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=verts_color)

    if data_pack.to_world_matrix is not None:
        mesh = mesh.apply_transform(data_pack.to_world_matrix)

    if args.mesh_fname is not None: fname = args.mesh_fname
    outpath = os.path.join(outdir, f'{fname}.ply')
    mesh.export(outpath)
    print('Saved to', outpath)