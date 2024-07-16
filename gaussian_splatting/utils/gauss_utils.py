import torch
import numpy as np

def depth_to_3d(depth_map, intrinsics):
    h, w = depth_map.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x = (x - cx) / fx
    y = (y - cy) / fy

    z = depth_map

    x = x * z
    y = y * z

    points_3d = torch.stack([x, y, z], dim=-1)
    return points_3d
