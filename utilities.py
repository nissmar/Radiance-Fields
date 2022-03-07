import os
from tqdm import tqdm
import imageio
import numpy as np
import json
from torch.utils.data import Dataset
import torch



# FROM PLENOXELS

def get_data(root="../nerf_example_data/nerf_synthetic/lego", stage="train"):
    all_c2w = []
    all_gt = []

    data_path = os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    print('LOAD DATA', data_path)
    j = json.load(open(data_json, 'r'))

    for frame in tqdm(j['frames']):
        fpath = os.path.join(data_path, os.path.basename(
            frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        im_gt = im_gt[..., :3] * im_gt[..., 3:]
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5,
                       np.arange(H, dtype=np.float32) + 0.5, indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# CUSTOM

def get_cameras_centers(rays_or_dir):
    len_rays = len(rays_or_dir)
    centers = np.zeros((len_rays, 3))
    for i in range(len_rays):
        # all cameras share same center
        centers[i, :] = rays_or_dir[i][0][0, 0]
    return centers

def reduce_data(all_c2w, all_gt, focal, red_fac, N_points):
    H,W = all_gt[0].shape[:2]
    red_ims = [gt[::red_fac,::red_fac,:] for gt in all_gt]
    red_rays_or_dir = [get_rays_np(H,W, focal, c2w) for c2w in all_c2w]
    rays = [e[0][::red_fac,::red_fac,:, None] + np.arange(N_points)/10*e[1][::red_fac, ::red_fac,:, None] for e in red_rays_or_dir]
    
    return red_ims, rays

class RayDataset(Dataset):
    def __init__(self, target_ims, rays, device):
        im_w = target_ims[0].shape[0]

        self.tensor_rays = []
        self.tensor_target_pixels = []
        
        for image_ind in tqdm(range(im_w)):
            for i in range(im_w):
                for j in range(im_w):
                    self.tensor_rays.append(torch.tensor(rays[image_ind][i,j], dtype=torch.float32).to(device).T)
                    self.tensor_target_pixels.append(torch.tensor(target_ims[image_ind][i,j], dtype=torch.float32).to(device))

    def __getitem__(self, index):
        return self.tensor_rays[index], self.tensor_target_pixels[index]
    def __len__(self):
        return len(self.tensor_rays)




