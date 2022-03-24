from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from utilities import *
from VoxelGrid import *
import os

import argparse
import time as tm

t_start = tm.time()
parser = argparse.ArgumentParser(description='Compute a voxel grid from images. All lists must have the same size')
parser.add_argument('-model', default="drums", help='dataset folder')
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device='cuda' if torch.cuda.is_available() else 'cpu'


model=args.model
dataset= "../nerf_synthetic/" + model
focal, all_c2w, all_gt = get_data(dataset, "train")

back_focal, back_c2w, back_gt = get_data(dataset, "train", True)

red = 4
target_ims_carve, rays_carve = reduce_data(back_c2w, back_gt.squeeze(), back_focal, red)
red = 4
target_ims, rays = reduce_data(all_c2w, all_gt, focal, red)
red = 4
disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, red)
losses=[]

D_carve = RayDataset(target_ims_carve, rays_carve, device)
train_loader_carve = torch.utils.data.DataLoader(D_carve, batch_size=5000, shuffle=False)

D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=5000, shuffle=True)

    
VG = VoxelGridSphericalCarve(128, 1.4, 40, 9)

carve(VG, train_loader_carve, 900)

color_sph_base(VG, train_loader, 900)

losses+=color_sph_sgd(VG, train_loader, 900, 0.9)
VG.smooth_colors()

for lr in tqdm([0.9, 0.1, 0.1]):
    losses+=color_sph_sgd(VG, train_loader, 900, lr)
    plt.clf()
    plt.plot((np.array(losses)))
    plt.savefig('screenshots/training.png')


#VG.save(model+'_carve.obj')
# PSNR

test_focal, test_c2w, test_gt = get_data("../nerf_synthetic/" + model, "test")

red = 8
disp_ims_test, disp_rays_test = reduce_data(test_c2w, test_gt,test_focal, red)
disp_im_w = disp_ims_test[0].shape[0]

print(model, compute_psnr(VG, disp_rays_test, disp_ims_test, 900), tm.time()-t_start)