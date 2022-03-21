from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import pickle 
from ply import write_ply

from utilities import *
from VoxelGrid import *
import os

from PIL import ImageFont
from PIL import ImageDraw 
import argparse
import time as tm

t_start = tm.time()
parser = argparse.ArgumentParser(description='Compute a voxel grid from images. All lists must have the same size')
parser.add_argument('-model', default="drums", help='model folder')
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device='cuda' if torch.cuda.is_available() else 'cpu'

init_size = 128
N_points = 900
model=args.model


dataset= "../nerf_synthetic/" + model
focal, all_c2w, all_gt = get_data(dataset, "train")

back_focal, back_c2w, back_gt = get_data(dataset, "train", True)

red = 4
target_ims_carve, rays_carve = reduce_data(back_c2w, back_gt.squeeze(), back_focal, red)
red = 4
target_ims, rays = reduce_data(all_c2w, all_gt, focal, red)

VG = VoxelGridCarve(init_size, 1.4, 40)

def carve(grid, loader, N_points):
    for batch_idx, (rays, pixels) in enumerate(tqdm(loader)):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        mask = (pixels==1)
        grid.carve((rays[0][mask],rays[1][mask]) , N_points)

def color(grid, loader, N_points):
    for batch_idx, (rays, pixels) in enumerate(tqdm(loader)):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        mask = (pixels==1).all(1)
        mask = torch.logical_not(mask)
        grid.color((rays[0][mask],rays[1][mask]), pixels[mask], N_points)
    with torch.no_grad():
        mask = VG.colors_sum>0
        VG.colors[mask] = VG.colors[mask]/(VG.colors_sum[mask, None])

print('Making carve dataloader')
D_carve = RayDataset(target_ims_carve, rays_carve, device)
train_loader = torch.utils.data.DataLoader(D_carve, batch_size=5000, shuffle=True)
carve(VG, train_loader, N_points)
plt.imsave('screenshots/render.png', VG.render_large_image_from_rays(rays[30],(N_points,1.2)))
plt.show()

#VG.subdivide()

print('Making color dataloader')
D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=5000, shuffle=True)

color(VG, train_loader, N_points)
plt.imsave('screenshots/render2.png', VG.render_large_image_from_rays(rays[30],(N_points,1.2)))
plt.show()

VG.save_pointcloud(0, model+'.ply')
VG.save(model+'_carve.obj')

print("Computed in ", tm.time()-t_start, " seconds")