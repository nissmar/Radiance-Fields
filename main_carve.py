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
import argparse
import time as tm

t_start = tm.time()
parser = argparse.ArgumentParser(description='Compute a voxel grid from images. All lists must have the same size')
parser.add_argument('-model', default="drums", help='dataset folder')
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

D_carve = RayDataset(target_ims_carve, rays_carve, device)
train_loader_carve = torch.utils.data.DataLoader(D_carve, batch_size=5000, shuffle=False)

D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=5000, shuffle=False)

VG = VoxelGridCarve(init_size, 1.4, 30)


#print('Carving model')
carve(VG, train_loader_carve, N_points)
plt.imsave('screenshots/render.png', VG.render_large_image_from_rays(rays[30],(N_points,1.2)))
plt.show()

#VG.subdivide()

#print('Coloring model')
color(VG, train_loader, N_points)
plt.imsave('screenshots/render2.png', VG.render_large_image_from_rays(rays[30],(N_points,1.2)))
plt.show()

#VG.save(model+'_carve.obj')

print("Computed in ", tm.time()-t_start, " seconds")

test_focal, test_c2w, test_gt = get_data("../nerf_synthetic/" + model, "test")
red = 4
disp_ims_test, disp_rays_test = reduce_data(test_c2w, test_gt,test_focal, red)
disp_im_w = disp_ims_test[0].shape[0]

print(model, compute_psnr(VG, disp_rays_test, disp_ims_test, N_points))