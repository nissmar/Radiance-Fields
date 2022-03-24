from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from utilities import *
from VoxelGrid import *
import os

import argparse
import time as tm

parser = argparse.ArgumentParser(description='Compute a movie from a model.')
parser.add_argument('-model', default="chair_carve", help='model')
parser.add_argument('-dataset', default="chair", help='dataset folder')
args = parser.parse_args()


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device='cuda' if torch.cuda.is_available() else 'cpu'


dataset= "../nerf_synthetic/" + args.dataset
focal, all_c2w, all_gt = get_data(dataset, "test")
cust_c2ws = create_rotation_matrices(1.5, -20, n=120)
red_fac=2
ordir_rays=[]
for c2w in cust_c2ws:
    ray_np = get_rays_np(800,800, focal, c2w)
    oris = ray_np[0][::red_fac,::red_fac]
    direct = ray_np[1][::red_fac,::red_fac] # direction. optimal fac:3
    ordir_rays.append((oris, direct))


VG=VoxelGridSphericalCarve(128, 1.4, 40, 9)
VG.load(args.model+'.obj')

imgs=[]
for image_ind in tqdm(range(len(cust_c2ws))):
    with torch.no_grad():
        new_im = VG.render_large_image_from_rays(ordir_rays[image_ind],(900, 1.2))
        plt.imshow(new_im)
        plt.show()
        imgs.append(np.uint8(255*new_im))

imageio.mimwrite('exports/movies_'+args.model+'.gif', imgs,  format='GIF', duration=0.04)