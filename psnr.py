from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utilities import *
from VoxelGrid import *
import os

import argparse


device='cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Compute PSNR')
parser.add_argument('-dataset', default="../nerf_synthetic/chair", help='dataset folder')
parser.add_argument('-grid', default="chair", help='dataset folder')
parser.add_argument('-reduce', type = int, default=8, help='image reduction')
args = parser.parse_args()

print(args.grid)
VG = VoxelGrid()
VG.load(args.grid + ".obj")
#VG.subdivide()

test_focal, test_c2w, test_gt = get_data(args.dataset, "test")
red = args.reduce
disp_ims_test, disp_rays_test = reduce_data(test_c2w, test_gt,test_focal, red)
disp_im_w = disp_ims_test[0].shape[0]

print(compute_psnr(VG, disp_rays_test, disp_ims_test, 1000))