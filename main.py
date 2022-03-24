from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from utilities import *
from VoxelGrid import *
import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Compute a voxel grid from images. All lists must have the same size')
parser.add_argument('-dataset', default="../nerf_synthetic/materials", help='dataset folder')
parser.add_argument('-initial_size', type = int, default=16, help='initial voxel grid size')
parser.add_argument('-batch_size', type = int, default=5000, help='batch_size')
parser.add_argument('-epochs', nargs='+', type=int, default=[2, 4, 8, 16], help='list of epochs')
parser.add_argument('-npoints', nargs='+', type=int, default=[50, 100, 200, 200], help='list of samples along rays')
parser.add_argument('-lrs', nargs='+', type=int, default=[1000, 1000, 500, 500], help='list of learning rates')
args = parser.parse_args()

if len(args.npoints) != len(args.lrs) or len(args.npoints) != len(args.epochs):
    raise argparse.ArgumentTypeError('All lists must share the same size')

focal, all_c2w, all_gt = get_data(args.dataset)

red = 8
target_ims, rays = reduce_data(all_c2w, all_gt, focal, red)

red = 2
disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, red)


D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=args.batch_size, shuffle=True)

VG = VoxelGrid(args.initial_size, 1.4)

losses = []

def train(epoch, optimizer):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        optimizer.zero_grad()
        pix_estims = VG.render_rays(rays, (N_points))
        #loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0] + 0.0001*VG.total_variation()
        loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0]
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        VG.clamp()
        if batch_idx%10==0:
            print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        100.0 * batch_idx / len(train_loader),
                        loss.data.item(),
                    ),
                    flush = True
                )
    return losses

for epochs, N_points, lr in zip(tqdm(args.epochs), args.npoints, args.lrs):
    optimizer = torch.optim.SGD(
                [VG.colors, VG.opacities], 
                lr=lr
            )

    for epoch in range(epochs):
        new_im = VG.render_large_image_from_rays(disp_rays[0],(500,1.2))
        plt.imsave('screenshots/render.png', new_im)
        losses += train(epoch, optimizer)
        plt.clf()
        plt.plot(np.log(rolling_average(np.array(losses))))
        plt.savefig('screenshots/training.png')
        VG.save(args.dataset[18:-1]+'.obj')
    VG.subdivide()
    


    