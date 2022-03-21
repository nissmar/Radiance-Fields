from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from utilities import *
from VoxelGrid import *
import os
import argparse


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Compute a voxel grid from images. All lists must have the same size')
parser.add_argument('-dataset', default="../nerf_synthetic/chair", help='dataset folder')
parser.add_argument('-initial_size', type = int, default=128, help='initial voxel grid size')
parser.add_argument('-batch_size', type = int, default=5000, help='batch_size')
parser.add_argument('-epoch',  type=int, default=8, help='epochs')
parser.add_argument('-npoints', type=int, default=200, help='samples along rays')
parser.add_argument('-lr', type=int, default=10000, help='learning rate')
parser.add_argument('-load', default=None, help='model to load')
args = parser.parse_args()

focal, all_c2w, all_gt = get_data(args.dataset)

red = 8
target_ims, rays = reduce_data(all_c2w, all_gt, focal, red)
target_ims_carve = [ (e[:,:,0]==1)*(e[:,:,1]==1)*(e[:,:,2]==1) for e in target_ims]
target_ims_carve = [ 1-torch.nn.MaxPool2d(3, 1, 1)(torch.tensor((1-e)*1.0)[None,:])[0].detach().numpy() for e in target_ims_carve]

red = 2
disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, red)

print("Making Dataloader")
D_carve = RayDataset(target_ims_carve, rays, device)
D = RayDataset(target_ims, rays, device)
train_loader_carve = torch.utils.data.DataLoader(D_carve, batch_size=args.batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(D, batch_size=args.batch_size, shuffle=True)

VG = VoxelGridCarve(args.initial_size, 1.4)
if not(args.load is None):
    VG.load(args.load+'.obj')
losses=[]

print("Carving model")
carve(VG, train_loader_carve, 1000)
new_im = VG.render_large_image_from_rays(disp_rays[0],(1000,1.2))
plt.imsave('screenshots/render.png', new_im)
color(VG, train_loader, 1000)
new_im = VG.render_large_image_from_rays(disp_rays[0],(1000,1.2))
plt.imsave('screenshots/render2.png', new_im)

def train(epoch, optimizer, N_points):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        optimizer.zero_grad()
        pix_estims = VG.render_rays(rays, (N_points))
        loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0]
        loss.backward()
        losses.append(loss.item())
        mask = (VG.opacities==0)
        optimizer.step()
        VG.clamp()
        with torch.no_grad():
            VG.opacities[mask]=0
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

for epoch in tqdm(range(args.epoch)):
    optimizer = torch.optim.SGD(
        [VG.colors, VG.opacities], 
        lr=args.lr/2**epoch
    )
    new_im = VG.render_large_image_from_rays(disp_rays[0],(500,1.2))
    plt.imsave('screenshots/render.png', new_im)
    losses += train(epoch, optimizer, args.npoints)
    plt.clf()
    plt.plot(rolling_average(np.array(losses), 10))
    plt.savefig('screenshots/training.png')
    VG.save('out_carve.obj') 



    