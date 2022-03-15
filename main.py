from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from utilities import *
from VoxelGrid import *
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = "materials"

focal, all_c2w, all_gt = get_data("../nerf_synthetic/"+dataset)

red = 8
target_ims, rays = reduce_data(all_c2w, all_gt, focal, red)
im_w = target_ims[0].shape[0]

red = 2
disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, red)
disp_im_w = disp_ims[0].shape[0]


D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=5000, shuffle=True)

VG = VoxelGrid(8, 1.4)

losses=[]

def train(epoch, optimizer):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        optimizer.zero_grad()
        pix_estims = VG.render_rays(rays, (N_points))
        loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0] +0.0001*VG.total_variation()
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
                    )
                )
    return losses

epochss = [2,4,8, 8]
N_pointss = [50, 100, 200, 200]
lrs = [1000, 1000, 500, 500]


for epochs, N_points, lr in zip(epochss,N_pointss, lrs):
    VG.subdivide()
    optimizer = torch.optim.SGD(
                [VG.colors, VG.opacities], 
                lr=lr
            )


    for epoch in tqdm(range(epochs)):
        new_im = VG.render_large_image_from_rays(disp_rays[0],(500,1.2))
        plt.imsave('screenshots/render.png', new_im)
        losses += train(epoch, optimizer)
        plt.plot(np.log(rolling_average(np.array(losses))))
        plt.savefig('screenshots/training.png')
        VG.save(dataset+'.obj')
    


    