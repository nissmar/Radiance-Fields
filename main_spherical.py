from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import pickle 
from sklearn.cluster import KMeans

from utilities import *
from VoxelGrid import VoxelGrid, VoxelGridSpherical
import os

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter


#VARIABLES
dataset = "../lego"
base_model = '16sph17.obj'
subdivide=False
#OR
grid_size = 16 
bound_w = 1.2

VG = VoxelGridSpherical(grid_size, bound_w)
VG.load(base_model)
# if subdivide:   
#     VG.subdivide()


learning_rate = 1000
N_points = 200


epochs = 16
train_reduce = 8
test_reduce = 4
image_ind = 2

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device='cuda'
torch.cuda.empty_cache()

# load data
focal, all_c2w, all_gt = get_data(dataset)

# process data
target_ims, rays = reduce_data(all_c2w, all_gt, focal, train_reduce)
disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, test_reduce)
print('All data loaded. Making dataloader..')
D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=5000, shuffle=True)


def train(epoch, optimizer):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        optimizer.zero_grad()

        pix_estims = VG.render_rays(rays, (N_points))
        
        loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0] +0.0001*VG.total_variation()
        #loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0]
        loss.backward()
        losses.append(loss.data.item())
        #VG.update_grads(learning_rate)
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
    return np.array(losses).mean()
losses=[]


optimizer = torch.optim.SGD(
            [VG.colors, VG.opacities], 
            lr=learning_rate)
for epoch in tqdm(range(18, 40)):
    #   if epoch%5==0:
    new_im = VG.render_large_image_from_rays(disp_rays[image_ind],(1000,bound_w))
    plt.imsave('screenshots/a'+str(epoch)+'.png', np.clip(new_im, 0,1))
    losses.append(train(epoch, optimizer))
    plt.clf()
    plt.plot(losses)
    plt.savefig('screenshots/'+str(grid_size)+'_training.png') 
    VG.save(str(grid_size)+'sph'+str(epoch)+'.obj')
print(losses)