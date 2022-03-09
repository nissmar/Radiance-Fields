from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import pickle 
from sklearn.cluster import KMeans

from utilities import *
from VoxelGrid import VoxelGrid
import os

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter


#VARIABLES
base_model = '64_best.obj'
subdivide=True
#OR
grid_size = 64 
bound_w = 1.2

learning_rate = 30
N_points = 400


epochs = 40
train_reduce = 8
test_reduce = 4
image_ind = 2

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device='cuda'
torch.cuda.empty_cache()

# load data
focal, all_c2w, all_gt = get_data("../lego")

# process data
target_ims, rays = reduce_data(all_c2w, all_gt, focal, train_reduce)
disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, test_reduce)
print('All data loaded. Making dataloader..')
D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=5000, shuffle=True)


VG = VoxelGrid(grid_size, bound_w)
VG.load(base_model)
if subdivide:   
    VG.subdivide()

def train(epoch):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        pix_estims = VG.render_rays(rays, (N_points))
        
        loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0] + 0.001*VG.total_variation()
        #loss = ((pix_estims-pixels)**2).sum()/rays.shape[0]
        loss.backward()
        losses.append(loss.item())
        VG.update_grads(learning_rate)
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

losses=[]
for epoch in tqdm(range(epochs)):
    VG.save(str(grid_size)+'a_'+str(epoch)+'.obj')
    new_im = VG.render_image_from_rays(disp_rays[image_ind],(500,1.2))
    plt.imshow(new_im)
    plt.show()
    plt.imsave('screenshots/a'+str(epoch)+'.png', new_im)
    losses += train(epoch)
print(losses)

VG.save(str(grid_size)+'b_'+str(epoch+1)+'.obj')
plt.clf()
plt.plot(losses)
plt.savefig('screenshots/'+str(grid_size)+'_training.png')    