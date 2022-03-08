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
#base_model = '64_best.obj'

#OR
grid_size = 64 
bound_w = 1.2

learning_rate = 1000


epochs = 10
train_reduce = 8
test_reduce = 4
N_points = 350
image_ind = 2




#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device='cuda'
torch.cuda.empty_cache()

# load data
focal, all_c2w, all_gt = get_data("../lego")


# load data
target_ims, rays = reduce_data(all_c2w, all_gt, focal, train_reduce, N_points)
im_w = target_ims[0].shape[0]

disp_ims, disp_rays = reduce_data(all_c2w, all_gt, focal, test_reduce, N_points)
disp_im_w = disp_ims[0].shape[0]
print('All data loaded. Making dataloader..')

D = RayDataset(target_ims, rays, device)
train_loader = torch.utils.data.DataLoader(D, batch_size=3000, shuffle=True)


VG = VoxelGrid(grid_size, bound_w)
#VG.load(base_model)

def train(epoch):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        pix_estims = VG.render_rays(rays)
        
        loss = ((pix_estims-pixels)**2).sum()/rays.shape[0] + 0.001*VG.total_variation()
        #loss = ((pix_estims-pixels)**2).sum()/rays.shape[0]
        loss.backward()
        losses.append(loss.item)
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
    losses += train(epoch)
    if epoch%5==0:
        VG.save(str(grid_size)+'b_'+str(epoch)+'.obj')
    tensor_rays_img = torch.tensor(disp_rays[image_ind], dtype=torch.float32).to(device).permute((0,1,3,2)).view((disp_im_w*disp_im_w,N_points,3))

    new_im = VG.render_rays(tensor_rays_img).view((disp_im_w,disp_im_w,3)).cpu().detach().numpy()

    plt.imsave('screenshots/b'+str(epoch)+'.png', new_im)
   
plt.plot(losses)
plt.savefig('screenshots/'+str(grid_size)+'_training.png')    