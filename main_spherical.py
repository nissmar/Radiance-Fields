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
base_model = 'sph64_best.obj'
subdivide=True
#OR
grid_size = 128 
bound_w = 1.2

VG = VoxelGridSpherical(grid_size, bound_w)
VG.load(base_model)
if subdivide:   
    VG.subdivide()


learning_rate = 1000
N_points = 200


epochs = 50
train_reduce = 4
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


 

optimizer = torch.optim.SGD(
            [VG.colors, VG.opacities], 
            lr=learning_rate
        )

def train(epoch):
    losses=[]
    for batch_idx, (rays, pixels) in enumerate(train_loader):
        rays, pixels = (rays[0].to(device),rays[1].to(device)), pixels.to(device)
        optimizer.zero_grad()

        pix_estims = VG.render_rays(rays, (N_points))
        
        #loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0] + 0.001*VG.total_variation()
        loss = ((pix_estims-pixels)**2).sum()/rays[0].shape[0]
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
for epoch in tqdm(range(epochs)):
      
    if epoch%5==0 and epoch!=0:
        VG.save('sph'+str(grid_size)+'a_'+str(epoch)+'.obj')
    
    new_im = VG.render_image_from_rays(disp_rays[image_ind],(500,bound_w))
    plt.imshow(new_im)
    plt.show()
    plt.imsave('screenshots/a'+str(epoch)+'.png', new_im)
    losses.append(train(epoch))
    plt.clf()
    plt.plot(losses)
    plt.savefig('screenshots/'+str(grid_size)+'_training.png') 
print(losses)

VG.save('sph'+str(grid_size)+'final_'+str(epoch+1)+'.obj')
 