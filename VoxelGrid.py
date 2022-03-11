from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import pickle 
from sklearn.cluster import KMeans
from ply import write_ply
from scipy.special import sph_harm


from utilities import *

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

device='cuda' if torch.cuda.is_available() else 'cpu'


def laplacian3d(arr):
    dx = (arr[:-1,:-1,:-1,...] - arr[1:,:-1,:-1,...])**2
    dy = (arr[:-1,:-1,:-1,...] - arr[:-1,1:,:-1,...])**2
    dz = (arr[:-1,:-1,:-1,...] - arr[:-1,:-1,1:,...])**2
    return torch.sqrt(dx+dy+dz + 10**-5)


class VoxelGrid():
    def __init__(self, size=128, bound_w=1):
        self.size = size
        self.bound_w = bound_w
        self.colors =  Variable(torch.rand((size*size*size,3)).to(device), requires_grad=True)
        self.opacities =  Variable(torch.rand((size*size*size)).to(device), requires_grad=True)
    def in_bounds_indices(self, p):
        ''' input: Nx3 array
            output: index of valid arrays'''
        in_x = (p[...,0]>=0)*(p[...,0]<self.size)
        in_y = (p[...,1]>=0)*(p[...,1]<self.size)
        in_z = (p[...,2]>=0)*(p[...,2]<self.size)
        return in_x*in_y*in_z
    
    def descartes_to_indices(self, p):
        ''' input: Nx3 array, 3D points
            out: Nx3 array, 3D indices in [0, size['''        
        return (p+self.bound_w)/(2*self.bound_w)*self.size

    def flatten_3d_indices(self, inds_3d):
        ''' input: Nx3 array of indices
            out: N array of flatten indices'''
        return inds_3d[...,0] + self.size*inds_3d[...,1] + self.size*self.size*inds_3d[...,2]

    def update_grads(self, lr):
        self.colors.data -= lr * self.colors.grad.data        
        self.opacities.data -= lr * self.opacities.grad.data
       
        self.opacities.data = torch.clamp(self.opacities.data, 0)
        self.colors.data = torch.clamp(self.colors.data, 0,1)

        self.colors.grad.data.zero_()
        self.opacities.grad.data.zero_()
        
    def copy(self):
        CopyGrid = VoxelGrid(self.size,self.bound_w)
        CopyGrid.colors = torch.clone(self.colors)
        CopyGrid.opacities = torch.clone(self.opacities)
        return CopyGrid
    
    def save(self, filename="VoxelGrid.obj"):
        filehandler = open("saved_grids/" + filename, 'wb') 
        pickle.dump(self, filehandler)
        
    def load(self, filename="VoxelGrid.obj"):
        filehandler = open("saved_grids/" + filename, 'rb') 
        self.__dict__.update(pickle.load(filehandler).__dict__)
    
    def treshold_opacity(self, r):
        voxel_opacity = (1-torch.exp(-self.opacities))
        with torch.no_grad():
            self.opacities[voxel_opacity<r]=0
            self.colors[voxel_opacity<r]=0
            
    def make_palette(self, n=50):
        kmeans = KMeans(n_clusters=n).fit(self.colors.cpu().detach().numpy())
        colors = np.ones((n,4))
        colors[:,:3]=kmeans.cluster_centers_
        return kmeans.labels_, np.minimum(255, np.maximum(0, 255*colors)).astype(int)
        

    def save_magica_vox(self, tresh=0, filename="test.vox"):
        out_voxels = np.zeros((self.size, self.size, self.size), dtype='B')
        palette = []
        #TODO: add unfold_array
        in_palette, colors = self.make_palette(50)
        for i in tqdm(range(self.size)):
            for j in range(self.size):
                for k in range(self.size):
                    ind = self.flatten_3d_indices(np.array([k,j,i]))
                    voxel_opacity = (1-torch.exp(-self.opacities[ind,None]))
                    if voxel_opacity>tresh:
                        out_voxels[i,j,k] = in_palette[ind]
        vox = Vox.from_dense(out_voxels)
        vox.palette = colors
        VoxWriter("saved_grids/" + filename, vox).write()       
        
    def unfold_arrays(self):
        unfolded_colors = self.colors.view((self.size, self.size, self.size, 3)).transpose(2,0)
        unfolded_opacities =  self.opacities.view((self.size, self.size, self.size)).transpose(2,0)
        return unfolded_colors, unfolded_opacities
    
    def total_variation(self):
        unfolded_colors, unfolded_opacities = self.unfold_arrays()
        lap_color = laplacian3d(unfolded_colors).sum()
        lap_opacities = laplacian3d(unfolded_opacities).sum()
        return (lap_color/3)/(2*self.size**3)

    def subdivide(self):
        with torch.no_grad():

            old_ind = regular_3d_indexes(self.size)

            self.size = 2 * self.size

            new_colors =  Variable(torch.rand((self.size*self.size*self.size,3)).to(device), requires_grad=True)
            new_opacities =  Variable(torch.rand((self.size*self.size*self.size)).to(device), requires_grad=True)

            offsets = [np.array([0,0,0]), np.array([1,0,0]),np.array([0,1,0]), np.array([0,0,1]),
                      np.array([0,1,1]), np.array([1,1,0]), np.array([1,0,1]), np.array([1,1,1])]
            
            for off in offsets:
                target_inds = self.flatten_3d_indices(2*old_ind+off)
                new_colors[target_inds,:] = self.colors[:]
                new_opacities[target_inds] = self.opacities[:]
            self.colors = new_colors
            self.opacities = new_opacities
            
    
    def render_rays(self, ordir_tuple, N_points, inv_depth=1.2):
        ori = ordir_tuple[0][:, None,:]
        
        # WARNING: Assuming constant distance
        distances = 10*torch.sqrt( (ordir_tuple[1]**2).sum(1, keepdim=True))/inv_depth/(N_points-1)
        scatter_points = torch.rand_like(distances)*distances + torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth

        p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])        
        with torch.no_grad():
            # extract valid indices
            inds_3d = torch.round(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long())
            mesh_coords[torch.logical_not(in_bounds)] = 0
        
        colors = self.colors[mesh_coords]
        opacities = self.opacities[mesh_coords]*in_bounds.float() # not_in bounds: 0 opacity
        
        opacities = opacities*distances
        cumsum_opacities = torch.cumsum(opacities, 1)
        
        transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
        return (colors*transp_term[..., None]).sum(1)
    
    def render_image_from_rays(self, im_rays, kwargs):
        disp_im_w = im_rays[0].shape[0]
        ori = torch.tensor(im_rays[0], dtype=torch.float32, device=device).view((disp_im_w*disp_im_w,3))
        direct = torch.tensor(im_rays[1], dtype=torch.float32, device=device).view((disp_im_w*disp_im_w,3))
        return self.render_rays((ori,direct),*kwargs).view((disp_im_w,disp_im_w,3)).cpu().detach().numpy()
    def render_large_image_from_rays(self, im_rays, kwargs, batch_size=1000):
        with torch.no_grad():
            disp_im_w = im_rays[0].shape[0]
            ori = torch.tensor(im_rays[0], dtype=torch.float32, device=device).view((disp_im_w*disp_im_w,3))
            direct = torch.tensor(im_rays[1], dtype=torch.float32, device=device).view((disp_im_w*disp_im_w,3))

            out_img = torch.zeros((disp_im_w*disp_im_w,3))
            ind=0

            for ori_it, direct_it in zip(ori.split(batch_size), direct.split(batch_size)):
                out_img[batch_size*ind:batch_size*(ind+1)] = self.render_rays((ori_it,direct_it),*kwargs).detach()
                ind+=1

            return out_img.view((disp_im_w,disp_im_w,3)).cpu().detach().numpy()

    def save_pointcloud(self, tresh=0, filename="test.ply"):
        valid_ind = (self.opacities!=0).nonzero()
        
        j = torch.div(valid_ind, self.size, rounding_mode='floor')
        k = torch.div(j, self.size, rounding_mode='floor')


        cloud = torch.cat((valid_ind%self.size, 
                           j%self.size, 
                           k%self.size),1)/self.size
        write_ply("saved_grids/" + filename,
                  (cloud.cpu().detach().numpy(),
                   self.colors[valid_ind.flatten()].cpu().detach().numpy(), 
                   self.opacities[valid_ind.flatten()].cpu().detach().numpy()
                  ), ['x', 'y', 'z', 'r', 'g','b', 'opacity'])



class VoxelGridSpherical(VoxelGrid):
    def clamp(self):
        with torch.no_grad():
            self.opacities[:] = torch.clamp(self.opacities, 0)
            self.colors[:] = torch.clamp(self.colors, 0,1)
    def copy(self):
        CopyGrid = VoxelGridSpherical(self.size,self.bound_w, self.num_harm)
        CopyGrid.colors = torch.clone(self.colors)
        CopyGrid.opacities = torch.clone(self.opacities)
        return CopyGrid
    
    def subdivide(self):
        with torch.no_grad():

            old_ind = regular_3d_indexes(self.size)
            self.size = 2 * self.size
            new_colors =  Variable(torch.rand((self.size*self.size*self.size,3, self.num_harm)).to(device), requires_grad=True)
            new_opacities =  Variable(torch.rand((self.size*self.size*self.size)).to(device), requires_grad=True)

            offsets = [np.array([0,0,0]), np.array([1,0,0]),np.array([0,1,0]), np.array([0,0,1]),
                      np.array([0,1,1]), np.array([1,1,0]), np.array([1,0,1]), np.array([1,1,1])]
            
            for off in offsets:
                target_inds = self.flatten_3d_indices(2*old_ind+off)
                new_colors[target_inds,:,:] = self.colors[:]
                new_opacities[target_inds] = self.opacities[:]
            self.colors = new_colors
            self.opacities = new_opacities
            
    def __init__(self, size=128, bound_w=1, num_harm=4):
        self.size = size
        self.bound_w = bound_w
        self.num_harm = 4
        self.colors =  Variable(torch.rand((size*size*size,3,num_harm)).to(device), requires_grad=True)
        self.opacities =  Variable(torch.rand((size*size*size)).to(device), requires_grad=True)


    def view_harmonics(self, point):
        r1 =  torch.sqrt((point**2).sum())
        r2 = torch.sqrt((point[:2]**2).sum())
        theta = torch.arcsin(r2/r1).cpu().numpy()
        phi = torch.arccos(point[0].abs()/r2).cpu().numpy()
        phi *= 1 if point[1]>0 else -1
        harmonics = np.array([sph_harm(j,i,theta, phi) 
                              for i in range(int(np.sqrt(self.num_harm))) 
                              for j in range(-i,i+1)])
        return torch.tensor(np.abs(harmonics), dtype=torch.float32, device=device)

    def render_rays(self, ordir_tuple, N_points, inv_depth=1.2):

        ori = ordir_tuple[0][:, None,:]

        # WARNING: Assuming constant distance
        distances = 10*torch.sqrt( (ordir_tuple[1]**2).sum(1, keepdim=True))/inv_depth/(N_points-1)
        scatter_points = torch.rand_like(distances)*distances + torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth
        p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])
        displacement = (2*self.bound_w)/self.size
        with torch.no_grad():
            # extract valid indices
            inds_3d = torch.floor(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long())
            mesh_coords[torch.logical_not(in_bounds)] = 0
            harmonics = self.view_harmonics(p[0,0])

            
        colors = torch.tensordot(self.colors[mesh_coords],harmonics, dims=([-1], [0]))
        
        opacities = self.opacities[mesh_coords]*in_bounds.float() # not_in bounds: 0 opacity

        opacities = opacities*distances
        cumsum_opacities = torch.cumsum(opacities, 1)

        transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
        return (colors*transp_term[..., None]).sum(1)