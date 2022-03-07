from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import pickle 
from sklearn.cluster import KMeans

from utilities import *

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

device='cuda'


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
        
        x = np.linspace(-bound_w,bound_w,size)
        y = np.linspace(-bound_w,bound_w,size)
        z = np.linspace(-bound_w,bound_w,size)
        self.meshgrid = np.meshgrid(x,y,z)

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
        q = torch.zeros_like(p)
        q[..., 0] = (p[...,0]+self.bound_w)/(2*self.bound_w)*(self.size-1)
        q[..., 1] = (p[...,1]+self.bound_w)/(2*self.bound_w)*(self.size-1)
        q[..., 2] = (p[...,2]+self.bound_w)/(2*self.bound_w)*(self.size-1)
        return q

    def flatten_3d_indices(self, inds_3d):
        ''' input: Nx3 array of indices
            out: N array of flatten indices'''
        return inds_3d[...,0] + self.size*inds_3d[...,1] + self.size*self.size*inds_3d[...,2]

    def render_rays(self, p):
        with torch.no_grad():
            # extract valid indices
            inds_3d = torch.round(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long())
            mesh_coords[torch.logical_not(in_bounds)] = 0

        colors = self.colors[mesh_coords]
        opacities = self.opacities[mesh_coords]*in_bounds.float()
        cumsum_opacities = torch.cumsum(opacities, 1)
        
        transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
        return (colors*transp_term[..., None]).sum(1)

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