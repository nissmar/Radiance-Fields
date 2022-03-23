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
    def clamp(self):
        with torch.no_grad():
            self.opacities[:] = torch.clamp(self.opacities, 0)
            self.colors[:] = torch.clamp(self.colors, 0,1)

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
        return (lap_color/3+lap_opacities)/(2*self.size**3)

    def subdivide(self):
        with torch.no_grad():
            def old_flatten_3d(inds_3d):
                return inds_3d[...,0] + self.size//2*inds_3d[...,1] + self.size*self.size//4*inds_3d[...,2]

            old_ind = regular_3d_indexes(self.size)
            old_indm = regular_3d_indexes(self.size-1)

            self.size = 2 * self.size

            new_colors =  Variable(torch.zeros((self.size*self.size*self.size,3)).to(device), requires_grad=True)
            new_opacities =  Variable(torch.zeros((self.size*self.size*self.size)).to(device), requires_grad=True)
        
            target_inds = self.flatten_3d_indices(2*old_ind)
            new_colors[target_inds,:] = self.colors[:]
            new_opacities[target_inds] = self.opacities[:]

            # edges
            offsets = [np.array([1,0,0]),np.array([0,1,0]), np.array([0,0,1])]
            for off in offsets:
                c1_ind = old_flatten_3d(old_indm+off)
                c2_ind = old_flatten_3d(old_indm)
                
                target_inds = self.flatten_3d_indices(2*old_indm+off)
                new_colors[target_inds,:] = (self.colors[c1_ind,:]+self.colors[c2_ind,:])/2.0
                new_opacities[target_inds] = (self.opacities[c1_ind]+self.opacities[c2_ind])/2.0
                
            # Volume
            target_inds = self.flatten_3d_indices(2*old_indm+off[0]+off[1]+off[2])
            inds=[]
            inds.append(old_flatten_3d(old_indm+off[0]))
            inds.append(old_flatten_3d(old_indm+off[1]))
            inds.append(old_flatten_3d(old_indm+off[2]))
            inds.append(old_flatten_3d(old_indm))
            inds.append(old_flatten_3d(old_indm+off[0]+off[1]))
            inds.append(old_flatten_3d(old_indm+off[1]+off[2]))
            inds.append(old_flatten_3d(old_indm+off[2]+off[0]))
            inds.append(old_flatten_3d(old_indm+off[0]+off[1]+off[2]))
            new_colors[target_inds,:] = sum([self.colors[c_ind,:] for c_ind in inds])/8.0
            new_opacities[target_inds] = sum([self.opacities[c_ind] for c_ind in inds])/8.0

                
            offsets = [(np.array([1,0,0]),np.array([0,1,0]))
                       ,(np.array([0,1,0]),np.array([0,0,1]))
                       ,(np.array([0,0,1]),np.array([1,0,0]))]
            
            for (off1, off2) in offsets:
                c1_ind = old_flatten_3d(old_indm+off1)
                c2_ind = old_flatten_3d(old_indm+off2)
                c3_ind = old_flatten_3d(old_indm+off1+off2)
                c4_ind = old_flatten_3d(old_indm)
                
                target_inds = self.flatten_3d_indices(2*old_indm+off1+off2)
                new_colors[target_inds,:] = (self.colors[c1_ind,:]+self.colors[c2_ind,:]
                                            +self.colors[c3_ind,:]+self.colors[c4_ind,:])/4.0
                new_opacities[target_inds] = (self.opacities[c1_ind]+self.opacities[c2_ind]
                                             +self.opacities[c3_ind]+self.opacities[c4_ind])/4.0
                
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
        cumsum_opacities = torch.zeros_like(opacities, device=device)
        cumsum_opacities[:,1:] = torch.cumsum(opacities[:,:-1], 1)
        
        transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
        return (colors*transp_term[..., None]).sum(1) + torch.exp(-cumsum_opacities[:, -1])[..., None]
    
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

            out = out_img.view((disp_im_w,disp_im_w,3)).cpu().detach().numpy()
            return np.clip(out,0,1)

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
    def __init__(self, size=128, bound_w=1, num_harm=9):
        self.size = size
        self.bound_w = bound_w
        self.num_harm = num_harm
        self.colors =  Variable(torch.rand((size*size*size,3,num_harm)).to(device), requires_grad=True)
        self.opacities =  Variable(torch.rand((size*size*size)).to(device), requires_grad=True)

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
            

    def view_harmonics(self, points):
        norm = torch.sqrt((points**2).sum(1))
        points /= norm[:, None]
        r2 = torch.sqrt((points[:, :2]**2).sum(1))
        r2 = normalize01(r2, 0, 1) #avoid numerical errors
        phi = torch.arccos(r2)
        phi[points[:,2]<0] *= -1
        phi += np.pi/2
        r2[r2==0] = 10**-10 
        diam = points[:,0]/r2
        diam = normalize01(diam, -1, 1) #avoid numerical errors
        theta = torch.arccos(diam)
        theta[points[:,1]<0] *= -1  
        theta += np.pi
        theta = theta.cpu().numpy()
        phi = phi.cpu().numpy()
        harmonics = torch.zeros((points.shape[0], self.num_harm), device=device)
        
        ind=0
        for n in range(int(np.sqrt(self.num_harm))):
            for m in range(-n,n+1):
                Y = torch.tensor(sph_harm(m,n,theta, phi), device=device)
                if m < 0:
                    Y = np.sqrt(2) * Y.imag
                elif m > 0:
                    Y = np.sqrt(2) * Y.real
                harmonics[:, ind] = torch.abs(Y)
                ind+=1
        #harmonics /= torch.sqrt((harmonics**2).sum(1))[:, None]
        return harmonics

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
            
            harmonics = self.view_harmonics(ordir_tuple[1])
            
        colors = (harmonics[:, None, None, :]*self.colors[mesh_coords]).sum(-1)
        opacities = self.opacities[mesh_coords]*in_bounds.float() # not_in bounds: 0 opacity
        opacities = opacities*distances
        cumsum_opacities = torch.zeros_like(opacities, device=device)
        cumsum_opacities[:,1:] = torch.cumsum(opacities[:,:-1], 1)
        transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
        return (colors*transp_term[..., None]).sum(1) + torch.exp(-cumsum_opacities[:, -1])[..., None]
    
    def unfold_arrays(self):
        unfolded_colors = self.colors.view((self.size, self.size, self.size, 3, self.num_harm)).transpose(2,0)
        unfolded_opacities =  self.opacities.view((self.size, self.size, self.size)).transpose(2,0)
        return unfolded_colors, unfolded_opacities
    
    def total_variation(self):
        unfolded_colors, unfolded_opacities = self.unfold_arrays()
        lap_color = laplacian3d(unfolded_colors).sum()
        lap_opacities = laplacian3d(unfolded_opacities).sum() 
        return (lap_color/(3*self.num_harm)+lap_opacities)/(2*self.size**3)

class VoxelGridInterp(VoxelGrid):
    def in_bounds_indices(self, p):
        ''' input: Nx3 array
            output: index of valid arrays'''
        in_x = (p[...,0]>=0)*(p[...,0]<(self.size-1))
        in_y = (p[...,1]>=0)*(p[...,1]<(self.size-1))
        in_z = (p[...,2]>=0)*(p[...,2]<(self.size-1))
        return in_x*in_y*in_z
   
    def render_rays(self, ordir_tuple, N_points, inv_depth=1.2):
        with torch.no_grad():
            ori = ordir_tuple[0][:, None,:]
            # WARNING: Assuming constant distance
            distances = 10*torch.sqrt( (ordir_tuple[1]**2).sum(1, keepdim=True))/inv_depth/(N_points-1)
            scatter_points = torch.rand_like(distances)*distances + torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth
            p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])

            offsets = torch.tensor([[0,0,0], 
                               [1,0,0], 
                               [0,0,1],
                               [1,0,1],
                               [0,1,0],
                               [1,1,0],  
                               [0,1,1],
                               [1,1,1]], device=device)
            offsets[:, 1] *= self.size
            offsets[:, 2] *= self.size**2
            offsets = offsets.sum(1)

        
            #indicies
            p = self.descartes_to_indices(p)   
            p0 = torch.floor(p)
            in_bounds = self.in_bounds_indices(p)

            mesh_coords = self.flatten_3d_indices(p0.long())
            mesh_coords[torch.logical_not(in_bounds)] = 0

            diff = (p-p0)
            ind_offsets = []

            mcs = []
            for i in range(8):
                mcs.append( mesh_coords + offsets[i])                
        # color interp
        ciii = [self.colors[mcs[i]]
                *(1-diff[...,0, None])+
                self.colors[mcs[i+1]]
                *diff[...,0, None] for i in range(0,8,2)]
        cii = [ciii[i]*(1-diff[...,1, None]) + ciii[i+2]*diff[...,1, None] for i in [0,1]]
        colors = cii[0]*(1-diff[...,2, None])+cii[1]*diff[...,2, None]
        
        # opacities interp
        oiii = [self.opacities[mcs[i]]
               *(1-diff[...,0]) +
                self.opacities[mcs[i+1]]
                *diff[...,0] for i in range(0,8,2)]
        oii = [oiii[i]*(1-diff[...,1]) + oiii[i+2]*diff[...,1] for i in [0,1]]
        opacities = (oii[0]*(1-diff[...,2])+oii[1]*diff[...,2])*distances*in_bounds.float() 

        cumsum_opacities = torch.cumsum(opacities, 1)

        transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
        return (colors*transp_term[..., None]).sum(1) + torch.exp(-cumsum_opacities[:, -1])[..., None]
    

class VoxelGridCarve(VoxelGrid):
    def __init__(self, size=128, bound_w=1, init_op=3):
        super().__init__(size, bound_w)
        self.colors_sum = torch.zeros_like(self.opacities)
        with torch.no_grad():
            self.opacities[:] = init_op
            self.colors[:] = 0
    def subdivide(self):
        super().subdivide()
        self.colors_sum = torch.zeros_like(self.opacities)
    def smooth_colors(self):
        with torch.no_grad():
            new_ar = 6*self.colors.clone()
            for disp1 in [1, self.size, self.size**2]:
                new_ar[:-disp1] +=self.colors[disp1:]
                new_ar[disp1:] += self.colors[:-disp1]
            self.colors[:] = new_ar/12
            
    def smooth_opacities(self):
        with torch.no_grad():
            new_ar = 6*self.opacities.clone()
            for disp1 in [1, self.size, self.size**2]:
                new_ar[:-disp1] +=self.opacities[disp1:]
                new_ar[disp1:] += self.opacities[:-disp1]
            self.opacities[:] = new_ar/12

    def carve(self, ordir_tuple, N_points, inv_depth=1.2):
        with torch.no_grad():
            ori = ordir_tuple[0][:, None,:]

            # WARNING: Assuming constant distance
            distances = 8/(N_points-1)
            scatter_points = torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth
            p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])    

            # extract valid indices
            inds_3d = torch.floor(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long())
            mesh_coords[torch.logical_not(in_bounds)] = 0
            
            self.opacities[mesh_coords] = 0
            
    def color(self, ordir_tuple, pixels, N_points, inv_depth=1.2):
        with torch.no_grad():
            ori = ordir_tuple[0][:, None,:]

            # WARNING: Assuming constant distance
            distances = 8/(N_points-1)
            scatter_points = torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth
            p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])    

            # extract valid indices
            inds_3d = torch.floor(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long()).long()
            mesh_coords[torch.logical_not(in_bounds)] = 0
            opacities = self.opacities[mesh_coords]*in_bounds.float() # not_in bounds: 0 opacity
            opacities = opacities*distances
            cumsum_opacities = torch.zeros_like(opacities, device=device)
            cumsum_opacities[:,1:] = torch.cumsum(opacities[:,:-1], 1)

            transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
            
            self.colors[mesh_coords,:] += transp_term[..., None]*pixels[:, None, :]
            self.colors_sum[mesh_coords] += transp_term

class VoxelGridSphericalCarve(VoxelGridSpherical):
    def __init__(self, size=128, bound_w=1, init_op=3, num_harm=9):
        super().__init__(size, bound_w, num_harm)
        self.colors_sum = torch.zeros((self.opacities.shape[0], num_harm), device=device)
        with torch.no_grad():
            self.opacities[:] = init_op
            self.colors[:] = 0
    def carve(self, ordir_tuple, N_points, inv_depth=1.2):
        with torch.no_grad():
            ori = ordir_tuple[0][:, None,:]

            # WARNING: Assuming constant distance
            distances = 8/(N_points-1)
            scatter_points = torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth
            p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])    

            # extract valid indices
            inds_3d = torch.floor(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long())
            mesh_coords[torch.logical_not(in_bounds)] = 0
            
            self.opacities[mesh_coords] = 0
            
    def color(self, ordir_tuple, pixels, N_points, inv_depth=1.2):
        with torch.no_grad():
            ori = ordir_tuple[0][:, None,:]

            # WARNING: Assuming constant distance
            distances = 8/(N_points-1)
            scatter_points = torch.linspace(0,10, N_points, device=device)[None, :]/inv_depth
            p = ori + scatter_points[:,:,None]*(ordir_tuple[1][:, None, :])    

            # extract valid indices
            inds_3d = torch.floor(self.descartes_to_indices(p))
            in_bounds = self.in_bounds_indices(inds_3d)
            # meshgrid coordinates
            mesh_coords = self.flatten_3d_indices(inds_3d.long()).long()
            mesh_coords[torch.logical_not(in_bounds)] = 0
            opacities = self.opacities[mesh_coords]*in_bounds.float() # not_in bounds: 0 opacity
            opacities = opacities*distances
            cumsum_opacities = torch.zeros_like(opacities, device=device)
            cumsum_opacities[:,1:] = torch.cumsum(opacities[:,:-1], 1)

            transp_term = torch.exp(-cumsum_opacities)*(1-torch.exp(-opacities))
            harmonics_base = self.view_harmonics(ordir_tuple[1])
            harmonics = harmonics_base[:, None, :]*pixels[:,:,None]
            
            self.colors[mesh_coords,:] += harmonics[:, None, :,: ]*transp_term[..., None, None]
            self.colors_sum[mesh_coords] += harmonics_base[:, None, :]*transp_term[..., None]