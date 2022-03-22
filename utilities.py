import os
from tqdm import tqdm
import imageio
import numpy as np
import json
from torch.utils.data import Dataset
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio




# FROM PLENOXELS
def get_data(root="../nerf_example_data/nerf_synthetic/lego", stage="train", background=False):
    all_c2w = []
    all_gt = []

    data_path = os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    print('LOAD DATA', data_path)
    j = json.load(open(data_json, 'r'))

    for frame in tqdm(j['frames']):
        fpath = os.path.join(data_path, os.path.basename(
            frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        if background:
            im_gt = (im_gt[..., 3:]==0)*1.0
        else:    
            im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5,
                       np.arange(H, dtype=np.float32) + 0.5, indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# CUSTOM

def get_cameras_centers(rays_or_dir):
    len_rays = len(rays_or_dir)
    centers = np.zeros((len_rays, 3))
    for i in range(len_rays):
        # all cameras share same center
        centers[i, :] = rays_or_dir[i][0][0, 0]
    return centers

def reshape_numpy(arr, red_fac):
    H,W = arr.shape[:2]
    im = Image.fromarray(np.uint8(255*arr))
    im = im.resize((H//red_fac,W//red_fac), Image.ANTIALIAS)
    return np.array(im)/255.0

def reduce_data(all_c2w, all_gt, focal, red_fac):
    H,W = all_gt[0].shape[:2]
    red_ims = [reshape_numpy(gt,red_fac) for gt in all_gt]
    
    ordir_rays = []
    for c2w in all_c2w:
        ray_np = get_rays_np(H,W, focal, c2w)
        oris = ray_np[0][::red_fac,::red_fac]
        direct = ray_np[1][::red_fac,::red_fac] # direction. optimal fac:3
        ordir_rays.append((oris, direct))
    return red_ims, ordir_rays

def regular_3d_indexes(n):
    i = np.arange(n)
    j = np.arange(n)
    k = np.arange(n)
    return np.transpose([np.tile(i, len(j)*len(k)), np.tile(np.repeat(j, len(i)), len(k)), np.repeat(k, len(i)*len(j))])

def rolling_average(p, k=100):
    p2 = np.zeros((p.shape[0]-k))
    for i in range(k):
        p2 += p[i:-(k-i)]
    return p2/k

def compute_psnr(grid, disp_rays_test, disp_ims_test, N_points=500):
    m = np.zeros(len(disp_ims_test))
    for i in tqdm(range(len(disp_ims_test))):
        with torch.no_grad():
            new_im = grid.render_large_image_from_rays(disp_rays_test[i], (N_points, 1.2))
            m[i] = peak_signal_noise_ratio(new_im, disp_ims_test[i].astype('float32'))
    return m.mean()
    
# DATASETS

class RayDataset(Dataset):
    def __init__(self, target_ims, ordir_rays, device):
        im_w = target_ims[0].shape[0]

        self.tensor_rays = [] # (tuple (origin, first_point))
        self.tensor_target_pixels = []
        
        for image_ind in tqdm(range(len(ordir_rays))):
            direct = torch.tensor(ordir_rays[image_ind][1], dtype=torch.float32)
            ori =  torch.tensor(ordir_rays[image_ind][0], dtype=torch.float32)
            pixels = torch.tensor(target_ims[image_ind], dtype=torch.float32)
            Lor = list(ori.flatten(0,1))
            Ldir = list(direct.flatten(0,1))
            Lpix = list(pixels.flatten(0,1))
            self.tensor_rays += [(Lor[i], Ldir[i]) for i in range(len(Lor))]
            self.tensor_target_pixels += Lpix
    def __getitem__(self, index):
        return self.tensor_rays[index], self.tensor_target_pixels[index]
    def __len__(self):
        return len(self.tensor_rays)

# FLY around
def get_rot_x(angle):
    Rx = np.zeros(shape=(3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)
    return Rx

def get_rot_y(angle):
    Ry = np.zeros(shape=(3, 3))
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)
    Ry[1, 1] = 1
    return Ry

def get_rot_z(angle):
    Rz = np.zeros(shape=(3, 3))
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)
    Rz[2, 2] = 1
    
    return Rz

def create_rotation_transformation_matrix(center, theta=0,phi=0,alpha=0):
    
    out = np.identity(4)
    net = np.identity(3)
    
    for transf, angle in zip([get_rot_y,  get_rot_z,get_rot_x],[alpha, theta+np.pi/2, phi+np.pi/2]):
        net = np.matmul(net, transf(angle))
        
    out[:3,:3] = net
    out[:3, -1] = center
    return out

def create_rotation_matrices(height, view_angle=-20, n=10):

    t = np.linspace(0,2*np.pi, n+1)[:-1]
    cust_centers = np.zeros((n,3))

    radius = np.sqrt(17-height**2)
    cust_centers[:,0] = np.cos(t)*radius
    cust_centers[:,1] = np.sin(t)*radius
    cust_centers[:,2] = height
    
    return [create_rotation_transformation_matrix(cust_centers[i], t[i], np.pi*view_angle/180) for i in range(n)]

