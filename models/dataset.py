from unittest.mock import patch
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def compose_transf(P1, P2):

    R1 = P1[:3,:3]
    T1 = P1[:3,3:4]
    R2 = P2[:3,:3]
    T2 = P2[:3,3:4]

    R = R1@R2
    T = T1 + R1@T2
    return np.concatenate([R,T], axis=-1)

def make4x4(P):
    assert P.shape[-1] == 4
    assert len(P.shape) == 2
    assert P.shape[0] == 3 or P.shape[0] == 4
    ret = np.eye(4)
    ret[:P.shape[0]] = P
    return ret

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def img_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])

class Dataset:
    def __init__(self, conf, load_images=True):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.ignore_0 = conf.get("ignore_zero_RGB", False)

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        
        if load_images:
            if os.path.exists(os.path.join(self.data_dir, 'images.npy')):
                self.images_np = np.load(os.path.join(self.data_dir, 'images.npy'))
                max_intensity = float(camera_dict.get("max_intensity"))
                self.images_np = self.images_np / max_intensity * 5
                self.saturation_intensity = 5
                self.images_id = list(range(len(self.images_np)))
                

            elif len(glob(os.path.join(self.data_dir, 'image/*.exr'))) > 0: 
                # exr format
                self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.exr')))
                self.images_np = np.stack([cv.imread(im_name, -1)[...,2::-1] for im_name in self.images_lis])
                self.saturation_intensity = float(camera_dict.get("max_intensity", np.inf))
                self.images_id = [img_id(p) for p in self.images_lis]
            else:
                # png format
                self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
                self.images_np = np.stack([cv.imread(im_name)[...,::-1] for im_name in self.images_lis]) / 256.0
                self.saturation_intensity = float(camera_dict.get("max_intensity", 255 / 256.0 - 0.001))
                self.images_id = [img_id(p) for p in self.images_lis]
    
            self.n_images = len(self.images_np)
        

            if conf.get('ignore_mask', False):
                self.masks_np = np.ones_like(self.images_np)
                self.no_mask = True
            else:
                mask_dic = {img_id(p):p for p in glob(os.path.join(self.data_dir, 'mask/*.png'))}
                self.masks_lis = [mask_dic[i] for i in self.images_id]
                self.masks_np = (np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0 > 0.5).astype(np.float) 
                self.no_mask = False

            
            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W

            self.images_np = None
            self.masks_np = None
            print(f'Load images: End ({self.n_images} images at {self.H}X{self.W}) ')
        else:
            self.images_id = sorted([int(k.split('_')[-1]) for k in camera_dict if k.startswith('world_mat_')])
            self.n_images = len(self.images_id)
            # example_input = cv.imread(glob(os.path.join(self.data_dir, 'mask/*.png'))[0])
            # self.H, self.W = example_input.shape[0], example_input.shape[1]
            
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.images_id]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [make4x4(camera_dict['scale_mat_%d' % idx]).astype(np.float32) for idx in self.images_id]

        self.light_energies = [camera_dict.get('light_energy_%d' % idx, np.zeros(3)).astype(np.float32) for idx in self.images_id]

        print(f"{(np.stack(self.light_energies).max(-1) > 0).sum()} out of {len(self.light_energies)} images have flashlight on.")

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = compose_transf(world_mat, scale_mat)
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]

        self.intrinsics_all = self.intrinsics_all.to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = self.intrinsics_all_inv.to(self.device)  # [n_images, 4, 4]
        
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = make4x4(np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0'])
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]


    def cap_pixel_val(self, img_idx):
        return self.saturation_intensity

    def dist_to_depth_map(self, img_idx, resolution_level=1):
        '''
        returns ratio of depth over distance, of size (H,W)
        '''
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        return rays_v[:,:,2].transpose(0, 1)

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_light_params(self, img_idx):
        light_o = self.pose_all[img_idx, :3, 3].cuda()
        light_lum = torch.from_numpy(self.light_energies[img_idx]).cuda()
        return light_o, light_lum

    def gen_random_rays_patch(self, img_idx, patch_size, foreground_only=True, shift=(0,0), seed=None):
        if seed is not None:
            np.random.seed(seed)
        patch_size = min(patch_size, self.W, self.H)
        assert all((s-0.5)*(s+0.5)<=0 for s in shift)
        if foreground_only:
            for i in range(10):
                patch_origin_x = np.random.randint(0, self.W - patch_size)
                patch_origin_y = np.random.randint(0, self.H - patch_size)
                pixels_y, pixels_x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
                pixels_x = (pixels_x + patch_origin_x).reshape(-1)
                pixels_y = (pixels_y + patch_origin_y).reshape(-1)

                color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
                if mask.sum() > 0:
                    break
            
            if mask.sum() == 0: # failed too many times, revert to mask sampling
                pixels_y, pixels_x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
                pixels_x = pixels_x[self.masks[img_idx].mean(-1) > 0.5] 
                pixels_y = pixels_y[self.masks[img_idx].mean(-1) > 0.5]

                choice = torch.from_numpy(np.random.choice(len(pixels_x), 1, replace=False))
                patch_origin_x = min(int(pixels_x[choice[0]]), self.W-patch_size)
                patch_origin_y = min(int(pixels_y[choice[0]]), self.H-patch_size)

                pixels_y, pixels_x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
                pixels_x = (pixels_x + patch_origin_x).reshape(-1)
                pixels_y = (pixels_y + patch_origin_y).reshape(-1)

                color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        else:
            patch_origin_x = np.random.randint(0, self.W - patch_size)
            patch_origin_y = np.random.randint(0, self.H - patch_size)

            pixels_y, pixels_x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
            pixels_x = (pixels_x + patch_origin_x).reshape(-1)
            pixels_y = (pixels_y + patch_origin_y).reshape(-1)

            color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
            mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3

        p = torch.stack([pixels_x+shift[0], pixels_y+shift[1], torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3

        if self.ignore_0:
            mask = mask[:,:1] * (color.sum(-1) > 0).float()
        else:
            mask = mask[:,:1]

        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask], dim=-1).cuda()

        



    def gen_random_rays_at(self, img_idx, batch_size, foreground_only=True, shift=(0,0), seed=None):
        """
        Generate random rays at world space from one camera.
        """
        if seed is not None:
            np.random.seed(seed)
        assert all((s-0.5)*(s+0.5)<=0 for s in shift)
        if foreground_only and (not self.no_mask):
            pixels_y, pixels_x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
            pixels_x = pixels_x[self.masks[img_idx].mean(-1) > 0.5] 
            pixels_y = pixels_y[self.masks[img_idx].mean(-1) > 0.5]

            batch_size = min(len(pixels_x), batch_size)

            choice = torch.from_numpy(np.random.choice(len(pixels_x), batch_size, replace=False)).cuda()
            pixels_x = pixels_x[choice]
            pixels_y = pixels_y[choice]
        else:
            pixels_x = torch.from_numpy(np.random.randint(low=0, high=self.W, size=[batch_size])).cuda()
            pixels_y = torch.from_numpy(np.random.randint(low=0, high=self.H, size=[batch_size])).cuda()

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x+shift[0], pixels_y+shift[1], torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3

        if self.ignore_0:
            mask = mask[:,:1] * (color.sum(-1) > 0).float()
        else:
            mask = mask[:,:1]

        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask], dim=-1).cuda()    # batch_size, 10

    def gen_light_params_between(self, idx_0, idx_1, ratio):
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        light_o = trans[:3].cuda()
        light_lum_0 = torch.from_numpy(self.light_energies[idx_0]).cuda()
        light_lum_1 = torch.from_numpy(self.light_energies[idx_1]).cuda()
        light_lum = light_lum_0 * (1.0 - ratio) + light_lum_1 * ratio
        return light_o, light_lum

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near.clip(min=0), far

    def image_at(self, idx, resolution_level, to256=True):
        if to256:
            ratio = 256
        else:
            ratio = 1
        img = self.images[idx].numpy() * ratio
        img = (cv.resize(img, (self.W // resolution_level, self.H // resolution_level)))
        if to256:
            img = img.clip(0, 255)
        return img
    
    def mask_at(self, idx, resolution_level, to256=True):
        mask = self.masks[idx].numpy()
        mask = (cv.resize(mask, (self.W // resolution_level, self.H // resolution_level)))

        if to256:
            mask = (mask * 256).clip(0, 255)
        
        return mask


