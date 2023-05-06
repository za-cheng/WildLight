import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.physicalshader import PhysicalRenderingNetwork, PhysicalNeRF
from models.renderer import NeuSRenderer
from models.ssim import loss_dssim
import xatlas
import scipy
import scipy.interpolate
from tabulate import tabulate
import gdown

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from models.uv_mapping import generate_uv_map

def parametrize(vertices, faces):
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.padding = 2
    pack_options.create_image = True
    atlas.generate(chart_options, pack_options, True)
    return atlas[0], max(atlas.chart_image.shape)


def batch_feed(func, data, batch_size=4096):
    ret_data = dict()

    for chunk in tqdm(np.array_split(data, data.shape[0]//batch_size, axis=0)):
        rst = func(chunk)
        for k in rst:
            ret_data[k] = ret_data.get(k, list())
            ret_data[k].append(rst[k])
    
    return {k: np.concatenate(ret_data[k]) for k in ret_data}

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, download_dataset=False):
        self.device = torch.device('cuda')
        self.case = case

        if download_dataset:
            self.download_dataset() # download dataset to default location

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        if args.mode.startswith('validate_image'):
            self.dataset = Dataset(self.conf['dataset_val'])
        elif args.mode == 'validate_mesh':
            self.dataset = Dataset(self.conf['dataset'], load_images=False)
        else:
            self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.rgb_batch_size = self.conf.get_int('train.rgb_batch_size', self.conf.get_int('train.batch_size', 0))
        self.alpha_batch_size = self.conf.get_int('train.alpha_batch_size', self.conf.get_int('train.batch_size', 0))
        assert self.rgb_batch_size*self.alpha_batch_size > 0
        self.batch_size = self.rgb_batch_size
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        self.samples_per_pixel = self.conf.get_int("train.samples_per_pixel", 1)
        # Weights
        self.rgb_loss_type = self.conf.get('train.rgb_loss_type', 'l1')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.dssim_weight = self.conf.get_float('train.dssim_weight')
        self.dssim_window_size = self.conf.get_int('train.dssim_window_size')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = PhysicalNeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = PhysicalRenderingNetwork(self.conf['model.physical_rendering_network'], self.conf['model.brdf_settings']).to(self.device)


        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     brdf_settings=self.conf['model.brdf_settings'],
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
    
    def download_dataset(self):

        if self.case == 'bunny' and not os.path.exists('datasets/synthetic/bunny'):
            gdown.download(id="1wDyQfYL4IKvCOuwnGW9vgr66cXNWRm4B", output="datasets/bunny.zip")
            gdown.extractall("datasets/bunny.zip", "datasets")
            os.remove("datasets/bunny.zip")
        if self.case == 'armadillo' and not os.path.exists('datasets/synthetic/armadillo'):
            gdown.download(id="185YLMR3HJ7QnqYy39RBxmfRn52oajf_u", output="datasets/armadillo.zip")
            gdown.extractall("datasets/armadillo.zip", "datasets")
            os.remove("datasets/armadillo.zip")
        if self.case == 'legocar' and not os.path.exists('datasets/synthetic/legocar'):
            gdown.download(id="1xJjPWSKT_CfTFVdvrxRXFgHSO_RZV7UN", output="datasets/legocar.zip")
            gdown.extractall("datasets/legocar.zip", "datasets")
            os.remove("datasets/legocar.zip")
        if self.case == 'plant' and not os.path.exists('datasets/synthetic/plant'):
            gdown.download(id="1CBfF1YyxYGUhQ--9E-tifhpLlZebV2mF", output="datasets/plant.zip")
            gdown.extractall("datasets/plant.zip", "datasets")
            os.remove("datasets/plant.zip")
        if self.case == 'bulldozer' and not os.path.exists('datasets/real/bulldozer'):
            gdown.download(id="16MrLpocSsWBv2x9izMl65Hs-92qa3VKr", output="datasets/bulldozer.zip")
            gdown.extractall("datasets/bulldozer.zip", "datasets")
            os.remove("datasets/bulldozer.zip")
        if self.case == 'cokecan' and not os.path.exists('datasets/real/cokecan'):
            gdown.download(id="1LzG8L5_vSuPBFJ8u1IqWM0umILWjDbJf", output="datasets/cokecan.zip")
            gdown.extractall("datasets/cokecan.zip", "datasets")
            os.remove("datasets/cokecan.zip")
        if self.case == 'face' and not os.path.exists('datasets/real/face'):
            gdown.download(id="1PziEgW_X_f4hNAiE_WWmbvALrzD06sJR", output="datasets/face.zip")
            gdown.extractall("datasets/face.zip", "datasets")
            os.remove("datasets/face.zip")   


    @torch.no_grad()
    def validate_images(self, resolution_level=1):
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        psnr, ssim = [], []


        os.makedirs(os.path.join(self.base_exp_dir, 'novel_view'), exist_ok=True)

        pbar = tqdm(range(self.dataset.n_images))

        for i in pbar:
            img = self.dataset.image_at(i, resolution_level, False)
            mask = self.dataset.mask_at(i, resolution_level, False)

            img_rendered = self.validate_image(i, resolution_level, printf=pbar.set_description)

            img = img.clip(0, self.dataset.saturation_intensity)
            img_rendered = img_rendered.clip(0, self.dataset.saturation_intensity)

            img[mask < 0.5] = 0
            img_rendered[mask < 0.5] = 0

            #print(img.shape, img_rendered.shape)
            psnr.append(peak_signal_noise_ratio(img, img_rendered, data_range=img.max()))
            ssim.append(structural_similarity(img, img_rendered, data_range=img.max(),multichannel=True))

            cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view', f"{i:02}.exr"), np.concatenate([img_rendered[...,::-1], mask.reshape(img_rendered.shape[:-1] + (-1,))[...,:1]], axis=-1))

        psnr = np.array(psnr)
        ssim = np.array(ssim)

        has_flash_light = np.stack(self.dataset.light_energies).max(-1) > 0

        psnr_ambient, psnr_flash, psnr_all = psnr[has_flash_light==False].mean(), psnr[has_flash_light].mean(), psnr.mean()
        ssim_ambient, ssim_flash, ssim_all = ssim[has_flash_light==False].mean(), ssim[has_flash_light].mean(), ssim.mean()

        print( tabulate([['img', 'ambient', 'flash', 'all'], 
                         ['psnr', psnr_ambient, psnr_flash, psnr_all], 
                         ['ssim', ssim_ambient, ssim_flash, ssim_all]], headers='firstrow', tablefmt='github'))

        return psnr, ssim    

    def train(self, sample_mode="batch"):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = max(0, self.end_iter - self.iter_step)
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            img_idx = self.iter_step % len(image_perm)
            cap_pixel_val = self.dataset.cap_pixel_val(image_perm[img_idx])

            seed = int(time.time() * 1000) % 1000000 + iter_i

            def closure(mode="rgb", x_shift=0, y_shift=0):
                if sample_mode == "patch":  
                    data = self.dataset.gen_random_rays_patch(image_perm[img_idx], int(np.sqrt(self.rgb_batch_size)), mode=="rgb", shift=(x_shift,y_shift), seed=seed)
                else:
                    data = self.dataset.gen_random_rays_at(image_perm[img_idx], self.rgb_batch_size, mode=="rgb", shift=(x_shift,y_shift), seed=seed)
                
                light_o, light_lumen = self.dataset.gen_light_params(image_perm[img_idx])
                rays_o, rays_d, true_rgb, mask = data[..., :3], data[..., 3: 6], data[..., 6: 9], data[..., 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                mask = (mask > 0.5).float()

                if mode == "rgb":
                    render_out = self.renderer.render(rays_o, rays_d, light_o, light_lumen, near, far, 
                                                    background_rgb=background_rgb,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio())
                elif mode == 'alpha':

                    render_out = self.renderer.render_alpha(rays_o, rays_d, light_o, light_lumen, near, far, 
                                                    background_rgb=background_rgb,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio())
                return render_out, true_rgb, mask, light_lumen

            def sample_pixels(n_samples):
                n_samples = int(np.sqrt(n_samples))
                ret = dict()
                keys=['color_fine', 's_val', 'cdf_fine', 'gradient_error', 'weight_max', 'weight_sum']
                for i in np.arange(0.5/n_samples,1,1/n_samples):
                    for j in np.arange(0.5/n_samples,1,1/n_samples):  
                        offset_x = i - 0.5
                        offset_y = j - 0.5
                        
                        render_out, true_rgb, mask, light_lumen = closure("rgb", offset_x, offset_y)
                        for k in keys:
                            ret[k] = (ret.get(k,0) + render_out[k]/(n_samples**2))

                return ret, true_rgb, mask, light_lumen

            render_out, true_rgb, mask, light_lumen = sample_pixels(self.samples_per_pixel)

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            valid_pixel_mask = ((color_fine < cap_pixel_val) | (true_rgb < cap_pixel_val)).float()
            color_error = (color_fine - true_rgb) * mask * valid_pixel_mask

            mask_sum = (mask * valid_pixel_mask).sum() + 1e-5

            if sample_mode == "patch":  
                color_dssim = loss_dssim(color_fine, true_rgb, mask>0, valid_pixel_mask>0, cap_pixel_val, self.dssim_window_size)
            else:
                color_dssim = 0

            if self.rgb_loss_type == 'l1':
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            elif self.rgb_loss_type == 'l2':
                color_fine_loss = F.mse_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            else:
                print(self.rgb_loss_type)
                raise NotImplementedError

            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            self.optimizer.zero_grad()

            loss = color_fine_loss +\
                   color_dssim * self.dssim_weight +\
                   eikonal_loss * self.igr_weight
            loss.backward()
            
            if self.mask_weight > 0:
                render_out_alpha, _, mask_alpha, _ = closure("alpha")
                mask_loss = F.binary_cross_entropy(render_out_alpha['weight_sum'].clip(1e-3, 1.0 - 1e-3), mask_alpha)

                (mask_loss * self.mask_weight).backward()
                loss = loss + mask_loss * self.mask_weight

            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_dssim, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
        
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image(log_to_tb=True)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(world_space=True, log_to_tb=True)

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    @torch.no_grad()
    def save_normal_and_depth(self, path):
        normal_maps, depth_maps = [], []

        for idx in tqdm(range(self.dataset.n_images)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
            depth_distance_ratio = self.dataset.dist_to_depth_map(idx, resolution_level=1).cpu().numpy()
            light_o, light_lum = self.dataset.gen_light_params(idx)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_normal_fine = []
            out_dist_fine = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                light_o,
                                                light_lum,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                assert feasible('gradients') and feasible('weights')
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                dists = render_out['z'] * render_out['weights'][:, :n_samples]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                    dists = dists * render_out['inside_sphere'] 
                normals = normals.sum(dim=1).detach().cpu().numpy()
                dists = (dists.sum(dim=1) / render_out['weights'][:, :n_samples].sum(dim=1)).detach().cpu().numpy()
                out_normal_fine.append(normals)
                out_dist_fine.append(dists)
                del render_out

            normal_img = np.concatenate(out_normal_fine, axis=0).reshape(H,W,3)
            normal_img = normal_img / (1e-10 + np.linalg.norm(normal_img, axis=-1, keepdims=True))
            normal_maps.append(normal_img)

            dist_img = np.concatenate(out_dist_fine, axis=0).reshape(H,W)
            depth_img = depth_distance_ratio * dist_img
            depth_maps.append(depth_img)
        
        np.savez(path, depth_maps=np.stack(depth_maps,axis=0), normal_maps=np.stack(normal_maps,axis=0))


    @torch.no_grad()
    def validate_image(self, idx=-1, resolution_level=-1, log_to_tb=False, printf=print):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        printf('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        light_o, light_lum = self.dataset.gen_light_params(idx)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              light_o,
                                              light_lum,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)])[...,::-1])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i][...,::-1])

        img = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
        if log_to_tb and self.writer:
            self.writer.add_image(f'idx:{idx}', img, global_step=self.iter_step, dataformats='HWC')

        return img
    @torch.no_grad()
    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        light_o, light_lum = self.dataset.gen_light_params_between(idx_0, idx_1, ratio)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              light_o,
                                              light_lum,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
        return img_fine

    @torch.no_grad()
    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, simplify=False, bake_texture_maps=False, texture_resolution=2048, log_to_tb=False, save_to=None):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        print(f"Extracting mesh from marching cubes at resolution {resolution}...")
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        vertices, triangles = mesh.vertices, mesh.faces

        if save_to is None:
            save_to = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}'.format(self.iter_step))

        if simplify:
            mesh = mesh.as_open3d.simplify_quadric_decimation(131072, (0.5/resolution)**2) # max face number 131072 for exported meshes
            print(f"Simplified mesh: {vertices.shape[0]} verts, {triangles.shape[0]} faces -> {len(mesh.vertices)} verts, {len(mesh.triangles)} faces.")
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

        if bake_texture_maps:
            save_dir = os.path.join('{}_export'.format(save_to))
            os.makedirs(save_dir, exist_ok=True)
            print(f"Running UV unwraping (this can take a few minutes) ...")

            coords, vertices, triangles, uv = generate_uv_map(vertices, triangles, texture_resolution)

            print(f"Baking textures at resolution {coords.shape[0]}X{coords.shape[1]}.")
            brdf_params = batch_feed(self.renderer.extract_shading_params, coords.reshape(-1,3))

            for k in ['object_normal', 'subsurface', 'metallic', 'specular', 'clearcoat', 'roughness', 'clearcoat_gloss', 'base_color']:
                if k == 'object_normal':
                    texture = (brdf_params[k].reshape(coords.shape[0], coords.shape[1], -1) + 1) * 255 * 0.5
                else:
                    texture = brdf_params[k].reshape(coords.shape[0], coords.shape[1], -1) * 255
                if k == 'object_normal' or k == 'base_color':
                    cv.imwrite(os.path.join(save_dir, f"{k}.png"), texture.clip(0, 255)[...,::-1]) # opencv saves in BGR format
                else:
                    cv.imwrite(os.path.join(save_dir, f"{k}.png"), texture.clip(0, 255)) # gray image
            
            cv.imwrite(os.path.join(save_dir, f"coords.exr"), coords.reshape(coords.shape[0], coords.shape[1], -1).astype(np.float32)[...,::-1])
            

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        

        if bake_texture_maps:
            xatlas.export(os.path.join(save_dir, 'mesh.obj'), vertices, triangles, uv)
            print(f"Saved mesh and textures under '{save_dir}'.")
        else:
            trimesh.Trimesh(vertices, triangles).export('{}.ply'.format(save_to))
        
        if log_to_tb and self.writer:
            self.writer.add_mesh('shape', torch.from_numpy(vertices).unsqueeze(0), faces=torch.from_numpy(triangles).unsqueeze(0), global_step=self.iter_step)
        logging.info('End')

    
    def interpolate_view(self, img_idx_0, img_idx_1, n_frames = 60):
        images = []
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=2))
        for i in range(n_frames):
            images.append(images[n_frames - i - 2])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write((image*256).clip(0, 255).astype(np.uint8))

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--download_dataset', action='store_true')


    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.download_dataset)

    if args.mode.startswith('train'):
        if len(args.mode.split('_')) == 2:
            runner.train(args.mode.split('_')[1])
        else:
            runner.train()
    elif args.mode == 'validate_mesh':
        with torch.no_grad():
            runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold, simplify=True, bake_texture_maps=True, texture_resolution=4096)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        with torch.no_grad():
            _, img_idx_0, img_idx_1 = args.mode.split('_')
            img_idx_0 = int(img_idx_0)
            img_idx_1 = int(img_idx_1)
            runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode.startswith('validate_image'):  # Interpolate views given two image indices
        with torch.no_grad():
            if len(args.mode.split('_')) == 2:
                resolution_level = 1
            else:
                assert len(args.mode.split('_')) == 3
                resolution_level = int(args.mode.split('_')[2])

        runner.validate_images(resolution_level)
            
            

