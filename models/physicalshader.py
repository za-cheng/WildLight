from ast import Lambda
from math import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedder import get_embedder
from numpy import pi

def _format_brdf_params(param, newdim=True):
    '''
    param: tensor of shape (..., n_lobes)
    return: reshaped tensor of shape (n_lobes, ..., 1) if newdim, otherwise (n_lobes, ...)
    '''
    if newdim:
        return param.movedim(-1, 0).unsqueeze(-1)
    else:
        return param.movedim(-1, 0)

def unpack_brdf_params_GGX(brdf_params, n_lobes=None, trichromatic=False, constant_r0=False):
    '''
    unpack brdf_params with given format:
    args: 
        brdf_params : tensor of shape (..., D)
    returns:
        diffuse_albedo: tensor of shape (..., 3)
        specular_albedo: tensor of shape (n_lobes, ..., 3) or (n_lobes, ..., 1), depending on whether the model is trichromatic
        roughness: tensor of shape (n_lobes, ..., 1)
        r0: tensor of shape (n_lobes, ..., 1)
    '''
    if trichromatic:
        n_channels = 3
    else:
        n_channels = 1

    if n_lobes is None:
        n_lobes = (brdf_params-3) // (2+n_channels)
    
    assert n_lobes*(2+n_channels) + 3 == brdf_params.shape[-1]

    diffuse_albedo, specular_albedo, roughness, r0 = brdf_params.split([3, n_lobes*n_channels, n_lobes, n_lobes], -1)

    diffuse_albedo = F.relu(diffuse_albedo)
    specular_albedo = F.relu(specular_albedo).reshape(specular_albedo.shape[:-1] + (n_channels, n_lobes))
    roughness = F.sigmoid(roughness)
    if constant_r0:
        r0 = F.sigmoid(r0)
    else:
        r0 = torch.ones_like(r0)
    
    return diffuse_albedo, _format_brdf_params(specular_albedo, False), _format_brdf_params(roughness), _format_brdf_params(r0)


def unpack_brdf_params_burley(brdf_params, brdf_config):
    '''
    unpack brdf_params with given format:
    args: 
        brdf_params : tensor of shape (..., D)
    returns:
        diffuse_albedo: tensor of shape (..., 3)
        specular_albedo: tensor of shape (n_lobes, ..., 3) or (n_lobes, ..., 1), depending on whether the model is trichromatic
        roughness: tensor of shape (n_lobes, ..., 1)
        r0: tensor of shape (n_lobes, ..., 1)
    '''
    # [0,1]
    # subsurface  1
    # metallic  1
    # specular 1
    # clearcoat 1
    # roughness 1
    # clearcoat_gloss 1
    # base_color 3

    
    brdf_params = F.sigmoid(brdf_params)

    subsurface, metallic, specular, clearcoar, roughness, clearcoat_gloss, base_color = torch.split(brdf_params, [1,1,1,1,1,1,3], dim=-1)

    return subsurface*brdf_config.get_float("subsurface", 1.0),\
           metallic*brdf_config.get_float("metallic", 1.0),\
           specular*brdf_config.get_float("specular", 1.0),\
           clearcoar*brdf_config.get_float("clearcoar", 1.0),\
           roughness, clearcoat_gloss, base_color






def dot(tensor1, tensor2, dim=-1, keepdim=False, non_negative=False, epsilon=1e-6) -> torch.Tensor:
    x =  (tensor1 * tensor2).sum(dim=dim, keepdim=keepdim)
    if non_negative:
        x = torch.clamp_min(x, epsilon)
    return x

def _GGX_smith(hz, roughness, epsilon=1e-10):
    hz_sq = hz**2
    roughness_sq = roughness**2
    D = roughness_sq / pi / (hz_sq * (roughness_sq-1) + 1 + epsilon)**2 # GGX
    G = 2 / ( torch.sqrt(1 + roughness_sq * (1/hz_sq - 1)) + 1)

    return D, G

def _CC_smith(hz, roughness):
    hz_sq = hz**2
    roughness_sq = roughness**2
    D = (roughness_sq-1) / (pi*2*torch.log(roughness)*(1+(roughness_sq-1)*hz_sq))
    G = 2 / ( torch.sqrt(1 + 0.0625 * (1/hz_sq - 1)) + 1)

    return D, G

def _diffuse(hz, roughness, subsurface):
    F_D90 = 2*roughness + 0.5
    base_diffuse = (1 + (F_D90 - 1)*(1-hz)**5)**2 / pi

    F_SS = (1 + (roughness - 1)*(1-hz)**5)
    subsurface_diffuse = 1.25 / pi * (F_SS**2 * (0.5/(hz*0.9999+0.0001)-0.5) + 0.5)

    return (1-subsurface)*base_diffuse + subsurface*subsurface_diffuse



def _GGX_shading(normal_vecs, incident_vecs, view_vecs, roughness, r0=None, epsilon=1e-6):
    '''
    normal_vecs, incident_vecs, view_vecs: (...,3) normalised vectors
    roughness: (k_lobes, ..., 1) rms slope
    r0: (k_lobes, ..., 1)fresnel factor
    returns: (...,k_lobes) specular factors
    '''
    half_vecs = torch.nn.functional.normalize(incident_vecs+view_vecs, dim=-1)
    
    roughness = 0.0001 + (roughness) * (1-epsilon-0.0001)
    # Beckmann model for D
    h_n = dot(half_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    # cos_alpha_sq = h_n**2 # (...)
    # cos_alpha_sq = cos_alpha_sq.unsqueeze(dim=-1) # (..., 1)
    # cos_alpha_r_sq = torch.clamp_min(cos_alpha_sq*(roughness**2), epsilon) # ([k_lobes,] ..., 1)
    # # D = torch.exp( (cos_alpha_sq - 1) /  cos_alpha_r_sq ) / \
    # #     ( np.pi * cos_alpha_r_sq * cos_alpha_sq ) # ([k_lobes,] ..., 1)  # Beckmann

    # # GGX model
    # roughness_sq = roughness**2
    # D = roughness_sq / pi / (cos_alpha_sq * (roughness_sq-1) + 1 + epsilon)**2 # GGX

    # # Geometric term G
    # v_n = dot(view_vecs, normal_vecs, non_negative=True) # (...)
    v_h = dot(half_vecs, view_vecs, non_negative=True, keepdim=True) # (..., 1)
    # i_n = dot(incident_vecs, normal_vecs, non_negative=True) # (...)

    v_n = h_n
    i_n = h_n

    # # G = torch.clamp_max(torch.min(i_n, v_n) * 2 * h_n / v_h, 1) # (...)
    # # G = G.unsqueeze(dim=-1) # (..., 1)
    
    # # GGX
    # mask_G = (v_h > 0).float().unsqueeze(dim=-1) # (..., 1)
    # G = 2 / ( torch.sqrt(1 + roughness_sq * (1/v_n.unsqueeze(dim=-1)**2 - 1)) + torch.sqrt(1 + roughness_sq * (1/i_n.unsqueeze(dim=-1)**2 - 1)) )
    # G = G * mask_G

    D, G = _GGX_smith(h_n, roughness, epsilon)

    # Schlick's approximation for F
    if r0 is None:
        F = 1
    else:
        F = r0 + (1-r0) * ((1 - v_h) ** 5) # ([k_lobes,] ..., 1)

    ret = (D*F*G) / (pi*i_n*v_n+epsilon) # ([k_lobes,] ..., 1)

    return ret

def _burley_shading(normal_vecs, incident_vecs, view_vecs, brdf_params, brdf_config):

    half_vecs = torch.nn.functional.normalize(incident_vecs+view_vecs, dim=-1)
    h_n = dot(half_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    
    subsurface, metallic, specular, clearcoat, roughness, clearcoat_gloss, base_color = unpack_brdf_params_burley(brdf_params, brdf_config)

    clearcoat_roughness = 0.1 - 0.099 * clearcoat_gloss
    alpha = 0.0001 + (roughness**2) * (1-0.0002)

    D_metal, G_metal = _GGX_smith(h_n, alpha) #(..., 1)
    D_clearcoat, G_clearcoat = _CC_smith(h_n, clearcoat_roughness) #(..., 1)

    if brdf_config.get("mask_shadowing", "joint") == "independent":
        G_metal = G_metal * G_metal
        G_clearcoat = G_clearcoat * G_clearcoat
    
    F_metal = (1-metallic)*specular*0.08 + metallic*base_color # (..., 3)
    F_clearcoat = 0.04

    r_specular = D_metal * G_metal * F_metal / (4 * h_n * h_n) # (..., 3)
    r_clearcoat = D_clearcoat * G_clearcoat * F_clearcoat / (4 * h_n * h_n) # (..., 1)
    r_diffuse = _diffuse(h_n, roughness, subsurface) * base_color #(..., 3)

    return (1-metallic)*r_diffuse, (r_specular + 0.25*clearcoat*r_clearcoat)


def _apply_lighting_GGX(points, normals, view_dirs, light_dirs, irradiance, diffuse_albedo, specular_albedo, roughness, r0):
    """
    Args:
        points: torch tensor of shape (..., 3).
        normals: torch tensor of shape (..., 3), from inside to outside, only directions matter
        view_dirs: torch tensor of shape (..., 3), from viewpoint to object, only directions matter
        light_dirs: torch tensor of shape (..., 3), from light to object, only directions matter
        irradiance: torch tensor of shape (..., 3) or (..., 1), nonnegative
        diffuse_albedo: torch tensor of shape (..., 3) or (..., 1)
        diffuse_albedo: torch tensor of shape ([k_lobes], ..., 3) or ([k_lobes], ..., 1)
        roughness: torch tensor of shape ([k_lobes], ..., 1)
        r0: torch tensor of shape ([k_lobes], ..., 1)

    Returns:
        diffuse_color: # (..., 3) or (..., 1)
        specular_color: # ([k_lobes,] ..., 3) or ([k_lobes,] ..., 1)
    """
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible

    specular_reflectance = _GGX_shading(normals, -light_dirs_, -view_dirs, roughness, r0) * specular_albedo # ([k_lobes,] ..., 3) or ([k_lobes,] ..., 1)
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)
    
    diffuse_color = diffuse_albedo * irradiance # (..., 3) or (..., 1)
    specular_color = specular_reflectance * irradiance # ([k_lobes,] ..., 3) or ([k_lobes,] ..., 1)

    return diffuse_color, specular_color


def _apply_shading_burley(points, normals, view_dirs, light_dirs, irradiance, brdf_params, brdf_config):
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)

    diffuse, non_diffuse = _burley_shading(normals, -light_dirs_, -view_dirs, brdf_params, brdf_config)
    return diffuse*irradiance, non_diffuse*irradiance



class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, brdf_params, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.relu(x)
        return x

class PhysicalRenderingNetwork(nn.Module):
    def __init__(self,
                 config,
                 brdf_config,
                 ):
        super().__init__()

        self.n_lobes = brdf_config.get("n_specular_lobes", None)
        self.trichromatic = brdf_config.get("type") != "GGX" or brdf_config.get("trichromatic_specular", False)
        self.constant_r0 = brdf_config.get("ignore_Fresnel", False)
        self.no_nvs_grad = config.get("no_grad", False)
        self.ambient_net = RenderingNetwork(**config["ambient_network"])
        self.bsdf_type = brdf_config['type']
        self.bsdf_config = brdf_config
        self.is_darkroom = config.get("darkroom", False)
        self.n_brdf_dim = brdf_config.get_int("dims")

        self.gamma = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    def flash_light_gamma(self):
        g = self.gamma
        m = torch.clamp(g, min=-1.0, max=1.0)
        return m * (g - 0.5*m)

    def forward(self, points, normals, view_dirs, light_origin, light_lum, brdf_params, feature_vectors):
        
        
        light_dir = points - light_origin
        irradiance = light_lum / (light_dir*light_dir).sum(-1,keepdim=True)

        if self.bsdf_type == "GGX":
            diffuse_albedo, specular_albedo, roughness, r0 = unpack_brdf_params_GGX(brdf_params, self.n_lobes, self.trichromatic, self.constant_r0)
            diffuse_active_color, specular_active_color = _apply_lighting_GGX(
                                            points, normals, view_dirs, light_dir, irradiance,
                                            diffuse_albedo, specular_albedo, roughness, r0
                                            )
            specular_active_color = specular_active_color.sum(0)

        elif self.bsdf_type == "Burley":
            diffuse_active_color, specular_active_color = \
                _apply_shading_burley(points, normals, view_dirs, light_dir, irradiance, brdf_params, self.bsdf_config)

        else:
            raise NotImplementedError

        if self.is_darkroom:
            ambient_color = 0
        else:
            ambient_color = self.ambient_net(
                                            points, 
                                            normals, 
                                            view_dirs, 
                                            brdf_params, 
                                            feature_vectors
                                        )

            

        return ambient_color + (diffuse_active_color + specular_active_color) * self.flash_light_gamma()


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class PhysicalNeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(PhysicalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 6)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, light_origin, light_lum, return_density_only=False):

        if input_pts.shape[-1] == 4: # (..., 4)
            points = input_pts[..., :-1] / input_pts[..., -1:]
        else: # (..., 3)
            points = input_pts
        light_dir = points - light_origin
        irradiance = light_lum / (light_dir*light_dir).sum(-1,keepdim=True) # (..., 3) or (..., 1)

        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
            

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            if return_density_only:
                return alpha, None
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb_environ, rgb_active = torch.split(self.rgb_linear(h), 3, dim=-1)
            return alpha, rgb_environ + rgb_active * irradiance
        else:
            assert False