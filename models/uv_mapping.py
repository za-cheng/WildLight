
import numpy as np
import xatlas
import math
import scipy
import scipy.interpolate
import tqdm
import cv2

def parametrize(vertices, faces, padding):
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.padding = padding
    pack_options.create_image = True
    atlas.generate(chart_options, pack_options, False)
    (vmap, triangles, uv), resolution = atlas[0], atlas.chart_image.shape[:2]
    return vertices[vmap], triangles, uv, resolution

#@numba.jit(nopython=True)
#@numba.jit()
def rasterize_upper_triangle(atlas, up:float, down:float, x:float, left:float, right:float, triangle_id:int):
    i_up = math.ceil(up)
    i_down = math.floor(down)
    width = right - left
    for i in range(i_up, i_down+1):
        ratio = (i - up) / (down - up) 
        i_left = x + (left - x) * ratio
        i_right = x + (right - x) * ratio
        for j in range(math.ceil(i_left), math.floor(i_right) + 1):
            atlas[i,j] = triangle_id

#@numba.jit(nopython=True)
#@numba.jit()
def rasterize_lower_triangle(atlas, up:float, down:float, x:float, left:float, right:float, triangle_id:int):
    i_up = math.ceil(up)
    i_down = math.floor(down)
    width = right - left
    for i in range(i_up, i_down+1):
        ratio = (down - i) / (down - up) 
        i_left = x + (left - x) * ratio
        i_right = x + (right - x) * ratio
        for j in range(math.ceil(i_left), math.floor(i_right) + 1):
            atlas[i,j] = triangle_id

#@numba.jit(nopython=True)
#@numba.jit()
def uv_to_ij(uv, height:float, width:float):
    u, v = uv[...,0] % 1, uv[...,1] % 1
    j = u * (width - 1)
    i = (1-v) * (height - 1)
    return np.stack([i,j], axis=-1)

#@numba.jit(nopython=True)
#@numba.jit()
def ij_to_uv(ij, height:float, width:float):
    i, j = ij[...,0], ij[...,1]
    u = j / (width - 1)
    v = 1 - i / (height - 1)
    return np.stack([u,v], axis=-1)

#@numba.jit(nopython=True)
#@numba.jit()
def break_up_upper_triangles(top, left, right):
    assert left[0] == right[0]
    assert top[0] < left[0]
    if left[1] > right[1]:
        tmp = left
        left = right
        right = tmp
    return True, top[0], left[0], top[1], left[1], right[1]

#@numba.jit(nopython=True)
#@numba.jit()
def break_up_lower_triangles(down, left, right):
    assert left[0] == right[0]
    assert down[0] > left[0]
    if left[1] > right[1]:
        tmp = left
        left = right
        right = tmp
    return False, left[0], down[0], down[1], left[1], right[1]

#@numba.jit(nopython=True)
#@numba.jit()
def break_up_triangles(verts):
    a,b,c = verts[np.argsort(verts[:,0])]
    if a[0] < b[0] and b[0] < c[0]:
        d = (b[0], ( c[1]*(b[0]-a[0]) + a[1]*(c[0]-b[0]) ) / (c[0]-a[0]) )
        return (break_up_upper_triangles(a, b, d), break_up_lower_triangles(c, b, d))
    elif a[0] < b[0]:
        return (break_up_upper_triangles(a, b, c),)
    elif b[0] < c[0]:
        return (break_up_lower_triangles(c, a, b),)
    else:
        return ()

#@numba.jit(nopython=True)
#@numba.jit()
def rasterize_triangles(atlas_size, triangles, uv):
    height, width = atlas_size
    atlas = np.zeros(atlas_size, dtype=np.int64) - 1
    ij = uv_to_ij(uv, height, width)
    for tiangle_id, triangle in enumerate(tqdm.tqdm(triangles)):
        verts_ij = ij[triangle]
        for is_upper, up, down, x, left, right in break_up_triangles(verts_ij):
            if is_upper:
                rasterize_upper_triangle(atlas, up, down, x, left, right, tiangle_id)
            else:
                rasterize_lower_triangle(atlas, up, down, x, left, right, tiangle_id)
    return atlas

def cartesian_2_barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(axis=-1)
    d01 = (v0 * v1).sum(axis=-1)
    d11 = (v1 * v1).sum(axis=-1)
    d20 = (v2 * v0).sum(axis=-1)
    d21 = (v2 * v1).sum(axis=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    bary = np.stack([u, v, w], axis=-1)
    return bary

def barycentric2_cartesian(p, a, b, c):
    u,v,w = p[...,0], p[...,1], p[...,2]
    return u[...,None]*a + v[...,None]*b + w[...,None]*c

def dialation_mask(mask, padding=1):
    kernel = np.ones((padding*2+1,padding*2+1), np.uint8)
    dialated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool8)
    return np.logical_xor(mask, dialated_mask)



def generate_uv_map(vertices, faces, min_resolution=None, padding=1):
    verts, triangles, uvs, resolution = parametrize(vertices, faces, padding * 2)
    resize_ratio = max(min_resolution, min(resolution)) / min(resolution)
    resolution = math.ceil(resolution[0] * resize_ratio), math.ceil(resolution[1] * resize_ratio)


    coordinates_atlas = np.zeros(resolution + (3,))
    ij =  np.stack(np.meshgrid(np.arange(resolution[0]), np.arange(resolution[1]), indexing='ij'), axis=-1)
    uv_atlas = ij_to_uv(ij.astype(np.float64), resolution[0], resolution[1])
    
    triangle_id_atlas = rasterize_triangles(resolution, triangles, uvs)
    atlas_mask = triangle_id_atlas >= 0

    tri_ids = triangle_id_atlas[atlas_mask]
    verts_ids = triangles[tri_ids]
    bary_coords = cartesian_2_barycentric(uv_atlas[atlas_mask], uvs[verts_ids[:,0]], uvs[verts_ids[:,1]], uvs[verts_ids[:,2]])

    coordinates_atlas[atlas_mask] = barycentric2_cartesian(bary_coords, verts[verts_ids[:,0]], verts[verts_ids[:,1]], verts[verts_ids[:,2]])

    interp_mask = dialation_mask(atlas_mask, padding)
    interp = scipy.interpolate.LinearNDInterpolator(ij[atlas_mask], coordinates_atlas[atlas_mask], 0)
    coordinates_atlas[interp_mask] = interp(ij[interp_mask])

    coordinates_atlas[np.logical_not(interp_mask | atlas_mask)] = np.nan

    return coordinates_atlas, verts, triangles, uvs
