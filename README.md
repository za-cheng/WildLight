# WildLight: In-the-wild Inverse Rendering with a Flashlight (CVPR 2023)
![Teaser](https://junxuan-li.github.io/wildlight-website/static/images/teaser.jpg)
## \[[Project page](https://junxuan-li.github.io/wildlight-website/)\]|\[[Arxiv](https://arxiv.org/abs/2303.14190)\]

## Dependencies
Conda is recommended for installing all dependencies
```bash
conda create env -f environ.yaml
conda activate wildlight
```

## Data convention
Input data are organized in a single folder, where images are saved as exr/png files similar to [NeuS](https://github.com/Totoro97/NeuS), __OR__ packed within a single npy file
```
<case_name xxx>
|-- cameras_sphere.npz    # camera & lighting parameters

|-- images.npy
Or
|-- image
    |-- 000.exr        # target image for each view, either in exr or png format
    |-- 001.exr
    Or
    |-- 000.png        
    |-- 001.png
    ...
|-- mask [optional]
    |-- 000.png        # target mask each view, if available
    |-- 001.png
    ...
```
Camera and lighting parameters are stored in `cameras_sphere.npz ` with following key strings:
- `world_mat_x`: $K_x[R_x|T_x]$ projection matrix from world coordinates to image coordinates
- `scale_mat_x`: Sim(3) transformation matrix from object coordinates to world coordinates; we will only recover shape & material inside a unit sphere ROI in object coordinates. Usually this is identical accross all views.
- `light_energy_x`: an RGB vector for flashlight intensity per view. If using a fixed power flashlight, this is set to $(1,1,1)$ for images under flashlight, or to $(0,0,0)$ for images without flashlight.
- `max_intensity`: \[optional\] a scalar indicating maximum pixel density (e.g. 255 for 8-bit images), defaults to inf


## Config
Model and traning parameters are written into config files under `confs/*.conf`. We provide three configurations for our datasets: `confs/synthetic.conf` and `confs/synthetic_maskless.conf` for our synthetic data, and `confs/real.conf` for real data.

## Running

1. Train. Run following line to download and train on the synthetic `legocar` object dataset. We provide a total of 7 objects: `bunny`, `armadillo`, `legocar`, `plant` (synthetic w/ ground turth) and `bulldozer`, `cokecan` and `face` (real scene, images only).
    ```bash
    python exp_runner.py --case legocar --conf confs/synthetic.conf --mode train --download_dataset
    ```
    Intermidiate results can be found under ``exp/legocar/masked/` folder. 
2. Mesh and texture export.
    ```bash
    python exp_runner.py --case legocar --conf confs/synthetic.conf --mode validate_mesh --is_continue
    ```
    This will export a UV-unwraped OBJ file along with PBR texture maps from last checkpoint under `exp/legocar/masked/meshes/` (this might take a few minutes.
3. Validate novel view rendering. A `dataset_val` must be provided in config.
    ```bash
    python exp_runner.py --case legocar --conf confs/synthetic.conf --mode validate_image --is_continue
    ```
    Results will be saved to `exp/legocar/masked/novel_view/`.

## Results (rendered in blender)



https://user-images.githubusercontent.com/57708879/232410072-43d74df8-9438-4fc8-b302-0cd2c7f659ed.mp4

## Acknowledgement

This repo is heavily built upon [NeuS](https://github.com/Totoro97/NeuS). We would like to thank the authors for opening source. 
Special thanks goes to @wei-mao-2019, a friend and fellow researcher who agreed to appear in our dataset.

## BibTex
```latex
@article{cheng2023wildlight,
  title={WildLight: In-the-wild Inverse Rendering with a Flashlight},
  author={Cheng, Ziang and Li, Junxuan and Li, Hongdong},
  journal={arXiv preprint arXiv:2303.14190},
  year={2023}
}
```
