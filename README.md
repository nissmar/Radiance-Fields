# Radiance-Fields

NPM3D MVA project, coded from (almost) scratch, inspired by [Space carving](https://www.cs.toronto.edu/~kyros/pubs/00.ijcv.carve.pdf) and [Plenoxels](https://github.com/sxyu/svox2). Requires the [NeRF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

## Space carving, plenoxels and spherical harmonics

Composed of two steps: space carving and opaque voxels optimization. Running `main_carve_sph.py` takes approximately 4 minutes on a single GPU.

<img src="exports/spherical_harmonics/lego.gif" alt="gif loading" width="250"/> <img src="exports/spherical_harmonics/hotdog.gif" alt="gif loading" width="250"/> <img src="exports/spherical_harmonics/mic.gif" alt="gif loading" width="250"/>

<img src="exports/spherical_harmonics/chair.gif" alt="gif loading" width="250"/> <img src="exports/spherical_harmonics/materials.gif" alt="gif loading" width="250"/> <img src="exports/spherical_harmonics/ship.gif" alt="gif loading" width="250"/>

Spherical harmonics of degree 9 are used to render the shading of each voxel:

<img src="exports/spherical_harmonics/harmonics_0.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_1.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_2.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_3.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_4.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_5.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_6.png" alt="img" width="50"/> <img src="exports/spherical_harmonics/harmonics_7.png" alt="img" width="50"/>
<img src="exports/spherical_harmonics/harmonics_8.png" alt="img" width="50"/>


## Space carving and point cloud extraction

Running `main_carve.py` takes approximately 1 minute on a single GPU.

<img src="Report/figs/carve/chair.png" alt="img" width="200"/> <img src="Report/figs/carve/chairc.png" alt="img" width="200"/> <img src="Report/figs/carve/ficus.png" alt="img" width="200"/> <img src="Report/figs/carve/ficusc.png" alt="img" width="200"/> 

## Our implementation of plenoxels

<img src="exports/gifs/movies_chair.gif" alt="gif loading" width="250"/> <img src="exports/gifs/movies_lego.gif" alt="gif loading" width="250"/> <img src="exports/gifs/movies_ficus.gif" alt="gif loading" width="250"/>

<img src="exports/gifs/movies_mic.gif" alt="gif loading" width="250"/>  <img src="exports/gifs/movies_drums.gif" alt="gif loading" width="250"/> <img src="exports/gifs/movies_hotdog.gif" alt="gif loading" width="250"/>

### Training
Run `main.py` to start the training. It takes approximately 30 minutes on a single GPU and saves the final 128x128x128 voxel grid in `./saved_grids/`.

<img src="exports/movies_training.gif" alt="gif loading" width="300"/>
