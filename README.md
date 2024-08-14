# Roboreg
[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://github.com/lbr-stack/roboreg?tab=License-1-ov-file#readme)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Unified eye-in-hand / eye-to-hand calibration from RGB-D images using robot mesh as calibration target.

<body>
    <table>
    <caption>Mesh (purple) and Point Cloud (turqoise).</caption>
        <tr>
            <th align="left" width="50%">Unregistered</th>
            <th align="left" width="50%">Registered</th>
        </tr>
        <tr>
            <td align="center"><img src="doc/img//hydra_robust_icp_unregistered.png" alt="Unregistered Mesh and Point Cloud"></td>
            <td align="center"><img src="doc/img//hydra_robust_icp_registered.png" alt="Registered Mesh and Point Cloud"></td>
        </tr>
    </table>
</body>

## Installation
```shell
pip3 install git+https://github.com/lbr-stack/roboreg.git
```

## Command Line Interface
### Segment
This is a required step to generate robot masks (also support SAM 2: `rr-sam2`).

```shell
rr-sam \
    --path <path_to_images> \
    --pattern "*_image_*.png" \
    --checkpoint <full_path_to_checkpoint>/*.pth
```

### Hydra Robust ICP
This registration only works for registered point clouds!

```shell
rr-hydra \
    --path test/data/lbr_med7/zed2i/high_res \
    --mask-pattern mask_*.png \
    --xyz-pattern xyz_*.npy \
    --joint-states-pattern joint_states_*.npy \
    --number-of-points 5000 \
    --convex-hull \
    --output-file HT_hydra_robust.npy
```

### Stereo Differentiable Rendering
This rendering refinement requires a good initial estimate, as e.g. obtained from [Hydra Robust ICP](#hydra-robust-icp).

```shell
rr-stereo-dr \
    --device cuda \
    --optimizer SGD \
    --lr 0.0001 \
    --epochs 100 \
    --display-progress \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name link_0 \
    --end-link-name link_7 \
    --left-camera-info-file test/data/lbr_med7/zed2i/stereo_data/left_camera_info.yaml \
    --right-camera-info-file test/data/lbr_med7/zed2i/stereo_data/right_camera_info.yaml \
    --left-extrinsics-file test/data/lbr_med7/zed2i/stereo_data/HT_hydra_robust.npy \
    --right-extrinsics-file test/data/lbr_med7/zed2i/stereo_data/HT_right_to_left.npy \
    --path test/data/lbr_med7/zed2i/stereo_data \
    --left-image-pattern left_img_*.png \
    --right-image-pattern right_img_*.png \
    --joint-states-pattern joint_state_*.npy \
    --left-mask-pattern left_mask_*.png \
    --right-mask-pattern right_mask_*.png \
    --left-output-file HT_left_dr.npy \
    --right-output-file HT_right_dr.npy
```

### Render Results
Generate renders using the obtained extrinsics:

```shell
rr-render \
    --device cuda \
    --batch-size 1 \
    --num-workers 0 \
    --ros-package lbr_description \
    --xacro-path urdf/med7/med7.xacro \
    --root-link-name link_0 \
    --end-link-name link_7 \
    --camera-info-file test/data/lbr_med7/zed2i/stereo_data/left_camera_info.yaml \
    --extrinsics-file test/data/lbr_med7/zed2i/stereo_data/HT_left_dr.npy \
    --images-path test/data/lbr_med7/zed2i/stereo_data \
    --joint-states-path test/data/lbr_med7/zed2i/stereo_data \
    --image-pattern left_img_*.png \
    --joint-states-pattern joint_state_*.npy \
    --output-path test/data/lbr_med7/zed2i/stereo_data
```

## Acknowledgements
### Organizations and Grants
We would further like to acknowledge following supporters:

| Logo | Notes |
|:--:|:---|
| <img src="https://medicalengineering.org.uk/wp-content/themes/aalto-child/_assets/images/medicalengineering-logo.svg" alt="wellcome" width="150" align="left">  | This work was supported by core and project funding from the Wellcome/EPSRC [WT203148/Z/16/Z; NS/A000049/1; WT101957; NS/A000027/1]. |
| <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/1920px-Flag_of_Europe.svg.png" alt="eu_flag" width="150" align="left"> | This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101016985 (FAROS project). |
| <img src="https://rvim.online/author/avatar_hu8970a6942005977dc117387facf47a75_62303_270x270_fill_lanczos_center_2.png" alt="RViMLab" width="150" align="left"> | Built at [RViMLab](https://rvim.online/). |
| <img src="https://avatars.githubusercontent.com/u/75276868?s=200&v=4" alt="King's College London" width="150" align="left"> | Built at [CAI4CAI](https://cai4cai.ml/). |
| <img src="https://upload.wikimedia.org/wikipedia/commons/1/14/King%27s_College_London_logo.svg" alt="King's College London" width="150" align="left"> | Built at [King's College London](https://www.kcl.ac.uk/). |
