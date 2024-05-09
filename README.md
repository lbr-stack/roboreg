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

## Environment Setup
```shell
conda install mamba -c conda-forge
conda create -n roboreg_1.0.0
mamba env update -f env.yaml
```

## Segment
```shell
python3 scripts/generate_masks.py \
    --path <path_to_images> \
    --pattern "*_img_*.png" \
    --sam_checkpoint <full_path_to_checkpoint>/*.pth
```

## Hydra Robust ICP
Note, this registration only works for registered point clouds!
```shell
python3 scripts/register_hydra_robust_icp.py \
    --path <path_to_data>
```

## Render Results
```shell
python3 scripts/render_robot.py \
    --path <path_to_data>
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
