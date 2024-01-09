# Roboreg
Robot registration from RGB-D images. Allows for target-less camera to robot calibrations.

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

## Hydra Projection


## Baselines

## Environment
```shell
conda install mamba -c conda-forge
conda create -n roboreg_0.1.1
mamba env update -f env.yaml
```
