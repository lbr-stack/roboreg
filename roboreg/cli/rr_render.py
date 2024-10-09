import argparse
import os
import pathlib

import cv2
import numpy as np
import torch
from rich import progress
from torch.utils.data import DataLoader

from roboreg import differentiable as rrd
from roboreg.io import MonocularDataset
from roboreg.util import overlay_mask


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for rendering. For batch_size > 1, the last batch may be dropped.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--ros-package",
        type=str,
        default="lbr_description",
        help="Package where the URDF is located.",
    )
    parser.add_argument(
        "--xacro-path",
        type=str,
        default="urdf/med7/med7.xacro",
        help="Path to the xacro file, relative to --ros-package.",
    )
    parser.add_argument(
        "--root-link-name", type=str, default="link_0", help="Root link name."
    )
    parser.add_argument(
        "--end-link-name", type=str, default="link_7", help="End link name."
    )
    parser.add_argument(
        "--camera-info-file",
        type=str,
        required=True,
        help="Path to the camera parameters, <path_to>/camera_info.yaml.",
    )
    parser.add_argument(
        "--extrinsics-file",
        type=str,
        required=True,
        help="Homogeneous transform from base to camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument(
        "--images-path", type=str, required=True, help="Path to the images."
    )
    parser.add_argument(
        "--joint-states-path", type=str, required=True, help="Path to the joint states."
    )
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="image_*.png",
        help="Image file pattern.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    scene = rrd.robot_scene_factory(
        device=args.device,
        batch_size=args.batch_size,
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        camera_info_files={"camera": args.camera_info_file},
        extrinsics_files={"camera": args.extrinsics_file},
    )
    dataset = MonocularDataset(
        images_path=args.images_path,
        image_pattern=args.image_pattern,
        joint_states_path=args.joint_states_path,
        joint_states_pattern=args.joint_states_pattern,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    output_path = pathlib.Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for images, joint_states, image_files in progress.track(
        dataloader, description="Rendering..."
    ):
        # pre-process
        joint_states = joint_states.to(dtype=torch.float32, device=args.device)

        # configure robot
        scene.configure_robot_joint_states(joint_states)

        # render
        renders = scene.observe_from(list(scene.cameras.keys())[0])

        # save
        images = images.numpy()
        renders = (renders * 255.0).squeeze(-1).cpu().numpy().astype(np.uint8)
        for render, image, image_file in zip(renders, images, image_files):
            prefix = image_file.split(".")[0]
            suffix = image_file.split(".")[1]
            cv2.imwrite(
                os.path.join(str(output_path.absolute()), f"overlay_{prefix}.{suffix}"),
                overlay_mask(image, render, "b", scale=1.0),
            )


if __name__ == "__main__":
    main()
