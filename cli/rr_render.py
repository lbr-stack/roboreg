import os
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import typer
from rich import progress
from torch.utils.data import DataLoader

from roboreg.core import NVDiffRastRenderer, Robot, RobotScene, VirtualCamera
from roboreg.io import (
    MonocularDataset,
    load_robot_data_from_ros_xacro,
    load_robot_data_from_urdf_file,
)
from roboreg.util import overlay_mask

from .util.validate import validate_urdf_source

app = typer.Typer(add_completion=False)


class OverlayColor(str, Enum):
    RED = "r"
    GREEN = "g"
    BLUE = "b"


@app.command()
def main(
    camera_info_file: Path = typer.Option(
        ..., help="Path to the camera parameters, <path_to>/camera_info.yaml."
    ),
    extrinsics_file: Path = typer.Option(
        ...,
        help="Homogeneous transform from base to camera frame, <path_to>/HT_hydra_robust.npy.",
    ),
    images_path: Path = typer.Option(..., help="Path to the images."),
    joint_states_path: Path = typer.Option(..., help="Path to the joint states."),
    output_path: Path = typer.Option(..., help="Output path."),
    batch_size: int = typer.Option(
        1,
        help="Batch size for rendering. For batch_size > 1, the last batch may be dropped.",
    ),
    num_workers: int = typer.Option(0, help="Number of workers for data loading."),
    urdf_path: Optional[Path] = typer.Option(
        "test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf",
        help="Path to URDF file. Meshes resolved relative to this file. "
        "Mutually exclusive with --ros-package/--xacro-path.",
    ),
    ros_package: Optional[str] = typer.Option(
        None,
        help="ROS package containing robot description. "
        "Requires --xacro-path. Mutually exclusive with --urdf-path.",
    ),
    xacro_path: Optional[str] = typer.Option(
        None,
        help="Path to xacro file relative to --ros-package. "
        "Requires --ros-package. Mutually exclusive with --urdf-path.",
    ),
    root_link_name: str = typer.Option(
        "",
        help="Root link name. If unspecified, the first link with mesh will be used, which may cause errors.",
    ),
    end_link_name: str = typer.Option(
        "",
        help="End link name. If unspecified, the last link with mesh will be used, which may cause errors.",
    ),
    collision_meshes: bool = typer.Option(
        False, help="If set, collision meshes will be used instead of visual meshes."
    ),
    image_pattern: str = typer.Option("image_*.png", help="Image file pattern."),
    joint_states_pattern: str = typer.Option(
        "joint_states_*.npy", help="Joint state file pattern."
    ),
    color: OverlayColor = typer.Option(
        OverlayColor.BLUE, help="Color channel to overlay the render."
    ),
    max_jobs: int = typer.Option(
        2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    ),
) -> None:
    r"""Render robot mesh overlays for a set of images given known extrinsics."""
    validate_urdf_source(urdf_path, ros_package, xacro_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(max_jobs)  # limit number of concurrent jobs
    camera = {
        "camera": VirtualCamera.from_camera_configs(
            camera_info_file=camera_info_file,
            extrinsics_file=extrinsics_file,
            device=device,
        )
    }
    if urdf_path is not None:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=urdf_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision_meshes,
        )
    else:
        robot_data = load_robot_data_from_ros_xacro(
            ros_package=ros_package,
            xacro_path=xacro_path,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision_meshes,
        )
    robot = Robot.from_robot_data(
        robot_data=robot_data, batch_size=batch_size, device=device
    )
    scene = RobotScene(
        cameras=camera,
        robot=robot,
        renderer=NVDiffRastRenderer(device=device),
    )
    dataset = MonocularDataset(
        images_path=images_path,
        image_pattern=image_pattern,
        joint_states_path=joint_states_path,
        joint_states_pattern=joint_states_pattern,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    if not output_path.exists():
        output_path.mkdir(parents=True)

    for images, joint_states, image_files in progress.track(
        dataloader, description="Rendering..."
    ):
        # pre-process
        joint_states = joint_states.to(dtype=torch.float32, device=device)

        # configure robot
        scene.robot.configure(joint_states)

        # render
        renders = scene.observe_from(list(scene.cameras.keys())[0])

        # save
        images = images.numpy()
        renders = (renders * 255.0).squeeze(-1).cpu().numpy().astype(np.uint8)
        for render, image, image_file in zip(renders, images, image_files):
            image_file = Path(image_file)
            output_file = (
                output_path / f"overlay_render_{image_file.stem + image_file.suffix}"
            )
            cv2.imwrite(
                output_file,
                overlay_mask(image, render, color.value, scale=1.0),
            )


if __name__ == "__main__":
    app()
