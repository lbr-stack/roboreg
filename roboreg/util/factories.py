from typing import Dict, Optional

import numpy as np
import rich
import torch

from roboreg.differentiable import NVDiffRastRenderer, Robot, RobotScene, VirtualCamera
from roboreg.io import URDFParser, parse_camera_info


def create_virtual_camera(
    camera_info_file: str,
    extrinsics_file: Optional[str] = None,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
) -> VirtualCamera:
    height, width, intrinsics = parse_camera_info(camera_info_file=camera_info_file)
    extrinsics = None
    if extrinsics_file is not None:
        extrinsics = np.load(extrinsics_file)
    camera = VirtualCamera(
        resolution=[height, width],
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        device=device,
    )
    return camera


def create_robot_scene(
    batch_size: int,
    ros_package: str,
    xacro_path: str,
    root_link_name: str,
    end_link_name: str,
    cameras: Dict[str, VirtualCamera],
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    visual: bool = False,
    target_reduction: float = 0.0,
) -> RobotScene:
    # create URDF parser
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    if root_link_name == "":
        root_link_name = urdf_parser.link_names_with_meshes(visual=visual)[0]
        rich.print(
            f"Root link name not provided. Using the first link with mesh: '{root_link_name}'."
        )
    if end_link_name == "":
        end_link_name = urdf_parser.link_names_with_meshes(visual=visual)[-1]
        rich.print(
            f"End link name not provided. Using the last link with mesh: '{end_link_name}'."
        )

    # instantiate robot
    robot = Robot(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        visual=visual,
        batch_size=batch_size,
        device=device,
        target_reduction=target_reduction,
    )

    # instantiate renderer
    renderer = NVDiffRastRenderer(device=device)

    # instantiate and return scene
    return RobotScene(
        cameras=cameras,
        robot=robot,
        renderer=renderer,
    )
