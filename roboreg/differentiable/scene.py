from typing import Dict

import numpy as np
import rich
import torch

from roboreg.io import URDFParser, parse_camera_info

from .rendering import NVDiffRastRenderer
from .robot import Robot
from .structs import VirtualCamera


class RobotScene:
    __slots__ = [
        "_cameras",
        "_robot",
        "_renderer",
    ]

    def __init__(
        self,
        cameras: Dict[str, VirtualCamera],
        robot: Robot,  # TODO: ideally this is any combinations of TorchMeshContainers, i.e. Dict[str, TorchMeshContainer] (future work)
        renderer: NVDiffRastRenderer,
    ) -> None:
        self._cameras = cameras
        self._robot = robot
        self._renderer = renderer
        self._verify_devices()

    def _verify_devices(self) -> None:
        for camera_name in self._cameras.keys():
            if not all(
                [
                    self._cameras[camera_name].device == self._robot.device,
                    self._robot.device == self._renderer.device,
                ]
            ):
                raise ValueError(
                    "All devices must be the same. Got:\n"
                    f"Camera '{camera_name}' on: {self._cameras[camera_name].device}\n"
                    f"Robot on: {self._robot.device}\n"
                    f"Renderer on: {self._renderer.device}"
                )

    def observe_from(
        self, camera_name: str, reference_transform: torch.FloatTensor = None
    ) -> torch.Tensor:
        if reference_transform is None:
            reference_transform = torch.eye(
                4,
                dtype=self._cameras[camera_name].extrinsics.dtype,
                device=self._cameras[camera_name].extrinsics.device,
            )
        observed_vertices = torch.matmul(
            self._robot.configured_vertices,
            torch.matmul(
                torch.linalg.inv(
                    torch.matmul(
                        reference_transform,
                        torch.matmul(
                            self._cameras[camera_name].extrinsics,
                            self._cameras[camera_name].ht_optical,
                        ),
                    )
                ).transpose(-1, -2),
                self._cameras[camera_name].perspective_projection.transpose(-1, -2),
            ),
        )
        return self._renderer.constant_color(
            observed_vertices,
            self._robot.faces,
            self._cameras[camera_name].resolution,
        )

    @property
    def cameras(self) -> Dict[str, VirtualCamera]:
        return self._cameras

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def renderer(self) -> NVDiffRastRenderer:
        return self._renderer


def robot_scene_factory(
    device: str,
    batch_size: int,
    ros_package: str,
    xacro_path: str,
    root_link_name: str,
    end_link_name: str,
    camera_info_files: Dict[str, str],
    extrinsics_files: Dict[str, str],
    visual: bool = False,
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
        target_reduction=0.0,
    )

    # instantiate renderer
    renderer = NVDiffRastRenderer(device=device)

    # instantiate camera
    if list(camera_info_files.keys()) != list(extrinsics_files.keys()):
        raise ValueError(
            "Camera names for camera_info_files and extrinsics_files do not match."
        )

    cameras = {}
    for camera_name in camera_info_files.keys():
        height, width, intrinsics = parse_camera_info(
            camera_info_file=camera_info_files[camera_name]
        )
        extrinsics = np.load(extrinsics_files[camera_name])
        cameras[camera_name] = VirtualCamera(
            resolution=[height, width],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            device=device,
        )

    # instantiate and return scene
    return RobotScene(
        cameras=cameras,
        robot=robot,
        renderer=renderer,
    )
