from typing import Dict

import numpy as np
import torch

from roboreg.io import URDFParser, parse_camera_info

from .kinematics import TorchKinematics
from .rendering import NVDiffRastRenderer
from .structs import TorchMeshContainer, VirtualCamera


class RobotScene:
    r"""Differentiable robot scene:

    - Contains utility functions to
        - Configure camera pose
        - Configure robot configuration
    - Currently only supports single robot
    - Supports multi-camera, e.g. stereo
    """

    __slots__ = [
        "_meshes",
        "_kinematics",
        "_renderer",
        "_cameras",
        "_ht_zero_lookup",
        "_observed_vertices",
    ]

    def __init__(
        self,
        meshes: TorchMeshContainer,
        kinematics: TorchKinematics,
        renderer: NVDiffRastRenderer,
        cameras: Dict[str, VirtualCamera],
    ) -> None:
        self._meshes = meshes
        self._observed_vertices = self._meshes.vertices.clone()
        self._kinematics = kinematics
        self._renderer = renderer
        self._cameras = cameras
        self._ht_zero_lookup = self._kinematics.mesh_forward_kinematics(
            torch.zeros(
                [self._meshes.batch_size, self._kinematics.chain.n_joints],
                dtype=torch.float32,
                device=self._meshes.device,
            )
        )  # track current transforms

        for link_name in self._ht_zero_lookup.keys():
            self._ht_zero_lookup[link_name] = torch.linalg.inv(
                self._ht_zero_lookup[link_name]
            )

        for camera_name in self._cameras.keys():
            if not all(
                [
                    self._meshes.device == self._kinematics.device,
                    self._kinematics.device == self._renderer.device,
                    self._renderer.device == self._cameras[camera_name].device,
                ]
            ):
                raise ValueError(
                    "All devices must be the same. Got:\n"
                    f"Meshes on: {self._meshes.device}\n"
                    f"Kinematics on: {self._kinematics.device}\n"
                    f"Renderer on: {self._renderer.device}\n"
                    f"Camera '{camera_name}' on: {self._cameras[camera_name].device}"
                )

    def configure_robot_joint_states(self, q: torch.FloatTensor) -> None:
        if self._kinematics.chain.n_joints != q.shape[-1]:
            raise ValueError(
                f"Expected joint configuration of shape {self._kinematics.chain.n_joints}, got {q.shape[-1]}."
            )
        if q.shape[0] != self._meshes.batch_size:
            raise ValueError(
                f"Batch size mismatch. Meshes: {self._meshes.batch_size}, joint states: {q.shape[0]}."
            )
        ht_target_lookup = self._kinematics.mesh_forward_kinematics(q)
        self._observed_vertices = self._meshes.vertices.clone()
        for link_name, ht in ht_target_lookup.items():
            self._observed_vertices[
                :,
                self._meshes.lower_vertex_index_lookup[
                    link_name
                ] : self._meshes.upper_vertex_index_lookup[link_name],
            ] = torch.matmul(
                self._observed_vertices[
                    :,
                    self._meshes.lower_vertex_index_lookup[
                        link_name
                    ] : self._meshes.upper_vertex_index_lookup[link_name],
                ],
                (ht @ self._ht_zero_lookup[link_name]).transpose(-1, -2),
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
            self._observed_vertices,
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
            self._meshes.faces,
            self._cameras[camera_name].resolution,
        )

    @property
    def meshes(self) -> TorchMeshContainer:
        return self._meshes

    @property
    def kinematics(self) -> TorchKinematics:
        return self._kinematics

    @property
    def renderer(self) -> NVDiffRastRenderer:
        return self._renderer

    @property
    def cameras(self) -> Dict[str, VirtualCamera]:
        return self._cameras


def robot_scene_factory(
    device: str,
    batch_size: int,
    ros_package: str,
    xacro_path: str,
    root_link_name: str,
    end_link_name: str,
    camera_info_files: Dict[str, str],
    extrinsics_files: Dict[str, str],
) -> RobotScene:
    # create URDF parser
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)

    # instantiate kinematics
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )

    # instantiate meshes
    meshes = TorchMeshContainer(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name
        ),
        batch_size=batch_size,
        device=device,
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
        meshes=meshes,
        kinematics=kinematics,
        renderer=renderer,
        cameras=cameras,
    )


class RobotSceneModule(torch.nn.Module):
    f"""Differentiable robot scene as module."""

    def __init__(
        self,
        meshes: TorchMeshContainer,
        kinematics: TorchKinematics,
        renderer: NVDiffRastRenderer,
        cameras: Dict[str, VirtualCamera],
    ) -> None:
        super().__init__()
        self._robot_scene = RobotScene(
            meshes=meshes,
            kinematics=kinematics,
            renderer=renderer,
            cameras=cameras,
        )

    def forward(
        self, pose: torch.FloatTensor, q: torch.FloatTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError
