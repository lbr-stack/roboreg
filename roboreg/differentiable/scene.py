from typing import Dict

import torch

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

    _meshes: TorchMeshContainer
    _kinematics: TorchKinematics
    _renderer: NVDiffRastRenderer
    _cameras: Dict[str, VirtualCamera]

    def __init__(
        self,
        meshes: TorchMeshContainer,
        kinematics: TorchKinematics,
        renderer: NVDiffRastRenderer,
        cameras: Dict[str, VirtualCamera],
    ) -> None:
        self._meshes = meshes
        self._kinematics = kinematics
        self._renderer = renderer
        self._cameras = cameras

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
        ht_lookup = self._kinematics.mesh_forward_kinematics(q)
        for link_name, ht in ht_lookup.items():
            self._meshes.transform_mesh(ht, link_name)

    def observe_from(self, camera_name: str) -> torch.Tensor:
        observed_vertices = torch.matmul(
            self._meshes.vertices,
            torch.linalg.inv(
                self._cameras[camera_name].extrinsics
                @ self._cameras[camera_name].ht_optical
            ).T
            @ self._cameras[camera_name].perspective_projection.T,
        )
        return self._renderer.constant_color(
            observed_vertices,
            self._meshes.faces,
            self._cameras[camera_name].resolution,
        )

    def observe(self) -> Dict[str, torch.Tensor]:
        return {
            camera_name: self.observe_from(camera_name)
            for camera_name in self._cameras.keys()
        }

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
