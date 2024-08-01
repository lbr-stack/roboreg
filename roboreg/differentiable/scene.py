from typing import Dict

import torch

from .kinematics import TorchKinematics
from .rendering import NVDiffRastRenderer
from .structs import Camera, TorchMeshContainer


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
    _cameras: Dict[str, Camera]

    def __init__(
        self,
        meshes: TorchMeshContainer,
        kinematics: TorchKinematics,
        renderer: NVDiffRastRenderer,
        cameras: Dict[str, Camera],
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

    def configure_camera(self, camera_name: str, ht: torch.FloatTensor) -> None:
        pass

    def configure_robot(self, q: torch.FloatTensor) -> None:
        pass

    def observe_from(self, camera_name: str) -> torch.Tensor:
        pass

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


class RobotSceneModule(torch.nn.Module):
    f"""Differentiable robot scene as module."""

    def __init__(self, urdf: str) -> None:
        super().__init__()
        self._robot_scene = RobotScene()

    # def forward(self, intrinsics: FloatTensor, pose: FloatTensor, q: FloatTensor) -> FloatTensor:
    #     pass
