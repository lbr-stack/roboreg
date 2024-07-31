import torch

from .kinematics import TorchKinematics
from .rendering import NVDiffRastRenderer
from .structs import TorchMeshContainer


class RobotScene:
    r"""Differentiable robot scene."""

    _meshes: TorchMeshContainer
    _kinematics: TorchKinematics
    _renderer: NVDiffRastRenderer

    def __init__(
        self,
        meshes: TorchMeshContainer,
        kinematics: TorchKinematics,
        renderer: NVDiffRastRenderer,
    ) -> None:
        self._meshes = meshes
        self._kinematics = kinematics
        self._renderer = renderer

    def view(self, camera_pose: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self._renderer.constant_color(
            self._meshes.vertices, self._meshes.faces, resolution=[256, 256]
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


class RobotSceneModule(torch.nn.Module):
    f"""Differentiable robot scene as module."""

    def __init__(self, urdf: str) -> None:
        super().__init__()
        self._robot_scene = RobotScene()

    # def forward(self, intrinsics: FloatTensor, pose: FloatTensor, q: FloatTensor) -> FloatTensor:
    #     pass
