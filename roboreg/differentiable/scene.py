import torch

from .kinematics import TorchKinematics
from .rendering import NVDiffRastRenderer
from .structs import TorchRobotMesh


class RobotScene:
    r"""Differentiable robot scene."""

    _mesh: TorchRobotMesh
    _kinematics: TorchKinematics
    _renderer: NVDiffRastRenderer

    def __init__(
        self,
        mesh: TorchRobotMesh,
        kinematics: TorchKinematics,
        renderer: NVDiffRastRenderer,
    ) -> None:
        self._mesh = mesh
        self._kinematics = kinematics
        self._renderer = renderer

    def view(self, camera_pose: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def mesh(self) -> TorchRobotMesh:
        return self._mesh

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
