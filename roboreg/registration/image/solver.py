import torch

from roboreg.core.robot import Robot
from roboreg.core.scene import RobotScene
from roboreg.core.structs import VirtualCamera
from roboreg.registration.image.config import (
    CameraSwarmRegistrationConfig,
    DiffRenderingRegistrationConfig,
)
from roboreg.registration.image.objectives import RenderingObjective
from roboreg.registration.image.request import ImageRegistrationRequest
from roboreg.registration.result import RegistrationResult
from roboreg.core.rendering import NVDiffRastRenderer
from roboreg.util.transform import rescale_intrinsics


class DiffRenderingRegistration:
    def __init__(
        self,
        config: DiffRenderingRegistrationConfig,
        objective: RenderingObjective,
        device: torch.device | str = "cuda",
    ) -> None:
        self._config = config
        self._objective = objective
        self._device = torch.device(device)

    def __call__(
        self,
        request: ImageRegistrationRequest,
    ) -> RegistrationResult:
        robot_scene = self._prepare_robot_scene(request)
        pass

    def _prepare_robot_scene(
        self,
        request: ImageRegistrationRequest,
    ) -> RobotScene:
        # prepare cameras
        cameras: dict[str, VirtualCamera] = {}
        for camera_name, camera_data in request.cameras.items():
            # handle intrinsic scaling between hardware resolution and rendering resolution
            camera_observations = request.observations.cameras[camera_name]
            native_resolution = camera_observations.shape
            target_resolution = (
                self._config.camera.target_resolution or native_resolution
            )
            intrinsics = rescale_intrinsics(
                intrinsics=camera_data.intrinsics,
                source_resolution=native_resolution,
                target_resolution=target_resolution,
            )
            # prepare virtual camera
            cameras[camera_name] = VirtualCamera(
                resolution=target_resolution,
                intrinsics=intrinsics,
                extrinsics=camera_data.extrinsics,
                z_min=self._config.camera.z_min,
                z_max=self._config.camera.z_max,
                device=self._device,
            )
        # prepare robot
        robot = Robot.from_robot_data(
            robot_data=request.robot_data,
            batch_size=len(request.observations.joint_states),
            device=self._device,
        )
        return RobotScene(
            cameras=cameras,
            robot=robot,
            renderer=NVDiffRastRenderer(device=self._device),
        )


class CameraSwarmRegistration:
    def __init__(
        self,
        config: CameraSwarmRegistrationConfig,
        objective: RenderingObjective,
        device: torch.device | str = "cuda",
    ) -> None:
        self._config = config
        self._objective = objective
        self._device = torch.device(device)

    def __call__(self, request: ImageRegistrationRequest) -> RegistrationResult:
        pass
