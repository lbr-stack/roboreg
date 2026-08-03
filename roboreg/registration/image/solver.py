from typing import Iterable

import numpy as np
import pytorch_kinematics as pk
import rich
import rich.progress
import torch

from roboreg.core.rendering import NVDiffRastRenderer
from roboreg.core.robot import Robot
from roboreg.core.scene import RobotScene
from roboreg.core.structs import VirtualCamera
from roboreg.registration.image.config import (
    CameraSwarmRegistrationConfig,
    DiffRenderingRegistrationConfig,
)
from roboreg.registration.image.objectives import RenderingObjective
from roboreg.registration.image.request import (
    ImageObservations,
    ImageRegistrationRequest,
)
from roboreg.registration.result import RegistrationResult, TerminationReason
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

        # TODO: add support for callbacks

    def __call__(
        self,
        request: ImageRegistrationRequest,
    ) -> RegistrationResult:
        # PREPARE TARGETS (targets, joint states, extrinsics)
        # prepare problem: enable gradient tracking and instantiate optimizer
        extrinsics_inv = torch.linalg.inv(
            torch.tensor(
                request.initial_extrinsics, dtype=torch.float32, device=self._device
            )
        )
        extrinsics_9d_inv = pk.matrix44_to_se3_9d(extrinsics_inv)
        extrinsics_9d_inv.requires_grad = True

        if not extrinsics_9d_inv.requires_grad:
            raise ValueError("Extrinsics require gradients.")
        if not torch.is_grad_enabled():
            raise ValueError("Gradients must be enabled.")

        joint_states, preprocessed_targets = self._prepare_image_observations(
            request.observations
        )
        # PREPARE TARGETS END (targets, joint states, extrinsics)
        robot_scene = self._create_robot_scene(request)
        optimizer = self._create_optimizer(params=[extrinsics_9d_inv])
        scheduler = self._create_reduce_on_plateau_scheduler(optimizer)
        # run optimization loop
        result = self._optimize(
            robot_scene=robot_scene,
            joint_states=joint_states,
            preprocessed_targets=preprocessed_targets,
            extrinsics_9d_inv=extrinsics_9d_inv,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return result

    def _prepare_image_observations(
        self,
        observations: ImageObservations,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        joint_states = torch.as_tensor(
            np.stack(observations.joint_states, axis=0),
            dtype=torch.float32,
            device=self._device,
        )
        preprocessed_targets: dict[str, torch.Tensor] = {}
        for camera_name, camera_observations in observations.cameras.items():
            targets = (
                torch.as_tensor(
                    np.stack(camera_observations.targets, axis=0),
                    dtype=torch.float32,
                    device=self._device,
                )
                / 255.0
            )
            preprocessed_targets[camera_name] = self._objective.preprocess_targets(
                targets
            )
        return joint_states, preprocessed_targets

    def _create_robot_scene(
        self,
        request: ImageRegistrationRequest,
    ) -> RobotScene:
        # prepare cameras
        cameras: dict[str, VirtualCamera] = {}
        for camera_name, camera_data in request.cameras.items():
            # handle intrinsic scaling: in case of
            # hardware resolution and rendering resolution mismatch
            camera_observations = request.observations.cameras[camera_name]
            native_resolution = camera_observations.shape
            target_resolution = (
                self._config.camera.target_resolution or native_resolution
            )
            intrinsics = rescale_intrinsics(
                intrinsics=camera_data.intrinsics,
                current_resolution=native_resolution,
                target_resolution=target_resolution,
            )
            # prepare virtual camera
            cameras[camera_name] = VirtualCamera(
                resolution=target_resolution,
                intrinsics=intrinsics,
                extrinsics=camera_data.reference_to_camera,
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

    def _create_optimizer(
        self, params: Iterable[torch.Tensor]
    ) -> torch.optim.Optimizer:
        return getattr(torch.optim, self._config.optimizer)(params, lr=self._config.lr)

    def _create_reduce_on_plateau_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self._config.plateau_scheduler.mode,
            factor=self._config.plateau_scheduler.factor,
            patience=self._config.plateau_scheduler.patience,
            threshold=self._config.plateau_scheduler.threshold,
        )

    def _optimize(
        self,
        robot_scene: RobotScene,
        joint_states: torch.Tensor,
        preprocessed_targets: dict[str, torch.Tensor],
        extrinsics_9d_inv: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) -> RegistrationResult:

        best_extrinsics_inv: torch.Tensor | None = None
        best_loss = float("inf")

        # TODO: add convergence check...

        for iteration in rich.progress.track(
            range(1, self._config.convergence.max_iterations + 1), "Optimizing..."
        ):

            extrinsics_inv = pk.se3_9d_to_matrix44(
                extrinsics_9d_inv
            )  ### that's the parameter here....
            robot_scene.robot.configure(joint_states, extrinsics_inv)

            # per camera render and loss
            camera_losses: list[torch.Tensor] = []
            for camera_name in robot_scene.cameras:
                render = robot_scene.observe_from(camera_name)
                camera_loss = self._objective(
                    preprocessed_targets=preprocessed_targets[camera_name],
                    renders=render,
                )
                camera_losses.append(camera_loss)
            loss = torch.stack(camera_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(metrics=loss)

            rich.print(
                f"Step [{iteration} / {self._config.convergence.max_iterations}], loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_extrinsics_inv = extrinsics_inv.detach().clone()

        return RegistrationResult(
            extrinsics=torch.linalg.inv(best_extrinsics_inv),
            iterations=iteration,
            termination_reason=TerminationReason.MAX_ITERATIONS,
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
