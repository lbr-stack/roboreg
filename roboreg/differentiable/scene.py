from typing import Dict

import torch

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
        robot: Robot,  # TODO: ideally this would be any combinations of TorchMeshContainers, i.e. Dict[str, TorchMeshContainer] (future work: RobotScene -> Scene, robot -> objects)
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
