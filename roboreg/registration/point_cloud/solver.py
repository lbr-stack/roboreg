from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

from roboreg.core.robot import Robot
from roboreg.registration.result import RegistrationResult
from roboreg.util.mask import mask_extract_extended_boundary
from roboreg.util.points import (
    clean_xyz,
    compute_vertex_normals,
    from_homogeneous,
    to_homogeneous,
)
from roboreg.util.transform import depth_to_xyz, generate_ht_optical

from .config import HydraConfig, HydraICPConfig, HydraRobustICPConfig
from .hydra import centroid_alignment, point_to_plane_robust_icp, point_to_point_icp
from .request import HydraRequest


@dataclass(frozen=True)
class HydraProblem:
    observed_vertices: List[torch.Tensor]
    reference_vertices: List[torch.Tensor]
    reference_normals: Optional[List[torch.Tensor]] = None


HydraCallback = Callable[
    ["HydraProblem", RegistrationResult],
    None,
]


def _prepare_hydra_problem(
    request: HydraRequest,
    config: HydraConfig,
    device: torch.device,
    compute_normals: bool = True,
) -> HydraProblem:
    # 1) construct robot on request
    robot = Robot.from_robot_data(
        robot_data=request.robot_data,
        batch_size=len(request.observations.joint_states),
        device=device,
    )

    # 2) to tensor
    joint_states = torch.tensor(
        np.stack(request.observations.joint_states), dtype=torch.float32, device=device
    )
    intrinsics = torch.tensor(request.intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(
        np.stack(request.observations.depths), dtype=torch.float32, device=device
    )

    # 3) perform forward kinematics
    robot.configure(joint_states)

    # 4) process depths
    xyzs = depth_to_xyz(
        depth=depths,
        intrinsics=intrinsics,
        z_min=config.depth_to_point_cloud.z_min,
        z_max=config.depth_to_point_cloud.z_max,
        conversion_factor=config.depth_to_point_cloud.depth_conversion_factor,
    )
    height, width = request.observations.shape
    xyzs = xyzs.view(-1, height * width, 3)  # flatten BxHxWx3 -> Bx(H*W)x3
    xyzs = to_homogeneous(xyzs)
    ht_optical = generate_ht_optical(xyzs.shape[0], dtype=torch.float32, device=device)
    xyzs = torch.matmul(xyzs, ht_optical.transpose(-1, -2))
    xyzs = from_homogeneous(xyzs)
    xyzs = xyzs.view(-1, height, width, 3)
    xyzs = [xyz.squeeze() for xyz in xyzs.cpu().numpy()]

    # 5) clean observed vertices and turn into tensor
    observed_vertices = [
        torch.tensor(
            clean_xyz(
                xyz=xyz,
                mask=(
                    mask_extract_extended_boundary(
                        mask,
                        dilation_kernel=np.ones(
                            [
                                config.depth_to_point_cloud.dilation_kernel_size,
                                config.depth_to_point_cloud.dilation_kernel_size,
                            ]
                        ),
                        erosion_kernel=np.ones(
                            [
                                config.depth_to_point_cloud.erosion_kernel_size,
                                config.depth_to_point_cloud.erosion_kernel_size,
                            ]
                        ),
                    )
                    if config.depth_to_point_cloud.use_mask_boundary
                    else mask
                ),
            ),
            dtype=torch.float32,
            device=device,
        )
        for xyz, mask in zip(xyzs, request.observations.masks)
    ]

    # mesh vertices to list
    batch_size = len(request.observations.joint_states)

    mesh_vertices = from_homogeneous(robot.configured_vertices)
    mesh_vertices = [mesh_vertices[i].contiguous() for i in range(batch_size)]

    mesh_normals: list[torch.Tensor] | None = None

    if compute_normals:
        mesh_normals = [
            compute_vertex_normals(
                vertices=mesh_vertices[i],
                faces=robot.mesh_container.faces,
            )
            for i in range(batch_size)
        ]

    # sample N points per mesh
    for i in range(batch_size):
        n_points = min(
            config.reference_points_per_mesh,
            mesh_vertices[i].shape[0],
        )

        idx = torch.randperm(
            mesh_vertices[i].shape[0],
            device=mesh_vertices[i].device,
        )[:n_points]

        mesh_vertices[i] = mesh_vertices[i][idx]

        if mesh_normals is not None:
            mesh_normals[i] = mesh_normals[i][idx]

    return HydraProblem(
        observed_vertices=observed_vertices,
        reference_vertices=mesh_vertices,
        reference_normals=mesh_normals,
    )


class HydraICP:
    def __init__(
        self,
        config: HydraICPConfig | None = None,
        device: torch.device | str = "cuda",
        callback: HydraCallback | None = None,
    ) -> None:
        self._config = config or HydraICPConfig()
        self._device = torch.device(device)
        self._callback = callback

    def __call__(self, request: HydraRequest) -> RegistrationResult:
        hydra_problem = _prepare_hydra_problem(
            request=request,
            config=self._config.hydra,
            device=self._device,
            compute_normals=False,
        )
        HT_init = centroid_alignment(
            hydra_problem.observed_vertices, hydra_problem.reference_vertices
        )
        result = point_to_point_icp(
            HT_init,
            hydra_problem.observed_vertices,
            hydra_problem.reference_vertices,
            max_correspondence_distance=self._config.hydra.max_correspondence_distance,
            max_iterations=self._config.max_iterations,
            rmse_change_tolerance=self._config.hydra.rmse_change_tolerance,
        )
        if self._callback is not None:
            self._callback(hydra_problem, result)
        return result


class HydraRobustICP:
    def __init__(
        self,
        config: HydraRobustICPConfig | None = None,
        device: torch.device | str = "cuda",
        callback: HydraCallback | None = None,
    ) -> None:
        self._config = config or HydraRobustICPConfig()
        self._device = torch.device(device)
        self._callback = callback

    def __call__(self, request: HydraRequest) -> RegistrationResult:
        hydra_problem = _prepare_hydra_problem(
            request=request,
            config=self._config.hydra,
            device=self._device,
            compute_normals=True,
        )
        HT_init = centroid_alignment(
            hydra_problem.observed_vertices, hydra_problem.reference_vertices
        )
        result = point_to_plane_robust_icp(
            HT_init,
            hydra_problem.observed_vertices,
            hydra_problem.reference_vertices,
            hydra_problem.reference_normals,
            max_correspondence_distance=self._config.hydra.max_correspondence_distance,
            max_outer_iterations=self._config.max_outer_iterations,
            max_inner_iterations=self._config.max_inner_iterations,
            rmse_change_tolerance=self._config.hydra.rmse_change_tolerance,
        )
        if self._callback is not None:
            self._callback(hydra_problem, result)
        return result
