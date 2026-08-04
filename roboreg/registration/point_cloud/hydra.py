from typing import List, Tuple

import torch

from roboreg.registration.result import RegistrationResult, TerminationReason


def kabsch_register(
    observed_vertices: torch.Tensor, reference_vertices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm.
    Computes rotation and translation such that observed_vertices @ R + t = reference_vertices.

    Args:
        observed_vertices (torch.Tensor): Observed vertices of shape (..., M, 3).
        reference_vertices (torch.Tensor): Reference vertices of shape (..., M, 3).

    Returns:
        Tuple[torch.Tensor,torch.Tensor]:
            - Rotation matrix of shape (..., 3, 3).
            - Translation vector of shape (..., 3).
    """
    # compute centroids
    observed_centroid = torch.mean(observed_vertices, dim=-2)
    reference_centroid = torch.mean(reference_vertices, dim=-2)

    # compute centered points
    observed_centered = observed_vertices - observed_centroid
    reference_centered = reference_vertices - reference_centroid

    # compute covariance matrix
    H = reference_centered.transpose(-1, -2) @ observed_centered

    # compute SVD
    U, _, V = torch.svd(H)

    E = torch.eye(3, dtype=U.dtype, device=U.device)
    E[-1, -1] = torch.det(V @ U.transpose(-1, -2))

    # compute rotation
    R = V @ E @ U.transpose(-1, -2)

    # compute translation
    t = reference_centroid - observed_centroid @ R
    return R, t


def correspondence_indices(
    observed_vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    max_correspondence_distance: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""For each point in input, find nearest neighbor index in target.

    Args:
        observed_vertices (torch.Tensor): Observed vertices of shape (M, 3) or (B, M, 3).
        reference_vertices (torch.Tensor): Reference vertices of shape (N, 3) or (B, N, 3).
        max_correspondence_distance (float): Maximum distance between point correspondences.

    Returns:
        Tuple[torch.Tensor,torch.Tensor]:
            - Match-indices of shape (M) or (B, M), where mi is the index of the nearest neighbor in target.
            - Mask of shape (M) or (B, M).
    """
    if observed_vertices.shape[-1] != 3 or reference_vertices.shape[-1] != 3:
        raise ValueError("Input and target must have shape (..., 3).")
    if max_correspondence_distance < 0:
        raise ValueError("Max distance must be positive.")
    distances = torch.cdist(observed_vertices, reference_vertices, p=2)  # (M, N)
    min_distance, match_indices = torch.min(distances, dim=-1)  # (M)
    mask = min_distance < max_correspondence_distance
    return match_indices, mask


def centroid_alignment(
    observed_vertices: List[torch.Tensor],
    reference_vertices: List[torch.Tensor],
) -> torch.Tensor:
    r"""Aligns centroids of observed_vertices and Ys as an initial guess.

    Args:
        observed_vertices (List[torch.Tensor]): List of poinclouds of shape (Mi, 3).
        reference_vertices (List[torch.Tensor]): List of pointclouds of shape (Ni, 3).

    Returns:
        torch.Tensor: Homogeneous transformation of shape (4, 4). HT @ observed_vertices = reference_vertices.
    """
    # for each cloud compute centroid
    observed_centroids = [
        torch.mean(observation, dim=-2) for observation in observed_vertices
    ]
    reference_centroids = [torch.mean(mesh, dim=-2) for mesh in reference_vertices]

    # estimate transform
    R, t = kabsch_register(
        observed_vertices=torch.stack(observed_centroids).unsqueeze(0),
        reference_vertices=torch.stack(reference_centroids).unsqueeze(0),
    )

    HT = torch.eye(4, dtype=R.dtype, device=R.device)
    R = R.squeeze(0)
    t = t.squeeze(0)
    HT[:3, :3] = R.T
    HT[:3, 3] = t
    return HT


def point_to_point_icp(
    HT_init: torch.Tensor,
    observed_vertices: List[torch.Tensor],
    reference_vertices: List[torch.Tensor],
    max_correspondence_distance: float = 0.1,
    max_iterations: int = 100,
    rmse_change_tolerance: float = 1e-6,
) -> RegistrationResult:
    r"""Hydra iterative closest point algorithm.

    Args:
        HT_init: Initial guess. HT_init @ observed_vertices = reference_vertices.
        observed_vertices: List of observed vertices of shape (Mi, 3).
        reference_vertices: List of reference vertices of shape (Ni, 3).
        max_correspondence_distance: Maximum distance between point correspondences.
        max_iterations: Maximum number of iterations.
        rmse_change_tolerance: Minimum change in rmse to continue iterating.

    Returns:
        RegistrationResult: Result with homogeneous transformation of shape (4, 4). HT @ observed_vertices = reference_vertices.
    """
    HT = HT_init
    # registration
    previous_rmse = float("inf")
    for iteration in range(max_iterations):
        observed_correspondences = []
        reference_correspondences = []
        for i in range(len(reference_vertices)):
            # search correspondences
            observations_tf = observed_vertices[i] @ HT[:3, :3].T + HT[:3, 3]
            match_indices, mask = correspondence_indices(
                observations_tf, reference_vertices[i], max_correspondence_distance
            )

            observed_correspondences.append(observed_vertices[i][mask])
            reference_correspondences.append(
                reference_vertices[i][match_indices[mask]].squeeze()
            )

        observed_correspondences = torch.concatenate(
            observed_correspondences
        ).unsqueeze(0)
        reference_correspondences = torch.concatenate(
            reference_correspondences
        ).unsqueeze(0)

        (
            R,
            t,
        ) = kabsch_register(
            observed_correspondences,
            reference_correspondences,
        )
        R = R.squeeze(0)
        t = t.squeeze(0)
        HT[:3, :3] = R.T
        HT[:3, 3] = t

        # compute rmse between observed_correspondences and reference_correspondences
        rmse = torch.sqrt(
            torch.mean(
                torch.sum(
                    torch.pow(
                        reference_correspondences - observed_correspondences,
                        2,
                    ),
                    dim=-1,
                )
            )
        )

        if abs(previous_rmse - rmse.item()) < rmse_change_tolerance:
            return RegistrationResult(
                extrinsics=HT,
                iterations=iteration,
                termination_reason=TerminationReason.CONVERGED,
            )

        previous_rmse = rmse.item()

    return RegistrationResult(
        extrinsics=HT,
        iterations=max_iterations,
        termination_reason=TerminationReason.MAX_ITERATIONS,
    )


def point_to_plane_robust_icp(
    HT_init: torch.Tensor,
    observed_vertices: List[torch.Tensor],
    reference_vertices: List[torch.Tensor],
    reference_normals: List[torch.Tensor],
    max_correspondence_distance: float = 0.1,
    max_outer_iterations: int = 100,
    max_inner_iterations: int = 3,
    rmse_change_tolerance: float = 1e-6,
) -> RegistrationResult:
    r"""Lie-algebra point-to-plane ICP with robust loss, refer to section 1
    https://drive.google.com/file/d/1iIUqKchAbcYzwyS2D6jNI1J6KotReD1h/view?usp=sharing.

    Args:
        HT_init: Initial guess. HT_init @ observed_vertices = reference_vertices.
        observed_vertices: List of observed vertices of shape (Mi, 3).
        reference_vertices: List of reference vertices of shape (Ni, 3).
        reference_normals: List of reference normals of shape (Ni, 3).
        max_correspondence_distance: Maximum distance between point correspondences.
        max_outer_iterations: Maximum number of outer iterations.
        max_inner_iterations: Maximum number of inner iterations.
        rmse_change_tolerance: Minimum change in rmse to continue iterating.

    Returns:
        RegistrationResult: Result with homogeneous transformation of shape (4, 4). HT @ observed_vertices = reference_vertices.
    """
    HT = HT_init  # HT @ observed_vertices = reference_vertices

    observed_cross_mat = []
    for i in range(len(observed_vertices)):
        # build observation cross product matrix, refer eq. 4 (gets created once)
        observed_cross_mat.append(
            torch.stack(
                [
                    torch.zeros_like(observed_vertices[i][:, 0]),
                    -observed_vertices[i][:, 2],
                    observed_vertices[i][:, 1],
                    observed_vertices[i][:, 2],
                    torch.zeros_like(observed_vertices[i][:, 0]),
                    -observed_vertices[i][:, 0],
                    -observed_vertices[i][:, 1],
                    observed_vertices[i][:, 0],
                    torch.zeros_like(observed_vertices[i][:, 0]),
                ],
                dim=-1,
            ).reshape(-1, 3, 3)
        )

    # implementation of algorithm 1
    previous_rmse = float("inf")
    dTh = torch.zeros_like(HT)
    for outer_iteration in range(max_outer_iterations):
        observed_correspondences = []
        observed_cross_mat_correspondences = []
        reference_correspondences = []
        reference_normals_correspondences = []

        for i in range(len(observed_vertices)):
            if len(observed_vertices) != len(reference_vertices):
                raise ValueError("Length of observations and meshes must be the same.")
            # search correspondences
            observed_vertices_tf = observed_vertices[i] @ HT[:3, :3].T + HT[:3, 3]
            match_indices, mask = correspondence_indices(
                observed_vertices_tf, reference_vertices[i], max_correspondence_distance
            )

            observed_correspondences.append(observed_vertices[i][mask])
            observed_cross_mat_correspondences.append(observed_cross_mat[i][mask])
            reference_correspondences.append(
                reference_vertices[i][match_indices[mask].squeeze()]
            )
            reference_normals_correspondences.append(
                reference_normals[i][match_indices[mask].squeeze()]
            )

        observed_correspondences = torch.cat(observed_correspondences)
        observed_cross_mat_correspondences = torch.cat(
            observed_cross_mat_correspondences
        )
        reference_correspondences = torch.cat(reference_correspondences)
        reference_normals_correspondences = torch.cat(reference_normals_correspondences)

        for _ in range(max_inner_iterations):
            # ||A @ dTh - B||^2, refer eq. 14
            Al = reference_normals_correspondences @ HT[:3, :3]  # eq. 18
            Au = -Al.unsqueeze(1) @ observed_cross_mat_correspondences  # eq. 19
            A = torch.cat((Au.squeeze(), Al.squeeze()), dim=-1)
            B = torch.linalg.vecdot(
                reference_normals_correspondences,
                reference_correspondences
                - (observed_correspondences @ HT[:3, :3].T + HT[:3, 3]),
            )
            # weight associated with Huber loss
            kappa = (
                1.345 * torch.median(torch.abs(B - torch.median(B))) / 0.6745
            )  # eq. 26
            W = torch.where(
                torch.abs(B) < kappa,
                torch.ones_like(B),
                torch.full_like(B, kappa) / torch.abs(B),
            )

            dTh_vec, resid, rank, singvals = torch.linalg.lstsq(W[:, None] * A, W * B)
            dTh[0, 1] = -dTh_vec[2]
            dTh[0, 2] = dTh_vec[1]
            dTh[1, 0] = dTh_vec[2]
            dTh[1, 2] = -dTh_vec[0]
            dTh[2, 0] = -dTh_vec[1]
            dTh[2, 1] = dTh_vec[0]

            dTh[0, 3] = dTh_vec[3]
            dTh[1, 3] = dTh_vec[4]
            dTh[2, 3] = dTh_vec[5]

            HT = HT @ torch.linalg.matrix_exp(dTh)

        # compute rmse between observation and mesh_correspondences
        rmse = torch.sqrt(
            torch.mean(
                torch.sum(
                    torch.pow(
                        reference_correspondences - observed_correspondences,
                        2,
                    ),
                    dim=-1,
                )
            )
        )

        if abs(previous_rmse - rmse.item()) < rmse_change_tolerance:
            return RegistrationResult(
                extrinsics=HT,
                iterations=outer_iteration,
                termination_reason=TerminationReason.CONVERGED,
            )

        previous_rmse = rmse.item()

    return RegistrationResult(
        extrinsics=HT,
        iterations=max_outer_iterations,
        termination_reason=TerminationReason.MAX_ITERATIONS,
    )
