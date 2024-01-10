from typing import List, Tuple

import faiss
import faiss.contrib.torch_utils
import torch
from rich import print
from rich.progress import track


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N) to (..., N+1) by appending ones."""
    return torch.nn.functional.pad(x, (0, 1), "constant", 1.0)


def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N+1) to (..., N)."""
    return x[..., :-1]


def kabsh_register(
    input: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Kabsh algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm.
    Computes rotation and translation such that input @ R + t = target.

    Args:
        input: input of shape (..., M, 3).
        target: target of shape (..., M, 3).

    Return:
        R: Rotation matrix of shape (..., 3, 3).
        t: Translation vector of shape (..., 3).
    """
    # compute centroids
    input_centroid = torch.mean(input, dim=-2)
    target_centroid = torch.mean(target, dim=-2)

    # compute centered points
    input_centered = input - input_centroid
    target_centered = target - target_centroid

    # compute covariance matrix
    H = target_centered.transpose(-1, -2) @ input_centered

    # compute SVD
    U, _, V = torch.svd(H)

    E = torch.eye(3, dtype=U.dtype, device=U.device)
    E[-1, -1] = torch.det(V @ U.transpose(-1, -2))

    # compute rotation
    R = V @ E @ U.transpose(-1, -2)

    # compute translation
    t = target_centroid - input_centroid @ R
    return R, t


def print_line():
    print("--------------------------------------------------")


def hydra_closest_correspondence_indices(
    observations: List[torch.Tensor],
    meshes: List[torch.Tensor],
    max_distance: float = 0.1,
) -> List[torch.Tensor]:
    r"""For each point in observation, find nearest neighbor index in mesh.

    Args:
        observations: List of observations of shape (Mi, 3).
        meshes: List of meshes of shape (Ni, 3).

    Returns:
        argmins: List of indices of shape (Mi).
    """
    argmins = []
    for observation, mesh in zip(observations, meshes):
        distance = torch.cdist(observation, mesh)  # (Mi, Ni)
        distance = torch.where(
            distance < max_distance, distance, torch.full_like(distance, float("inf"))
        )

        _, argmin = torch.min(distance, dim=-1)  # (Mi)
        argmins.append(argmin)

    return argmins


def hydra_gpu_index_flat_l2(meshes: List[torch.Tensor]) -> List[faiss.GpuIndexFlatL2]:
    indices = []
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    for mesh in meshes:
        index = faiss.GpuIndexFlatL2(res, 3, flat_config)
        index.add(mesh)
        indices.append(index)
    return indices


def hydra_centroid_alignment(
    Xs: List[torch.Tensor],
    Ys: List[torch.Tensor],
) -> torch.Tensor:
    r"""Aligns centroids of Xs and Ys as an initial guess.

    Args:
        Xs: List of poinclouds of shape (Mi, 3).
        Ys: List of pointclouds of shape (Ni, 3).

    Returns:
        HT: Homogeneous transformation of shape (4, 4). HT @ Xs = Ys.
    """
    # for each cloud compute centroid
    Xs_centroids = [torch.mean(observation, dim=-2) for observation in Xs]
    Ys_centroids = [torch.mean(mesh, dim=-2) for mesh in Ys]

    # estimate transform
    R, t = kabsh_register(
        torch.stack(Xs_centroids).unsqueeze(0),
        torch.stack(Ys_centroids).unsqueeze(0),
    )

    HT = torch.eye(4, dtype=R.dtype, device=R.device)
    R = R.squeeze(0)
    t = t.squeeze(0)
    HT[:3, :3] = R.T
    HT[:3, 3] = t
    return HT


def hydra_icp(
    HT_init: torch.Tensor,
    observations: List[torch.Tensor],
    meshes: List[torch.Tensor],
    max_distance: float = 0.1,
    max_iter: int = 100,
    rmse_change: float = 1e-6,
    exit_early: bool = True,
) -> torch.Tensor:
    r"""Hydra iterative closest point algorithm.

    Args:
        HT_init: Initial guess. HT_init @ observations = meshes.
        observations: List of observations of shape (Mi, 3).
        meshes: List of meshes of shape (Ni, 3).
        max_distance: Maximum distance between point correspondences.
        max_iter: Maximum number of iterations.
        rmse_change: Minimum change in rmse to continue iterating.

    Returns:
        HT: Homogeneous transformation of shape (4, 4). HT @ observations = meshes.
    """
    HT = HT_init

    # build index
    indices = hydra_gpu_index_flat_l2(meshes)

    # registration
    prev_rmse = float("inf")
    for _ in track(range(max_iter), description=f"Running Hydra ICP..."):
        observation_corr = []
        mesh_corr = []
        for i in range(len(meshes)):
            # search correspondences
            observations_tf = observations[i] @ HT[:3, :3].T + HT[:3, 3]
            distances, matchindices = indices[i].search(observations_tf, 1)

            # only keep matches within max_distance
            mask = distances.squeeze() < max_distance

            observation_corr.append(observations[i][mask])
            mesh_corr.append(meshes[i][matchindices[mask].squeeze()])

        observation_corr = torch.concatenate(observation_corr).unsqueeze(0)
        mesh_corr = torch.concatenate(mesh_corr).unsqueeze(0)

        (
            R,
            t,
        ) = kabsh_register(
            observation_corr,
            mesh_corr,
        )
        R = R.squeeze(0)
        t = t.squeeze(0)
        HT[:3, :3] = R.T
        HT[:3, 3] = t

        # compute rmse between observation and mesh_corr
        rmse = torch.sqrt(
            torch.mean(
                torch.sum(
                    torch.pow(
                        mesh_corr - observation_corr,
                        2,
                    ),
                    dim=-1,
                )
            )
        )

        if abs(prev_rmse - rmse.item()) < rmse_change and exit_early:
            print("Converged early. Exiting.")
            break

        prev_rmse = rmse.item()

    print_line()
    print("HT estimate:\n", HT)
    print_line()

    return HT


def hydra_robust_icp(
    HT_init: torch.Tensor,
    observations: List[torch.Tensor],
    meshes: List[torch.Tensor],
    mesh_normals: List[torch.Tensor],
    max_distance: float = 0.1,
    outer_max_iter: int = 100,
    inner_max_iter: int = 3,
    rmse_change: float = 1e-6,
) -> torch.Tensor:
    r"""Lie-algebra point-to-plane ICP with robust loss, refer to section 1
    https://drive.google.com/file/d/1iIUqKchAbcYzwyS2D6jNI1J6KotReD1h/view?usp=sharing.

    Args:
        HT_init: Initial guess. HT_init @ observations = meshes.
        observations: List of observations of shape (Mi, 3).
        meshes: List of meshes of shape (Ni, 3).
        mesh_normals: List of mesh normals of shape (Ni, 3).
        max_distance: Maximum distance between point correspondences.
        outer_max_iter: Maximum number of outer iterations.
        inner_max_iter: Maximum number of inner iterations.
        rmse_change: Minimum change in rmse to continue iterating.

    Returns:
        HT: Homogeneous transformation of shape (4, 4). HT @ observations = meshes.
    """
    HT = HT_init  # HT @ observation = mesh

    # build indices
    indices = hydra_gpu_index_flat_l2(meshes)

    observations_cross_mat = []
    for i in range(len(observations)):
        # build observation cross product matrix, refer eq. 4 (gets created once)
        observations_cross_mat.append(
            torch.stack(
                [
                    torch.zeros_like(observations[i][:, 0]),
                    -observations[i][:, 2],
                    observations[i][:, 1],
                    observations[i][:, 2],
                    torch.zeros_like(observations[i][:, 0]),
                    -observations[i][:, 0],
                    -observations[i][:, 1],
                    observations[i][:, 0],
                    torch.zeros_like(observations[i][:, 0]),
                ],
                dim=-1,
            ).reshape(-1, 3, 3)
        )

    # implementation of algorithm 1
    prev_rmse = float("inf")
    dTh = torch.zeros_like(HT)
    for _ in track(range(outer_max_iter), description=f"Running Hydra robust ICP..."):
        observations_corr = []
        observations_cross_mat_corr = []
        meshes_corr = []
        meshes_normals_corr = []

        for i in range(len(observations)):
            if len(observations) != len(meshes):
                raise ValueError("Length of observations and meshes must be the same.")
            # search correspondences
            observation_tf = observations[i] @ HT[:3, :3].T + HT[:3, 3]
            distances, matchindices = indices[i].search(observation_tf, 1)

            # only keep matches within max_distance
            mask = distances.squeeze() < max_distance

            observations_corr.append(observations[i][mask])
            observations_cross_mat_corr.append(observations_cross_mat[i][mask])
            meshes_corr.append(meshes[i][matchindices[mask].squeeze()])
            meshes_normals_corr.append(mesh_normals[i][matchindices[mask].squeeze()])

        observations_corr = torch.cat(observations_corr)
        observations_cross_mat_corr = torch.cat(observations_cross_mat_corr)
        meshes_corr = torch.cat(meshes_corr)
        meshes_normals_corr = torch.cat(meshes_normals_corr)

        for _ in range(inner_max_iter):
            # ||A @ dTh - B||^2, refer eq. 14
            Al = meshes_normals_corr @ HT[:3, :3]  # eq. 18
            Au = -Al.unsqueeze(1) @ observations_cross_mat_corr  # eq. 19
            A = torch.cat((Au.squeeze(), Al.squeeze()), dim=-1)
            B = torch.linalg.vecdot(
                meshes_normals_corr,
                meshes_corr - (observations_corr @ HT[:3, :3].T + HT[:3, 3]),
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

        # compute rmse between observation and mesh_corr
        rmse = torch.sqrt(
            torch.mean(
                torch.sum(
                    torch.pow(
                        meshes_corr - observations_corr,
                        2,
                    ),
                    dim=-1,
                )
            )
        )

        if abs(prev_rmse - rmse.item()) < rmse_change:
            print("Converged early. Exiting.")
            break

        prev_rmse = rmse.item()

    print_line()
    print("HT estimate:\n", HT)
    print_line()

    return HT
