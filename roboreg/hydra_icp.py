from typing import List

import faiss
import faiss.contrib.torch_utils
import torch
from pytorch3d.ops import \
    corresponding_points_alignment  # TODO: remove this dependency and use kabsh_register instead
from rich import print
from rich.progress import track


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N) to (..., N+1) by appending ones."""
    return torch.nn.functional.pad(x, (0, 1), "constant", 1.0)


def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N+1) to (..., N)."""
    return x[..., :-1]


def kabsh_register(observation: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
    r"""Kabsh algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm

    Args:
        observation: observation of shape (..., M, 3).
        mesh: mesh of shape (..., M, 3).
    """
    # compute centroids
    observation_centroid = torch.mean(observation, dim=-2)
    mesh_centroid = torch.mean(mesh, dim=-2)

    print("Observation centroid:", observation_centroid)
    print("Mesh centroid:", mesh_centroid)

    # compute centered points
    observation_centered = observation - observation_centroid
    mesh_centered = mesh - mesh_centroid

    # compute covariance matrix
    H = mesh_centered.T.mm(observation_centered)

    # compute SVD
    U, _, V = torch.svd(H)

    # compute rotation
    R = V.mm(U.T)

    # compute translation
    t = observation_centroid - R.mm(mesh_centroid.unsqueeze(-1)).squeeze(-1)

    # compute homogeneous transformation
    HT = torch.eye(4)
    HT[:3, :3] = R
    HT[:3, 3] = t
    return HT


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
    observations: List[torch.Tensor],
    meshes: List[torch.Tensor],
) -> torch.Tensor:
    r"""Aligns centroids of observations and meshes as an initial guess.

    Args:
        observations: List of observations of shape (Mi, 3).
        meshes: List of meshes of shape (Ni, 3).

    Returns:
        HT: Homogeneous transformation of shape (4, 4).
    """
    # for each cloud compute centroid
    observation_centroids = [
        torch.mean(observation, dim=-2) for observation in observations
    ]
    mesh_centroids = [torch.mean(mesh, dim=-2) for mesh in meshes]

    print_line()
    print("Observation clouds centroids:", observation_centroids)
    print("Mesh clouds centroids:", mesh_centroids)
    print_line()

    # estimate transform
    R, t, _ = corresponding_points_alignment(
        torch.stack(mesh_centroids).unsqueeze(0),
        torch.stack(observation_centroids).unsqueeze(0),
    )

    HT = torch.eye(4, dtype=torch.float32, device=observations[0].device)
    R = R.to(HT.device).squeeze(0)
    t = t.to(HT.device).squeeze(0)
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
) -> torch.Tensor:
    HT = HT_init

    # copy meshes
    indices = hydra_gpu_index_flat_l2(meshes)

    # registration
    prev_rsme = float("inf")
    for _ in track(range(max_iter), description=f"Running Hydra ICP..."):
        observation_corr = []
        mesh_corr = []
        n_matches = []
        for i in range(len(observations)):
            # search correspondences
            observation_tf = observations[i] @ HT[:3, :3].T + HT[:3, 3]
            distances, matchindices = indices[i].search(observation_tf, 1)

            # only keep matches within max_distance
            mask = distances.squeeze() < max_distance
            n_matches.append(mask.sum().item())

            observation_corr.append(observations[i][mask])
            mesh_corr.append(meshes[i][matchindices[mask].squeeze()])

        observation_corr = torch.concatenate(observation_corr).unsqueeze(0)
        mesh_corr = torch.concatenate(mesh_corr).unsqueeze(0)

        R, t, _ = corresponding_points_alignment(
            observation_corr,
            mesh_corr,
        )
        R = R.to(HT.device).squeeze(0)
        t = t.to(HT.device).squeeze(0)
        HT[:3, :3] = R.T
        HT[:3, 3] = t

        # compute rsme between observation and mesh_corr
        rsme = torch.sqrt(
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

        if abs(prev_rsme - rsme.item()) < rmse_change:
            print("Converged early. Exiting.")
            break

        prev_rsme = rsme.item()

    print_line()
    print("HT final:\n", HT)
    print_line()

    return HT


class HydraRobustICP(object):
    HT_init: torch.Tensor
    HT: torch.Tensor

    def __init__(self, device: str = "cuda") -> None:
        r"""Lie-algebra point-to-plane ICP with robust loss, refer to https://drive.google.com/file/d/1WxBUNWh07QH4ckzaACJJWsRCyJ1iraJ7/view?usp=sharing."""
        self.HT_init = torch.eye(4, device=device)
        self.HT = torch.eye(4, device=device)

    def __call__(
        self,
        observations: List[torch.Tensor],
        meshes: List[torch.Tensor],
        mesh_normals: List[torch.Tensor],
        max_distance: float = 0.1,
        outer_max_iter: int = 30,
        inner_max_iter: int = 1,
        initial_alignment: bool = True,
    ):
        # copy meshes
        observations_clone = [observation.clone() for observation in observations]

        if initial_alignment:
            # align centroids
            R, t = hydra_centroid_alignment(meshes, observations_clone)
            self.HT_init[:3, :3] = R.T
            self.HT_init[:3, 3] = t
            self.HT = self.HT_init
            print_line()
            print("HT estimate:\n", self.HT_init)
            print_line()

        observations_cross_mat = []
        for i in range(len(observations)):
            # build observation cross product matrix, refer eq. 4 (gets created once)
            observations_cross_mat.append(
                torch.stack(
                    [
                        torch.zeros_like(observations_clone[i][:, 0]),
                        -observations_clone[i][:, 2],
                        observations_clone[i][:, 1],
                        observations_clone[i][:, 2],
                        torch.zeros_like(observations_clone[i][:, 0]),
                        -observations_clone[i][:, 0],
                        -observations_clone[i][:, 1],
                        observations_clone[i][:, 0],
                        torch.zeros_like(observations_clone[i][:, 0]),
                    ],
                    dim=-1,
                ).reshape(-1, 3, 3)
            )
        observations_cross_mat = torch.concatenate(observations_cross_mat)

        # implementation of algorithm 1
        dTh = torch.zeros_like(self.HT)
        for _ in track(
            range(outer_max_iter), description=f"Running Hydra robust ICP..."
        ):
            for i in range(len(observations)):
                observations_clone[i] = (
                    observations[i] @ self.HT[:3, :3].T + self.HT[:3, 3]
                )

            # find correspondences per configuration
            argmins = hydra_closest_correspondence_indices(
                observations_clone, meshes, max_distance=max_distance
            )

            mesh_corr = []
            mesh_normals_corr = []
            for i in range(len(meshes)):
                mesh_corr.append(meshes[i][argmins[i]])
                mesh_normals_corr.append(mesh_normals[i][argmins[i]])

            mesh_corr = torch.concatenate(mesh_corr)
            mesh_normals_corr = torch.concatenate(mesh_normals_corr)

            observations_concat = torch.concatenate(observations_clone)
            observations_concat_tf = observations_concat.clone()

            for _ in range(inner_max_iter):
                # ||A @ dTh - B||^2, refer eq. 14
                Au = mesh_normals_corr @ self.HT[:3, :3]  # eq. 18
                Al = -Au.unsqueeze(1) @ observations_cross_mat  # eq. 19
                A = torch.cat((Au.squeeze(), Al.squeeze()), dim=-1)
                B = torch.linalg.vecdot(
                    mesh_normals_corr,
                    mesh_corr - observations_concat_tf,
                )

                dTh_vec, resid, rank, singvals = torch.linalg.lstsq(A, B)
                print(f"residuals={resid.cpu().numpy()}")
                dTh[0, 1] = dTh_vec[2]
                dTh[0, 2] = dTh_vec[1]
                dTh[1, 0] = dTh_vec[2]
                dTh[1, 2] = -dTh_vec[0]
                dTh[2, 0] = -dTh_vec[1]
                dTh[2, 1] = dTh_vec[0]

                dTh[0, 3] = dTh_vec[3]
                dTh[1, 3] = dTh_vec[4]
                dTh[2, 3] = dTh_vec[5]

                self.HT = self.HT @ torch.linalg.matrix_exp(dTh)

                observations_concat_tf = (
                    observations_concat @ self.HT[:3, :3].T + self.HT[:3, 3]
                )
