from typing import List

import torch
from pytorch3d.ops import corresponding_points_alignment
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

    print("observation centroid:", observation_centroid)
    print("mesh centroid:", mesh_centroid)

    # compute centered points
    observation_centered = observation - observation_centroid
    mesh_centered = mesh - mesh_centroid

    # compute covariance matrix
    H = mesh_centered.transpose(-2, -1).mm(observation_centered)

    # compute SVD
    U, _, V = torch.svd(H)

    # compute rotation
    R = V.mm(U.transpose(-2, -1))

    # compute translation
    t = observation_centroid - R.mm(mesh_centroid.unsqueeze(-1)).squeeze(-1)

    # compute homogeneous transformation
    HT = torch.eye(4)
    HT[:3, :3] = R
    HT[:3, 3] = t
    return HT


class HydraICP(object):
    HT: torch.Tensor
    HT_init: torch.Tensor

    def __init__(self) -> None:
        self.HT = torch.eye(4)
        self.HT_init = torch.eye(4)

    def __call__(
        self,
        observations: List[torch.Tensor],
        meshes: List[torch.Tensor],
        max_distance: float = 0.1,
        max_iter: int = 100,
        rmse_change: float = 1e-6,
    ) -> torch.Tensor:
        # copy meshes
        meshes_clone = [mesh.clone() for mesh in meshes]

        # for each cloud compute centroid
        observation_centroids = [
            torch.mean(observation, dim=-2) for observation in observations
        ]
        mesh_centroids = [torch.mean(mesh, dim=-2) for mesh in meshes_clone]

        print("Observation clouds centroids:", observation_centroids)
        print("Mesh clouds centroids:", mesh_centroids)

        # estimate transform
        R, t, _ = corresponding_points_alignment(
            torch.stack(mesh_centroids).unsqueeze(0),
            torch.stack(observation_centroids).unsqueeze(0),
        )

        R = R.squeeze(0)
        t = t.squeeze(0)

        self.HT_init[:3, :3] = R.transpose(-2, -1)
        self.HT_init[:3, 3] = t
        self.HT = self.HT_init

        print("HT estimate:", self.HT_init)

        prev_rsme = float("inf")

        for _ in track(range(max_iter), description=f"Running Hydra ICP..."):
            for i in range(len(meshes)):
                meshes_clone[i] = meshes[i].mm(R) + t

            argmins = self._find_correspondence_indices(
                observations, meshes_clone, max_distance
            )

            mesh_correspondences = []
            for i in range(len(meshes)):
                mesh_correspondences.append(meshes[i][argmins[i]])

            mesh_correspondences_concat = torch.concatenate(
                mesh_correspondences
            ).unsqueeze(0)
            observations_concat = torch.concatenate(observations).unsqueeze(0)

            R, t, _ = corresponding_points_alignment(
                mesh_correspondences_concat,
                observations_concat,
            )
            R = R.squeeze(0)
            t = t.squeeze(0)

            # compute rsme between observation and mesh_correspondences
            rsme = torch.sqrt(
                torch.mean(
                    torch.sum(
                        torch.pow(
                            mesh_correspondences_concat - observations_concat,
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

        self.HT[:3, :3] = R.transpose(-2, -1)
        self.HT[:3, 3] = t

        print("HT final:", self.HT)

    def _find_correspondence_indices(
        self,
        observations: List[torch.Tensor],
        meshes: List[torch.Tensor],
        max_dist: float = 0.1,
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
                distance < max_dist, distance, torch.full_like(distance, float("inf"))
            )
            _, argmin = torch.min(distance, dim=-1)  # (Mi)
            argmins.append(argmin)

        return argmins
