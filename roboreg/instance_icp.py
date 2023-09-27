from typing import List

import torch


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of shape (..., N) to (..., N+1) by appending ones."""
    return torch.nn.functional.pad(x, (0, 1), "constant", 1.0)


class InstanceICP(object):
    HT: torch.Tensor
    HT_init: torch.Tensor

    def __init__(self) -> None:
        HT = torch.eye(4)
        HT_init = torch.eye(4)

    def __call__(
        self, targets: List[torch.Tensor], sources: List[torch.Tensor]
    ) -> torch.Tensor:
        # for each cloud compute centroid
        target_centroids = [self._cloud_centroid(target) for target in targets]
        source_centroids = [self._cloud_centroid(source) for source in sources]

        print("Target clouds centroids:\n", target_centroids)
        print("Source clouds centroids:\n", source_centroids)

        # run Kabsh algorithm to estimate HT
        self.HT_init = self._kabsh_algorithm(
            torch.stack(target_centroids), torch.stack(source_centroids)
        )

        print("Initial HT using Kabsh algorithm:\n", self.HT_init)

    # def _find_correspondences(
    #         self, target: List[torch.Tensor], source: List[torch.Tensor]
    # ):

    def _cloud_centroid(self, cloud: torch.Tensor) -> torch.Tensor:
        r"""Compute the centroid of a point cloud.

        Args:
            cloud: A tensor of shape (..., N, 3).

        Returns:
            The centroid of the point cloud, a tensor of shape (..., 3).
        """
        return torch.mean(cloud, dim=-2)

    def _kabsh_algorithm(
        self, target: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        r"""Kabsh algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm

        Args:
            target: Target of shape (..., M, 3).
            source: Source of shape (..., M, 3).
        """
        # compute centroids
        target_centroid = self._cloud_centroid(target)
        source_centroid = self._cloud_centroid(source)

        print("Target centroid:\n", target_centroid)
        print("Source centroid:\n", source_centroid)

        # compute centered points
        target_centered = target - target_centroid
        source_centered = source - source_centroid

        # compute covariance matrix
        H = source_centered.transpose(-2, -1).mm(target_centered)

        # compute SVD
        U, _, V = torch.svd(H)

        # compute rotation
        R = V.mm(U.transpose(-2, -1))

        # compute translation
        t = target_centroid - R.mm(source_centroid.unsqueeze(-1)).squeeze(-1)

        # compute homogeneous transformation
        HT = torch.eye(4)
        HT[:3, :3] = R
        HT[:3, 3] = t
        return HT
