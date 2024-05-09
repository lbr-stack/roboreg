import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import transformations as tf
from common import load_data, visualize_registration

from roboreg.hydra_icp import hydra_centroid_alignment, hydra_icp


def test_hydra_centroid_alignment():
    mesh_centroids = [
        torch.FloatTensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.FloatTensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    ]

    HT_random = torch.from_numpy(tf.random_rotation_matrix()).float()
    HT_random[:3, 3] = torch.FloatTensor([1.0, 2.0, 3.0])

    observed_centroids = [
        mesh_centroid @ HT_random[:3, :3].T + HT_random[:3, 3]
        for mesh_centroid in mesh_centroids
    ]

    HT = hydra_centroid_alignment(mesh_centroids, observed_centroids)

    assert torch.allclose(HT, HT_random)


def test_hydra_icp():
    prefix = "test/data/lbr_med7/high_res"
    observed_xyzs, mesh_xyzs, _ = load_data(
        idcs=[i for i in range(7)],
        visualize=False,
        prefix=prefix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to torch
    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = torch.from_numpy(observed_xyzs[i]).to(
            dtype=torch.float32, device=device
        )
        mesh_xyzs[i] = torch.from_numpy(mesh_xyzs[i]).to(
            dtype=torch.float32, device=device
        )

    HT_init = hydra_centroid_alignment(observed_xyzs, mesh_xyzs)
    HT = hydra_icp(
        HT_init,
        observed_xyzs,
        mesh_xyzs,
        max_distance=0.1,
        max_iter=int(1e3),
        rmse_change=1e-8,
    )

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(prefix, "HT_hydra.npy"), HT)

    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    visualize_registration(observed_xyzs, mesh_xyzs, np.linalg.inv(HT))


if __name__ == "__main__":
    # test_hydra_centroid_alignment()
    test_hydra_icp()
