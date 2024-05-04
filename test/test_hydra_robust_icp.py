import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from common import load_data, visualize_registration

from roboreg.hydra_icp import hydra_centroid_alignment, hydra_robust_icp


def test_hydra_robust_icp():
    prefix = "test/data/lbr_med7/high_res"
    observed_xyzs, mesh_xyzs, mesh_xyzs_normals = load_data(
        idcs=[i for i in range(7)],
        visualize=False,
        prefix=prefix,
        number_of_points=8000,
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
        mesh_xyzs_normals[i] = torch.from_numpy(mesh_xyzs_normals[i]).to(
            dtype=torch.float32, device=device
        )

    HT_init = hydra_centroid_alignment(observed_xyzs, mesh_xyzs)
    HT = hydra_robust_icp(
        HT_init,
        observed_xyzs,
        mesh_xyzs,
        mesh_xyzs_normals,
        max_distance=0.01,
        outer_max_iter=int(50),
        inner_max_iter=10,
    )

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(prefix, "HT_hydra_robust.npy"), HT)

    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    visualize_registration(observed_xyzs, mesh_xyzs, np.linalg.inv(HT))


if __name__ == "__main__":
    test_hydra_robust_icp()
