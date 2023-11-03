import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import torch
from common import load_data, visualize_registration

from roboreg.hydra_robust_icp import HydraRobustICP


def test_hydra_robust_icp():
    prefix = "test/data/low_res"
    observed_xyzs, mesh_xyzs = load_data(
        idcs=[0, 1, 2, 3, 4],
        scan=False,
        visualize=False,
        prefix=prefix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to numpy
    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = torch.from_numpy(observed_xyzs[i]).to(
            dtype=torch.float32, device=device
        )
        mesh_xyzs[i] = torch.from_numpy(mesh_xyzs[i]).to(
            dtype=torch.float32, device=device
        )

    hydra_robust_icp = HydraRobustICP()
    hydra_robust_icp(observed_xyzs, mesh_xyzs, max_distance=1.0, max_iter=int(1e3))

    # visualize initial homogenous transform
    HT = hydra_robust_icp.HT

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(prefix, "HT_hydra_robust.npy"), HT)

    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    visualize_registration(observed_xyzs, mesh_xyzs, HT)


if __name__ == "__main__":
    test_hydra_robust_icp()
