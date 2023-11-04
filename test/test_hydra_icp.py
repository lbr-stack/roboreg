import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import torch
from common import load_data, visualize_registration

from roboreg.hydra_icp import HydraICP


def test_kabsh_algorithm():
    observed_xyzs, mesh_xyzs, _ = load_data(idcs=[0, 1, 2], scan=True, visualize=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to torch
    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = torch.from_numpy(observed_xyzs[i]).to(
            dtype=torch.float32, device=device
        )
        mesh_xyzs[i] = torch.from_numpy(mesh_xyzs[i]).to(
            dtype=torch.float32, device=device
        )

    hydra_icp = HydraICP()
    hydra_icp(observed_xyzs, mesh_xyzs)

    # visualize initial homogenous transform
    HT_init = hydra_icp.HT_init

    # to numpy
    HT_init = HT_init.cpu().numpy()
    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    # visualize
    observed_xyzs_pcds = [
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed_xyz))
        for observed_xyz in observed_xyzs
    ]
    mesh_xyzs_pcds = [
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_xyz))
        for mesh_xyz in mesh_xyzs
    ]

    # color
    [
        observed_xyzs_pcd.paint_uniform_color([1, 0.706, 0])
        for observed_xyzs_pcd in observed_xyzs_pcds
    ]
    [
        mesh_xyzs_pcd.paint_uniform_color([0, 0.651, 0.929])
        for mesh_xyzs_pcd in mesh_xyzs_pcds
    ]

    # visualize
    o3d.visualization.draw_geometries(observed_xyzs_pcds + mesh_xyzs_pcds)

    # transform mesh
    for i in range(len(mesh_xyzs_pcds)):
        mesh_xyzs_pcds[i] = mesh_xyzs_pcds[i].transform(HT_init)

    # visualize
    o3d.visualization.draw_geometries(observed_xyzs_pcds + mesh_xyzs_pcds)


def test_hydra_icp():
    prefix = "test/data/high_res"
    observed_xyzs, mesh_xyzs, _ = load_data(
        idcs=[0, 1, 2, 3, 4],
        scan=False,
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

    hydra_icp = HydraICP()
    hydra_icp(observed_xyzs, mesh_xyzs, max_distance=1.0, max_iter=int(1e3))

    # visualize initial homogenous transform
    HT = hydra_icp.HT

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(prefix, "HT_hydra.npy"), HT)

    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = observed_xyzs[i].cpu().numpy()
        mesh_xyzs[i] = mesh_xyzs[i].cpu().numpy()

    visualize_registration(observed_xyzs, mesh_xyzs, HT)


if __name__ == "__main__":
    # test_kabsh_algorithm()
    test_hydra_icp()
