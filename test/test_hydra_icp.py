import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import torch
from common import load_data

from roboreg.hydra_icp import HydraICP


def test_kabsh_algorithm():
    observed_xyzs, mesh_xyzs = load_data(idcs=[0, 1, 2], scan=True, visualize=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to numpy
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
    # fail cases: 
    # # high res 2,3,4
    # # high res 0
    # # high res 0, scan = True
    observed_xyzs, mesh_xyzs = load_data(
        idcs=[0, 1],
        scan=False,
        visualize=False,
        prefix="test/data/high_res",
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

    hydra_icp = HydraICP()
    hydra_icp(observed_xyzs, mesh_xyzs, max_iter=int(1e6))

    # visualize initial homogenous transform
    HT = hydra_icp.HT

    # to numpy
    HT = HT.cpu().numpy()
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

    # array of colors
    [
        observed_xyzs_pcd.paint_uniform_color(
            [
                0.5
                + (len(observed_xyzs_pcds) - idx - 1) / len(observed_xyzs_pcds) / 2.0,
                0.8,
                0.0,
            ]
        )
        for idx, observed_xyzs_pcd in enumerate(observed_xyzs_pcds)
    ]
    [
        mesh_xyzs_pcd.paint_uniform_color(
            [
                0.5,
                0.5,
                0.5 + (len(mesh_xyzs_pcds) - idx - 1) / len(mesh_xyzs_pcds) / 2.0,
            ]
        )
        for idx, mesh_xyzs_pcd in enumerate(mesh_xyzs_pcds)
    ]

    # visualize
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
    for observed_xyzs_pcd in observed_xyzs_pcds:
        visualizer.add_geometry(observed_xyzs_pcd)
    for mesh_xyzs_pcd in mesh_xyzs_pcds:
        visualizer.add_geometry(mesh_xyzs_pcd)
    visualizer.run()

    # transform mesh
    for i in range(len(mesh_xyzs_pcds)):
        mesh_xyzs_pcds[i] = mesh_xyzs_pcds[i].transform(HT)

    # visualize
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
    for observed_xyzs_pcd in observed_xyzs_pcds:
        visualizer.add_geometry(observed_xyzs_pcd)
    for mesh_xyzs_pcd in mesh_xyzs_pcds:
        visualizer.add_geometry(mesh_xyzs_pcd)
    visualizer.run()


if __name__ == "__main__":
    # test_kabsh_algorithm()
    test_hydra_icp()
