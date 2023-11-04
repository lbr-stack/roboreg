import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import itertools
import torch
from common import load_data

from roboreg.hydra_icp import to_homogeneous, from_homogeneous
from roboreg.hydra_icp import HydraICP


def main():
    N = 10
    k = 9
    sample_idcs = [i for i in range(N)]

    for comb in itertools.combinations(sample_idcs, k):
        observed_xyzs, mesh_xyzs, _ = load_data(
            idcs=comb,
            scan=False,
            visualize=False,
            prefix="test/data/low_res",
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
                    0.5,
                    0.8,
                    0.5
                    + (len(observed_xyzs_pcds) - idx - 1)
                    / len(observed_xyzs_pcds)
                    / 2.0,
                ]
            )
            for idx, observed_xyzs_pcd in enumerate(observed_xyzs_pcds)
        ]
        [
            mesh_xyzs_pcd.paint_uniform_color(
                [
                    0.5 + (len(mesh_xyzs_pcds) - idx - 1) / len(mesh_xyzs_pcds) / 2.0,
                    0.5,
                    0.8,
                ]
            )
            for idx, mesh_xyzs_pcd in enumerate(mesh_xyzs_pcds)
        ]

        # transform mesh
        for i in range(len(mesh_xyzs_pcds)):
            mesh_xyzs_pcds[i] = mesh_xyzs_pcds[i].transform(HT)

        # find ki not in combination
        for i in range(N):
            if i in comb:
                continue

            # load i
            observed_xyz, mesh_xyz, _ = load_data(
                idcs=[i],
                scan=False,
                visualize=False,
                prefix="test/data/low_res",
            )

            # to torch
            observed_xyz = torch.from_numpy(observed_xyz[0]).to(
                dtype=torch.float32, device=device
            )
            mesh_xyz = torch.from_numpy(mesh_xyz[0]).to(
                dtype=torch.float32, device=device
            )

            # to homogeneous
            mesh_xyz = to_homogeneous(mesh_xyz)

            # transform
            HT = torch.from_numpy(HT).to(dtype=torch.float32, device=device)
            mesh_xyz = torch.matmul(HT, mesh_xyz.transpose(-2, -1)).transpose(-2, -1)

            # from homogeneous
            mesh_xyz = from_homogeneous(mesh_xyz)

            # to numpy
            observed_xyz = observed_xyz.cpu().numpy()
            mesh_xyz = mesh_xyz.cpu().numpy()

            # visualize
            mesh_xyz_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_xyz))
            mesh_xyz_pcd.paint_uniform_color(
                [
                    0.5 + (len(mesh_xyzs_pcds) - i - 1) / len(mesh_xyzs_pcds) / 2.0,
                    0.5,
                    0.8,
                ]
            )
            observed_xyzs_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(observed_xyz)
            )
            observed_xyzs_pcd.paint_uniform_color(
                [
                    0.5,
                    0.8,
                    0.5 + (len(mesh_xyzs_pcds) - i - 1) / len(mesh_xyzs_pcds) / 2.0,
                ]
            )

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
            visualizer.add_geometry(observed_xyzs_pcd)
            visualizer.add_geometry(mesh_xyz_pcd)
            visualizer.run()
            visualizer.destroy_window()

        # visualize
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
        for observed_xyzs_pcd in observed_xyzs_pcds:
            visualizer.add_geometry(observed_xyzs_pcd)
        for mesh_xyzs_pcd in mesh_xyzs_pcds:
            visualizer.add_geometry(mesh_xyzs_pcd)
        visualizer.run()
        visualizer.destroy_window()


if __name__ == "__main__":
    main()
