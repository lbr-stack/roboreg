import os
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d

from roboreg.util import clean_xyz, generate_o3d_robot, mask_boundary


def load_data(
    path: str,
    mask_files: List[str],
    xyz_files: List[str],
    joint_state_files: List[str],
    number_of_points: int = 5000,
    masked_boundary: bool = True,
    erosion_kernel_size: int = 10,
    convex_hull: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    clean_observed_xyzs = []
    mesh_xyzs = []
    mesh_xyzs_normals = []

    # load robot
    robot = generate_o3d_robot(convex_hull=convex_hull)

    for mask_file, xyz_file, joint_state_file in zip(
        mask_files, xyz_files, joint_state_files
    ):
        # load data
        mask = cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE)
        if masked_boundary:
            mask = mask_boundary(
                mask, np.ones([erosion_kernel_size, erosion_kernel_size])
            )
        observed_xyz = np.load(os.path.join(path, xyz_file))
        joint_state = np.load(os.path.join(path, joint_state_file))

        # clean cloud
        clean_observed_xyzs.append(clean_xyz(observed_xyz, mask))

        # transform mesh
        mesh_xyz = None
        mesh_xyz_normals = None

        robot.set_joint_positions(joint_state)
        pcds = robot.sample_point_clouds_equally(number_of_points=number_of_points)
        mesh_xyz = np.concatenate([np.array(pcd.points) for pcd in pcds])
        mesh_xyz_normals = np.concatenate([np.array(pcd.normals) for pcd in pcds])
        mesh_xyzs.append(mesh_xyz)
        mesh_xyzs_normals.append(mesh_xyz_normals)

    return clean_observed_xyzs, mesh_xyzs, mesh_xyzs_normals


def visualize_registration(
    observed_xyzs: List[np.ndarray], mesh_xyzs: List[np.ndarray], HT: np.ndarray
) -> None:
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
                + (len(observed_xyzs_pcds) - idx - 1) / len(observed_xyzs_pcds) / 2.0,
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

    # visualize
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
    for observed_xyzs_pcd in observed_xyzs_pcds:
        visualizer.add_geometry(observed_xyzs_pcd)
    for mesh_xyzs_pcd in mesh_xyzs_pcds:
        visualizer.add_geometry(mesh_xyzs_pcd)
    visualizer.run()
    visualizer.close()

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
    visualizer.close()
