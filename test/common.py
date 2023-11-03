import os
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
import xacro
from ament_index_python import get_package_share_directory

from roboreg.o3d_robot import O3DRobot
from roboreg.ray_cast import RayCastRobot
from roboreg.util import clean_xyz


def load_data(
    idcs: List[int],
    scan: bool = True,
    visualize: bool = False,
    prefix: str = "test/data/low_res",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    clean_observed_xyzs = []
    mesh_xyzs = []
    mesh_xyzs_normals = []

    # load robot
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"),
            "urdf/med7/med7.urdf.xacro",
        )
    )
    robot = O3DRobot(urdf=urdf)

    for idx in idcs:
        # load data
        mask = cv2.imread(f"{prefix}/mask_{idx}.png", cv2.IMREAD_GRAYSCALE)
        observed_xyz = np.load(f"{prefix}/xyz_{idx}.npy")
        joint_state = np.load(f"{prefix}/joint_state_{idx}.npy")

        # clean cloud
        clean_observed_xyzs.append(clean_xyz(observed_xyz, mask))

        # visualize clean cloud
        if visualize:
            plotter = pv.Plotter()
            plotter.background_color = "black"
            plotter.add_mesh(clean_observed_xyzs[-1], point_size=2.0, color="white")
            plotter.show()

        # transform mesh
        mesh_xyz = None
        mesh_xyz_normals = None

        if scan:
            # raycast views
            ray_cast = RayCastRobot(robot)
            ray_cast.robot.set_joint_positions(joint_state)
            mesh_pcds = []
            eyes = [
                [0, 2, 0],
                [0, -2, 0],
                [2, 0, 0],
                [-2, 0, 0],
            ]
            for eye in eyes:
                mesh_pcds.append(
                    ray_cast.cast(
                        fov_deg=90,
                        center=o3d.core.Tensor([0, 0, 0]),
                        eye=o3d.core.Tensor(eye),
                        up=o3d.core.Tensor([0, 0, 1]),
                        width_px=640,
                        height_px=480,
                    )
                )

            mesh_xyz = np.concatenate(
                [mesh_pcd.point.positions.numpy() for mesh_pcd in mesh_pcds], axis=0
            )
            mesh_xyz_normals = np.concatenate(
                [mesh_pcd.point.normals.numpy() for mesh_pcd in mesh_pcds], axis=0
            )
        else:
            robot.set_joint_positions(joint_state)
            mesh_xyz = np.concatenate(
                [np.array(pcd.points) for pcd in robot.sample_point_clouds()]
            )
            mesh_xyz_normals = np.concatenate(
                [np.array(pcd.normals) for pcd in robot.sample_point_clouds()]
            )
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
