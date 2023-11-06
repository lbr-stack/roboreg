import pathlib
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyvista as pv

from roboreg.ray_cast import RayCastRobot
from roboreg.util import clean_xyz, generate_o3d_robot


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
    robot = generate_o3d_robot()

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
            [mesh_pcd.estimate_normals() for mesh_pcd in mesh_pcds]
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


def find_files(path: str, pattern: str = "img_*.png") -> List[str]:
    path = pathlib.Path(path)
    image_paths = list(path.glob(pattern))
    return [image_path.name for image_path in image_paths]
