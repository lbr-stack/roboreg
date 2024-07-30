import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
from pytorch_kinematics import urdf_parser_py

from roboreg.util import clean_xyz, generate_o3d_robot, mask_boundary


class URDFParser:
    _robot: urdf_parser_py.urdf.Robot

    def __init__(self) -> None:
        self._robot = None

    def from_urdf(self, urdf: str) -> None:
        self._robot = urdf_parser_py.urdf.Robot.from_xml_string(urdf)

    def from_ros_xacro(self, ros_package: str, xacro_path: str) -> None:
        self.from_urdf(
            urdf=self.urdf_from_ros_xacro(
                ros_package=ros_package, xacro_path=xacro_path
            )
        )

    def urdf_from_ros_xacro(self, ros_package: str, xacro_path: str) -> str:
        import xacro
        from ament_index_python import get_package_share_directory

        return xacro.process(
            os.path.join(get_package_share_directory(ros_package), xacro_path)
        )

    def chain_link_names(self, root_link_name: str, end_link_name: str) -> List[str]:
        self._verify_links_in_chain(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        link_names = [root_link_name]
        while link_names[-1] != end_link_name:
            children = self._robot.child_map[link_names[-1]]
            if len(children) != 1:
                raise RuntimeError(f"Expected 1 child, got {len(children)}.")
            _, child_link_name = children[0]
            if link_names[-1] == child_link_name:
                raise RuntimeError(f"End of chain without {end_link_name}.")
            link_names.append(child_link_name)
        return link_names

    def raw_mesh_paths(
        self, root_link_name: str, end_link_name: str, visual: bool = False
    ) -> Dict[str, str]:
        link_names = self.chain_link_names(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        raw_mesh_paths = {}
        # lookup paths
        for link_name in link_names:
            link: urdf_parser_py.urdf.Link = self._robot.link_map[link_name]
            if visual:
                if link.visual is None:
                    continue
                raw_mesh_paths[link_name] = link.visual.geometry.filename
            else:
                if link.collision is None:
                    continue
                raw_mesh_paths[link_name] = link.collision.geometry.filename
        return raw_mesh_paths

    def ros_package_mesh_paths(
        self, root_link_name: str, end_link_name: str, visual: bool = False
    ) -> Dict[str, str]:
        raw_mesh_paths = self.raw_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name, visual=visual
        )
        from ament_index_python import get_package_share_directory

        ros_package_mesh_paths = {}
        for link_name in raw_mesh_paths.keys():
            raw_mesh_path = raw_mesh_paths[link_name]
            if raw_mesh_path.startswith("package://"):
                raw_mesh_path = raw_mesh_path.replace("package://", "")
                package, relative_mesh_path = raw_mesh_path.split("/", 1)
                ros_package_mesh_paths[link_name] = os.path.join(
                    get_package_share_directory(package), relative_mesh_path
                )
            else:
                raise ValueError("Case unhandled.")
        return ros_package_mesh_paths

    def link_origins(
        self, root_link_name: str, end_link_name: str, visual: bool = False
    ) -> Dict[str, np.ndarray]:
        import transformations

        link_names = self.chain_link_names(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        link_origins = {}
        for link_name in link_names:
            link: urdf_parser_py.urdf.Link = self._robot.link_map[link_name]
            if visual:
                if link.visual is None:
                    continue
                link_origin = link.visual.origin
            else:
                if link.collision is None:
                    continue
                link_origin = link.collision.origin
            origin = transformations.euler_matrix(
                link_origin.rpy[0], link_origin.rpy[1], link_origin.rpy[2], "sxyz"
            )
            origin[:3, 3] = link_origin.xyz
            link_origins[link_name] = origin
        return link_origins

    def _verify_links_in_chain(self, root_link_name: str, end_link_name: str) -> None:
        if not self._robot:
            raise RuntimeError("Robot not initialized.")
        link_names = [link.name for link in self._robot.links]
        if not end_link_name in link_names:
            raise ValueError(f"Link {end_link_name} not in robot.")
        if not root_link_name in link_names:
            raise ValueError(f"Link {root_link_name} not in robot.")

    @property
    def robot(self) -> urdf_parser_py.urdf.Robot:
        return self._robot


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
