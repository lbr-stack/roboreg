import os
import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from pytorch_kinematics import urdf_parser_py
from torch.utils.data import Dataset

from roboreg.util import clean_xyz, generate_o3d_robot, mask_boundary


class URDFParser:
    __slots__ = ["_urdf", "_robot"]

    def __init__(self) -> None:
        self._urdf = None
        self._robot = None

    def from_urdf(self, urdf: str) -> None:
        self._urdf = urdf
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

        self._urdf = xacro.process(
            os.path.join(get_package_share_directory(ros_package), xacro_path)
        )
        return self._urdf

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
    def urdf(self) -> str:
        return self._urdf

    @property
    def robot(self) -> urdf_parser_py.urdf.Robot:
        return self._robot


def find_files(path: str, pattern: str = "image_*.png") -> List[str]:
    r"""Find files in a directory.

    Args:
        path: Path to the directory.
        pattern: Pattern to match. Warn: The sorting key strictly assumes that pattern includes '_{number}.ext'.

    Returns:
        List of file names.
    """
    path = pathlib.Path(path)
    image_paths = list(path.glob(pattern))
    return sorted(
        [image_path.name for image_path in image_paths],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )


class MonocularDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        image_pattern: str,
        joint_states_path: str,
        joint_states_pattern: str,
    ):
        self._images_path = images_path
        self._image_files = find_files(images_path, image_pattern)
        self._joint_states_path = joint_states_path
        self._joint_states_files = find_files(joint_states_path, joint_states_pattern)

        if len(self._image_files) != len(self._joint_states_files):
            raise ValueError(
                f"Number of images '{len(self._image_files)}' and joint states '{len(self._joint_states_files)}' do not match."
            )

        if len(self._image_files) == 0:
            raise ValueError("No images found.")

        if len(self._joint_states_files) == 0:
            raise ValueError("No joint states found.")

        for image_file, joint_states_file in zip(
            self._image_files, self._joint_states_files
        ):
            if (
                image_file.split("_")[-1].split(".")[0]
                != joint_states_file.split("_")[-1].split(".")[0]
            ):
                raise ValueError(
                    f"Image file index '{image_file}' and joint states file index '{joint_states_file}' do not match."
                )

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        image_file = self._image_files[idx]
        joint_states_file = self._joint_states_files[idx]
        image = cv2.imread(os.path.join(self._images_path, image_file))
        joint_states = np.load(os.path.join(self._joint_states_path, joint_states_file))
        return image, joint_states, image_file


def parse_camera_info(camera_info_file: str) -> Tuple[int, int, np.ndarray]:
    r"""Parse camera info file.

    Args:
        camera_info_file (str): Absolute path to the camera info file.

    Returns:
        height (int): Height of the image.
        width (int): Width of the image.
        intrinsic_matrix (np.ndarray): Intrinsic matrix of shape 3x3.
    """
    with open(camera_info_file, "r") as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    height = camera_info["height"]
    width = camera_info["width"]
    if len(camera_info["k"]) != 9:
        raise ValueError("Camera matrix must be 3x3.")
    intrinsic_matrix = np.array(camera_info["k"]).reshape(3, 3)
    return height, width, intrinsic_matrix


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
