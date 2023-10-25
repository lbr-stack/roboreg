import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List, Tuple

import kinpy
import numpy as np
import pyvista
import transformations as tf
import trimesh
from ament_index_python import get_package_share_directory
from kinpy.chain import Chain
from pyvista import pyvista_ndarray
from rich import print


class MeshifyRobot:
    chain: Chain
    joint_names: List[str]
    dof: int
    paths: List[str]
    link_names: List[str]
    meshes: List[pyvista.PolyData]

    def __init__(self, urdf: str, resolution: str = "collision") -> None:
        self.chain = self._load_chain(urdf)
        self.joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(self.joint_names)
        if resolution not in ["collision", "visual"]:
            raise ValueError(f"Resolution {resolution} not supported.")
        self.paths, self.link_names = self._get_mesh_paths(urdf, resolution)
        self.meshes = self._load_meshes(self.paths)

    def transformed_meshes(self, q: np.ndarray) -> List[pyvista.PolyData]:
        zero_tf_dict = self._get_transforms(np.zeros(self.dof))
        tf_dict = self._get_transforms(q)

        meshes = deepcopy(self.meshes)
        for mesh, link in zip(meshes, self.link_names):
            # zero transform
            r0 = tf.quaternion_matrix(zero_tf_dict[link].rot)
            t0 = tf.translation_matrix(zero_tf_dict[link].pos)
            ht0 = np.eye(4)
            ht0[:3, :3] = r0[:3, :3]
            ht0[:3, 3] = t0[:3, 3]

            # desired transform
            r = tf.quaternion_matrix(tf_dict[link].rot)
            t = tf.translation_matrix(tf_dict[link].pos)
            ht = np.eye(4)
            ht[:3, :3] = r[:3, :3]
            ht[:3, 3] = t[:3, 3]

            mesh.transform(ht @ np.linalg.inv(ht0))
        return meshes

    def plot_meshes(
        self, meshes, background_color: str = "black", mesh_color: str = "white"
    ) -> None:
        plotter = pyvista.Plotter()
        plotter.background_color = background_color
        for mesh in meshes:
            plotter.add_mesh(mesh, color=mesh_color, point_size=2.0)
        plotter.show()

    def plot_point_clouds(
        self, meshes, background_color: str = "black", point_color: str = "white"
    ) -> None:
        point_clouds = self.meshes_to_point_clouds(meshes)
        point_clouds = [pyvista.PolyData(point_cloud) for point_cloud in point_clouds]
        plotter = pyvista.Plotter()
        plotter.background_color = background_color
        for point_cloud in point_clouds:
            plotter.add_mesh(point_cloud, point_size=2.0, color=point_color)
        plotter.show()

    def meshes_to_point_clouds(
        self, meshes: List[pyvista.DataSet]
    ) -> List[pyvista_ndarray]:
        point_clouds = [self.mesh_to_point_cloud(mesh) for mesh in meshes]
        return point_clouds

    def meshes_to_point_cloud(self, meshes: List[pyvista.DataSet]) -> np.ndarray:
        point_clouds = self.meshes_to_point_clouds(meshes)
        point_cloud = np.concatenate(point_clouds, axis=0)
        return point_cloud

    def mesh_to_point_cloud(self, mesh: pyvista.DataSet) -> pyvista_ndarray:
        point_cloud = mesh.points
        return point_cloud

    def homogenous_point_cloud_sampling(
        self, point_cloud: np.ndarray, N: int
    ) -> np.ndarray:
        # define a bounding box
        min_x, max_x = point_cloud[:, 0].min(), point_cloud[:, 0].max()
        min_y, max_y = point_cloud[:, 1].min(), point_cloud[:, 1].max()
        min_z, max_z = point_cloud[:, 2].min(), point_cloud[:, 2].max()

        # sample points
        x = np.random.uniform(min_x, max_x, N)
        y = np.random.uniform(min_y, max_y, N)
        z = np.random.uniform(min_z, max_z, N)

        # find corresponding points in point cloud
        sampled_points = []
        for i in range(N):
            dist = np.linalg.norm(point_cloud - np.array([x[i], y[i], z[i]]), axis=1)
            idx = np.argmin(dist)
            sampled_points.append(point_cloud[idx])

        return np.array(sampled_points)

    def _sub_sample(self, data: np.ndarray, N: int):
        indices = np.random.choice(data.shape[0], N, replace=False)
        sampled_points = data[indices]
        return sampled_points

    def _get_transforms(self, q: np.ndarray) -> Dict[str, kinpy.Transform]:
        transforms = self.chain.forward_kinematics(q)
        return transforms

    def _get_mesh_paths(
        self, urdf: str, resolution: str = "collision"
    ) -> Tuple[List[str], List[str]]:
        paths = []
        names = []

        def handle_package_path(package: str, filename: str):
            package_path = get_package_share_directory(package)
            return os.path.join(package_path, filename)

        robot = ET.fromstring(urdf)
        for link in robot.findall("link"):
            visual = link.find(resolution)
            if visual:
                name = link.attrib["name"]
                geometry = visual.find("geometry")
                mesh = geometry.find("mesh")
                filename = mesh.attrib["filename"]

                if filename.startswith("package://"):
                    filename = filename.replace("package://", "")
                    package, filename = filename.split("/", 1)
                    path = handle_package_path(package, filename)
                    names.append(name)
                    paths.append(path)
        return paths, names

    def _load_mesh(self, path: str) -> pyvista.PolyData:
        print(f"Loading mesh from {path}")
        if path.endswith(".stl"):
            mesh = pyvista.read(path)
        elif path.endswith(".dae"):
            scene = trimesh.load_mesh(path)
            vertices = []
            faces = []
            for geometry in scene.geometry.values():
                vertices.append(geometry.vertices)
                faces.append(geometry.faces)
            vertices = np.concatenate(vertices, axis=0).tolist()
            faces = np.concatenate(faces, axis=0).tolist()
            mesh = pyvista.PolyData(vertices, faces)
        else:
            raise NotImplementedError(f"File type {path} not supported.")
        return mesh

    def _load_meshes(self, paths: List[str]) -> List[pyvista.PolyData]:
        meshes = [self._load_mesh(path) for path in paths]
        return meshes

    def _load_chain(self, urdf: str) -> Chain:
        chain = kinpy.build_chain_from_urdf(urdf)
        return chain
