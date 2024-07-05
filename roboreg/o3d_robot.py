import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import kinpy
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import transformations as tf
from ament_index_python import get_package_share_directory
from kinpy.chain import Chain
from rich import print


class O3DRobot:
    chain: Chain
    joint_names: List[str]
    dof: int
    paths: List[str]
    link_names: List[str]
    meshes: List[o3d.t.geometry.TriangleMesh]

    def __init__(self, urdf: str, convex_hull: bool = False) -> None:
        self.chain = self._load_chain(urdf)
        self.joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(self.joint_names)
        self.q = np.zeros(self.dof)
        self.paths, self.link_names, self.collision_origins = self._extract_urdf_data(
            urdf
        )
        self.meshes = self._load_meshes(self.paths, convex_hull)
        self._initial_meshes_transform()
        self._render_setup = False

    def set_joint_positions(self, q: np.ndarray) -> None:
        current_tf = self._get_transforms(self.q)
        tf_dict = self._get_transforms(q)

        for idx, link in enumerate(self.link_names):
            # zero transform
            ht0 = current_tf[link].matrix()

            # desired transform
            ht = tf_dict[link].matrix()

            self.meshes[idx].transform(ht @ np.linalg.inv(ht0))
        self.q = q

    def visualize_meshes(self) -> None:
        o3d.visualization.draw(self.meshes)

    def visualize_point_clouds(self) -> None:
        clouds = self.sample_point_clouds_equally()
        o3d.visualization.draw_geometries(clouds)

    def sample_point_clouds(
        self,
        number_of_points_per_link: int = 1000,
    ) -> List[o3d.geometry.PointCloud]:
        return [
            self.mesh_to_point_cloud(mesh, number_of_points_per_link)
            for mesh in self.meshes
        ]

    def sample_point_clouds_equally(
        self,
        number_of_points: int = 5000,
    ) -> List[o3d.geometry.PointCloud]:
        # compute bounding box volume per mesh
        bounding_box_volumes = [
            mesh.get_axis_aligned_bounding_box().volume() for mesh in self.meshes
        ]
        total_bounding_box_volume = sum(bounding_box_volumes)
        samples_per_mesh = [
            int(number_of_points * bounding_box_volume / total_bounding_box_volume)
            for bounding_box_volume in bounding_box_volumes
        ]
        return [
            self.mesh_to_point_cloud(mesh, samples_per_mesh[idx])
            for idx, mesh in enumerate(self.meshes)
        ]

    def mesh_to_point_cloud(
        self, mesh: o3d.geometry.TriangleMesh, number_of_points: int = 1000
    ) -> o3d.geometry.PointCloud:
        return mesh.to_legacy().sample_points_poisson_disk(
            number_of_points=number_of_points
        )

    def setup_render(
        self,
        intrinsic_matrix: np.ndarray,
        extrinsic_matrix: np.ndarray,
        width: int,
        height: int,
        material_color: List[float] = [1.0, 1.0, 1.0, 1.0],
        background_color: List[float] = [0.0, 0.0, 0.0, 1.0],
    ) -> None:
        if self._render_setup:
            return

        # create rendering scene
        self._render = rendering.OffscreenRenderer(width, height)
        self._mtl = o3d.visualization.rendering.MaterialRecord()
        self._mtl.base_color = material_color
        self._mtl.shader = "defaultUnlit"
        self._render.scene.set_background(background_color)

        self._render.setup_camera(
            intrinsic_matrix,
            extrinsic_matrix,
            width,
            height,
        )

        # # compute up eye center from extrinsic matrix
        # up = -extrinsic_matrix[1, :3]
        # eye = -np.linalg.inv(extrinsic_matrix[:3, :3]) @ extrinsic_matrix[:3, 3]
        # center = eye + extrinsic_matrix[2, :3]
        # render.scene.camera.look_at(center, eye, up)
        self._render_setup = True

    def render(
        self,
    ) -> np.ndarray:
        for idx, mesh in enumerate(self.meshes):
            self._render.scene.add_geometry(f"link_{idx}", mesh, self._mtl)
        if not self._render_setup:
            raise RuntimeError("Render not prepared. Call setup_render first.")
        render = self._render.render_to_image()
        # remove geometries
        for idx in range(len(self.meshes)):
            self._render.scene.remove_geometry(f"link_{idx}")
        return np.asarray(render)

    def _initial_meshes_transform(self) -> None:
        current_tf = self._get_transforms([0.0] * self.dof)
        for idx, link in enumerate(self.link_names):
            coll_ht = self.collision_origins[idx]
            ht0 = current_tf[link].matrix()
            self.meshes[idx].transform(ht0 @ coll_ht)

    def _get_transforms(self, q: np.ndarray) -> Dict[str, kinpy.Transform]:
        transforms = self.chain.forward_kinematics(q)
        return transforms

    def _extract_urdf_data(
        self, urdf: str
    ) -> Tuple[List[str], List[str], List[np.ndarray]]:
        paths = []
        names = []
        origins = []

        def handle_package_path(package: str, filename: str):
            package_path = get_package_share_directory(package)
            return os.path.join(package_path, filename)

        robot = ET.fromstring(urdf)
        for link in robot.findall("link"):
            visual = link.find("collision")
            if visual is not None:
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

                origin = visual.find("origin")
                if origin is not None:
                    xyz = origin.attrib.get("xyz", "0 0 0").split()
                    xyz = np.array([float(x) for x in xyz])
                    rpy = origin.attrib.get("rpy", "0 0 0").split()
                    rpy = [float(x) for x in rpy]
                    collision_origin = tf.euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")
                    collision_origin[:3, 3] = xyz
                    origins.append(collision_origin)
        return paths, names, origins

    @staticmethod
    def _load_mesh(path: str, convex_hull: bool = False) -> o3d.t.geometry.TriangleMesh:
        print(f"Loading mesh from {path}")
        if not (path.endswith(".stl") or path.endswith(".STL")):
            raise NotImplementedError(f"File type {path} not supported yet.")
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(path))
        if convex_hull:
            mesh = mesh.compute_convex_hull()
        mesh = mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def _load_meshes(
        paths: List[str], convex_hull: bool = False
    ) -> List[o3d.t.geometry.TriangleMesh]:
        meshes = [O3DRobot._load_mesh(path, convex_hull) for path in paths]
        return meshes

    @staticmethod
    def _load_chain(urdf: str) -> Chain:
        chain = kinpy.build_chain_from_urdf(urdf)
        return chain
