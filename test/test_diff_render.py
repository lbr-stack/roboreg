import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import cv2
import kinpy
import numpy as np
import torch
import transformations as tf
import trimesh
import xacro
from ament_index_python import get_package_share_directory

from roboreg.differentiable.data_structures import TorchRobotMesh
from roboreg.differentiable.renderer import NVDiffRastRenderer


class FK:
    def __init__(self) -> None:
        self.urdf = xacro.process(
            os.path.join(
                get_package_share_directory("lbr_description"), "urdf/med7/med7.xacro"
            )
        )
        self.chain = kinpy.build_chain_from_urdf(self.urdf)
        self.joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(self.joint_names)
        self.paths, self.link_names, self.collision_origins = self._extract_urdf_data(
            self.urdf
        )
        self.q_current = np.zeros(7)

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

    def initial_meshes_transform(self) -> List[np.ndarray]:
        current_tf = self.get_transforms([0.0] * self.dof)
        hts = []
        for idx, link in enumerate(self.link_names):
            coll_ht = self.collision_origins[idx]
            ht0 = current_tf[link].matrix()
            hts.append(ht0 @ coll_ht)
        return hts

    def get_transforms(self, q: np.ndarray) -> Dict[str, kinpy.Transform]:
        transforms = self.chain.forward_kinematics(q)
        return transforms

    def compute(self, q: np.ndarray) -> List[np.ndarray]:
        current_tf = self.get_transforms(self.q_current)
        tf_dict = self.get_transforms(q)
        hts = []

        for link in self.link_names:
            # zero transform
            ht0 = current_tf[link].matrix()

            # desired transform
            ht = tf_dict[link].matrix()
            hts.append(ht @ np.linalg.inv(ht0))
        self.q_current = q
        return hts


def test_pipeline() -> None:
    meshes: List[trimesh.Geometry] = [
        trimesh.load(f"test/data/lbr_med7/mesh/link_{idx}.stl") for idx in range(8)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_robot_mesh = TorchRobotMesh(meshes=meshes, device=device)

    try:
        cnt = 0

        #### init transforms !!!!!!!!!
        fk = FK()
        HTS = fk.initial_meshes_transform()
        # for idx, ht in enumerate(HTS):
        #     if idx < 7:
        #         torch_robot_mesh.link_vertices = torch.matmul(
        #             torch_robot_mesh.link_vertices(idx),
        #             torch.from_numpy(ht).float().to(device).t(),
        #         )
        q = np.random.rand(7) * 2 - 1
        print(q)
        HTS = fk.compute(q)

        # apply HTS to pos

        # pos = torch.matmul(pos, DUMMY_HT.t())
        for idx, ht in enumerate(HTS):
            if idx < 7:
                torch_robot_mesh.set_link_vertices(
                    idx,
                    torch.matmul(
                        torch_robot_mesh.get_link_vertices(idx),
                        torch.from_numpy(ht).float().to(device).t(),
                    ),
                )

        DUMMY_HT = torch.from_numpy(
            tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32")
        ).to(device)
        DUMMY_HT[3, 2] = 0.1
        torch_robot_mesh.vertices = torch.matmul(
            torch_robot_mesh.vertices, DUMMY_HT.t()
        )

        render = NVDiffRastRenderer(device=device)

        while True:
            cnt += 1
            DUMMY_HT = torch.from_numpy(
                tf.euler_matrix(0.0, 0.05, 0.0).astype("float32")
            ).to(device)
            torch_robot_mesh.vertices = torch.matmul(
                torch_robot_mesh.vertices, DUMMY_HT.t()
            )
            color = render.constant_color(
                torch_robot_mesh.vertices,
                torch_robot_mesh.faces,
                [512, 512],
                [1.0, 1.0, 0.0],
            )

            # display render
            color = color.detach().cpu().numpy()
            cv2.imshow("render", color[0])
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass


def test_nvdiffrast_renderer() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meshes: List[trimesh.Geometry] = [
        trimesh.load(f"test/data/lbr_med7/mesh/link_{idx}.stl") for idx in range(8)
    ]
    torch_robot_mesh = TorchRobotMesh(meshes=meshes, device=device)
    renderer = NVDiffRastRenderer(device=device)

    # transform mesh to it becomes visible
    HT_TARGET = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )
    torch_robot_mesh.vertices = torch.matmul(torch_robot_mesh.vertices, HT_TARGET.T)

    # create a target render
    resolution = [256, 256]
    target_render = renderer.constant_color(
        torch_robot_mesh.vertices, torch_robot_mesh.faces, resolution
    )

    # modify transform
    HT = torch.tensor(
        tf.euler_matrix(0.0, 0.0, np.pi / 16.0).astype("float32"),
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )

    # create an optimizer an optimize HT -> HT_TARGET
    optimizer = torch.optim.Adam([HT], lr=0.001)
    metric = torch.nn.MSELoss()
    try:
        for i in range(1000):
            vertices = torch.matmul(torch_robot_mesh.vertices, HT.T)
            current_render = renderer.constant_color(
                vertices, torch_robot_mesh.faces, resolution
            )
            loss = metric(current_render, target_render)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize
            current_image = current_render.squeeze().cpu().detach().numpy()
            target_image = target_render.squeeze().cpu().numpy()
            difference_image = current_image - target_image
            concatenated_image = np.concatenate(
                [target_image, current_image, difference_image], axis=-1
            )
            cv2.imshow(
                "target render / current render / difference", concatenated_image
            )
            cv2.waitKey(0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # test_pipeline()
    test_nvdiffrast_renderer()
