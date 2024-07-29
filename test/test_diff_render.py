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
from torch import FloatTensor

import nvdiffrast.torch as dr


class DiffBot:
    r"""Differentiable robot."""

    def __init__(self, meshes: List[str]) -> None:
        pass

    def render(self, intrinsics: FloatTensor, pose: FloatTensor) -> FloatTensor:
        pass

    def fk(self, q: FloatTensor) -> None:
        pass


class DiffBotModule(torch.nn.Module):
    f"""Differentiable robot module."""

    def __init__(self, meshes: List[str]) -> None:
        super().__init__()
        self._robot = DiffBot(meshes)

    def forward(self, intrinsics: FloatTensor, pose: FloatTensor) -> FloatTensor:
        pass


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


if __name__ == "__main__":
    #### robot:
    ### - render(intrinsics, pose)
    ### - fk(q)

    # Step 0: Load mesh and visualize
    meshes: List[trimesh.Geometry] = [
        trimesh.load(f"test/data/lbr_med7/mesh/link_{idx}.stl") for idx in range(8)
    ]

    # mesh = trimesh.load("test/data/xarm/mesh/link_base.STL")
    # mesh_0.show()

    # apply different transforms per mesh

    # TODO: mesh -> meshes + FK

    # Step 1: Render sample mesh (dae / stl)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = torch.cat(
        [
            torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
            for mesh in meshes
        ],
        dim=0,
    )

    # xyz -> xyz1
    pos = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1)

    # create ranges
    lengths = [len(mesh.vertices) for mesh in meshes]
    lengths = torch.tensor(lengths, device="cpu", dtype=torch.int32)

    # offsets
    offsets = torch.tensor(
        [0] + list(np.cumsum(lengths)[:-1]), device=device, dtype=torch.int32
    )

    print("offsets")
    print(offsets)
    print("lenghts")
    print(lengths)

    # FK: N_links x 4 x 4 and applied to range of vertices
    # # TODO: projections (range mode) including FK
    pos_idx = torch.cat(  ### offset these indices by ranges
        [
            torch.add(
                torch.tensor(meshes[i].faces, device=device, dtype=torch.int32),
                offsets[i],
            )
            for i in range(len(meshes))
        ],
        dim=0,
    )

    # # offsets = torch.cat(
    # #     [
    # #         torch.tensor(
    # #             [offsets[i]] * len(meshes[i].faces),
    # #             device=device,
    # #             dtype=torch.int32,
    # #         )
    # #         for i in range(len(meshes))
    # #     ]
    # # )
    # print("************************+")
    # print(offsets.shape)

    # # replicate offsets Nx1 -> Nx3
    # offsets = torch.cat([offsets.unsqueeze(1)] * 3, dim=1)

    # print("offset")
    # print(pos_idx)

    # pos_idx = pos_idx + offsets.unsqueeze(1)

    ## target resolution, target color,....

    # some dummy transform

    try:
        cnt = 0
        pos = pos.unsqueeze(0)

        #### init transforms
        fk = FK()
        HTS = fk.initial_meshes_transform()
        for idx, ht in enumerate(HTS):
            if idx < 7:
                print(
                    f"applying tf in range {offsets[idx]} : {offsets[idx + 1]}:\n {ht}"
                )
                pos[:, offsets[idx] : offsets[idx + 1]] = torch.matmul(
                    pos[:, offsets[idx] : offsets[idx + 1]],
                    torch.from_numpy(ht).float().to(device).t(),
                )
        q = np.random.rand(7) * 2 - 1
        # q[1] = 0.5
        print(q)

        HTS = fk.compute(q)

        # apply HTS to pos

        # pos = torch.matmul(pos, DUMMY_HT.t())
        for idx, ht in enumerate(HTS):
            if idx < 7:
                print(
                    f"applying tf in range {offsets[idx]} : {offsets[idx + 1]}:\n {ht}"
                )
                pos[:, offsets[idx] : offsets[idx + 1]] = torch.matmul(
                    pos[:, offsets[idx] : offsets[idx + 1]],
                    torch.from_numpy(ht).float().to(device).t(),
                )

        DUMMY_HT = torch.from_numpy(
            tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32")
        ).to(device)
        DUMMY_HT[3, 2] = 0.1
        pos = torch.matmul(pos, DUMMY_HT.t())

        while True:
            cnt += 1
            DUMMY_HT = torch.from_numpy(
                tf.euler_matrix(0.0, 0.05, 0.0).astype("float32")
            ).to(device)
            pos = torch.matmul(pos, DUMMY_HT.t())

            #### apply fk to pos
            # kinpy hacking bullshit

            # pos = torch.matmul(pos, DUMMY_HT.inverse().t())

            ###

            ctx = dr.RasterizeCudaContext(device=device)
            rast, _ = dr.rasterize(
                ctx,
                pos,
                pos_idx,
                resolution=[512, 512],
                # ranges=torch.tensor(
                #     [0, 4038], device="cpu", dtype=torch.int32
                # ).unsqueeze(0),
            )
            col = torch.tensor(
                [[1.0, 1.0, 1.0]], device=device, dtype=torch.float32
            ).unsqueeze(0)
            col_idx = torch.zeros_like(pos_idx)
            color, _ = dr.interpolate(col, rast, col_idx)
            color = dr.antialias(color, rast, pos, pos_idx)

            # display rasterized image

            rast = rast.detach().cpu().numpy()
            cv2.imshow("rasterized", rast[0])
            cv2.waitKey()

            # # display interpolation
            # color = color.detach().cpu().numpy()
            # print(color.shape)
            # cv2.imshow("interpolated", color[0])
            # cv2.waitKey()
    except KeyboardInterrupt:
        pass
