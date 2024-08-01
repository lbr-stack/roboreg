import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import cv2
import numpy as np
import torch
import transformations as tf
from tqdm import tqdm

from roboreg.differentiable.kinematics import TorchKinematics
from roboreg.differentiable.rendering import NVDiffRastRenderer
from roboreg.differentiable.structs import TorchMeshContainer
from roboreg.io import URDFParser


def test_torch_kinematics() -> None:
    urdf_parser = URDFParser()
    urdf = urdf_parser.urdf_from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf=urdf, root_link_name="link_0", end_link_name="link_ee", device=device
    )
    q = torch.zeros([1, 7], device=device)

    # copy q (batch)
    q = torch.cat([q, q], dim=0)
    ht_lookup = kinematics.mesh_forward_kinematics(q)
    print(ht_lookup)


def test_torch_kinematics_on_mesh() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name="link_0",
        end_link_name="link_7",
        device=device,
    )
    meshes = TorchMeshContainer(
        urdf_parser.ros_package_mesh_paths("link_0", "link_7"), device=device
    )

    # compute forward kinematics and apply transforms to the meshes
    q = torch.zeros([1, 7], device=device)
    q[:, 1] = torch.pi / 2.0
    q[:, 3] = torch.pi / 2.0
    ht_lookup = kinematics.mesh_forward_kinematics(q)

    # apply transforms to the meshes
    for link_name, ht in ht_lookup.items():
        meshes.transform_mesh(ht, link_name)

    # render the scene for reference
    # transform view
    ht_view = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )
    meshes.vertices = torch.matmul(meshes.vertices, ht_view.T)

    renderer = NVDiffRastRenderer(device=device)
    render = renderer.constant_color(meshes.vertices, meshes.faces, [256, 256])

    cv2.imshow("render", render.detach().cpu().numpy().squeeze())
    cv2.waitKey(0)


def test_diff_kinematics() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name="link_0",
        end_link_name="link_7",
        device=device,
    )
    meshes = TorchMeshContainer(
        urdf_parser.ros_package_mesh_paths("link_0", "link_7"), device=device
    )

    # compute forward kinematics and apply transforms to the meshes
    q_target = torch.zeros(
        [1, kinematics.chain.n_joints],
        device=device,
    )
    q_target[:, 1] = torch.pi / 2.0
    q_target[:, 3] = torch.pi / 2.0
    q_target[:, 5] = torch.pi / 4.0

    ht_lookup = kinematics.mesh_forward_kinematics(q_target)
    for link_name, ht in ht_lookup.items():
        meshes.transform_mesh(ht, link_name)

    target_vertices = meshes.vertices.clone()

    # revert transforms
    for link_name, ht in ht_lookup.items():
        meshes.transform_mesh(torch.linalg.inv(ht), link_name)

    metric = torch.nn.MSELoss()
    q_current = torch.full(
        [1, kinematics.chain.n_joints],
        torch.pi / 2.0,
        device=device,
        requires_grad=True,
    )
    optim = torch.optim.Adam([q_current], lr=0.1)

    # render for visualization
    ht_view = torch.tensor(
        tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32"),
        device=device,
        dtype=torch.float32,
    )

    resolution = [512, 512]
    renderer = NVDiffRastRenderer(device=device)
    target_render = (
        renderer.constant_color(
            torch.matmul(target_vertices, ht_view.T), meshes.faces, resolution
        )
        .cpu()
        .numpy()
        .squeeze()
    )

    # run optimization such that q_current -> q_target
    try:
        for _ in tqdm(range(200)):
            ht_lookup = kinematics.mesh_forward_kinematics(q_current)
            current_vertices = meshes.vertices.clone()
            for link_name, ht in ht_lookup.items():
                current_vertices[
                    :,
                    meshes.lower_index_lookup[link_name] : meshes.upper_index_lookup[
                        link_name
                    ],
                ] = torch.matmul(
                    current_vertices[
                        :,
                        meshes.lower_index_lookup[
                            link_name
                        ] : meshes.upper_index_lookup[link_name],
                    ].clone(),
                    ht.transpose(-1, -2),
                )

            loss = metric(current_vertices, target_vertices)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # print(f"loss: {loss}, q_current: {q_current}")

            current_render = (
                renderer.constant_color(
                    torch.matmul(current_vertices, ht_view.T), meshes.faces, resolution
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            difference_render = current_render - target_render

            cv2.imshow(
                "target joint states / current joint states / difference",
                np.concatenate(
                    [target_render, current_render, difference_render], axis=-1
                ),
            )
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass


def test_joint_offset() -> None:
    urdf_parser = URDFParser()
    urdf = urdf_parser.urdf_from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf=urdf,
        root_link_name="link_0",
        end_link_name="link_7",
        device=device,
    )

    for link_name in kinematics.chain.get_link_names():
        print(link_name)
        print(kinematics.chain.get_frame_indices(link_name))


if __name__ == "__main__":
    # test_torch_kinematics()
    # test_torch_kinematics_on_mesh()
    test_diff_kinematics()
    # test_joint_offset()
