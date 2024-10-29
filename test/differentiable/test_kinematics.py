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
from roboreg.util import from_homogeneous


def test_torch_kinematics(
    ros_package: str = "lbr_description",
    xacro_path: str = "urdf/med7/med7.xacro",
    root_link_name: str = "lbr_link_0",
    end_link_name: str = "lbr_link_7",
) -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )
    q = torch.zeros([1, 7], device=device)

    # copy q (batch)
    q = torch.cat([q, q], dim=0)
    ht_lookup = kinematics.mesh_forward_kinematics(q)
    print(ht_lookup)


def test_torch_kinematics_on_mesh(
    ros_package: str = "lbr_description",
    xacro_path: str = "urdf/med7/med7.xacro",
    root_link_name: str = "lbr_link_0",
    end_link_name: str = "lbr_link_7",
) -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=ros_package, xacro_path=xacro_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )
    meshes = TorchMeshContainer(
        urdf_parser.ros_package_mesh_paths(root_link_name, end_link_name), device=device
    )

    # compute forward kinematics and apply transforms to the meshes
    q = torch.zeros([1, kinematics.chain.n_joints], device=device)
    q[:, 1] = torch.pi / 2.0
    q[:, 3] = torch.pi / 2.0
    ht_lookup = kinematics.mesh_forward_kinematics(q)

    # apply transforms to the meshes
    vertices = meshes.vertices.clone()
    for link_name, ht in ht_lookup.items():
        vertices[
            :,
            meshes.lower_vertex_index_lookup[
                link_name
            ] : meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            vertices[
                :,
                meshes.lower_vertex_index_lookup[
                    link_name
                ] : meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )

    def test_display_xyz(vertices: torch.Tensor) -> None:
        import pyvista as pv

        pl = pv.Plotter()
        pl.background_color = [0, 0, 0]
        pl.add_points(vertices, point_size=1)
        pl.show()

    test_display_xyz(from_homogeneous(vertices.cpu().numpy())[0])


def test_diff_kinematics() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kinematics = TorchKinematics(
        urdf_parser=urdf_parser,
        root_link_name="lbr_link_0",
        end_link_name="lbr_link_7",
        device=device,
    )
    meshes = TorchMeshContainer(
        urdf_parser.ros_package_mesh_paths("lbr_link_0", "lbr_link_7"), device=device
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
    target_vertices = meshes.vertices.clone()
    for link_name, ht in ht_lookup.items():
        target_vertices[
            :,
            meshes.lower_vertex_index_lookup[
                link_name
            ] : meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            target_vertices[
                :,
                meshes.lower_vertex_index_lookup[
                    link_name
                ] : meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )

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
        dtype=torch.float32,
        device=device,
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
                    meshes.lower_vertex_index_lookup[
                        link_name
                    ] : meshes.upper_vertex_index_lookup[link_name],
                ] = torch.matmul(
                    current_vertices[
                        :,
                        meshes.lower_vertex_index_lookup[
                            link_name
                        ] : meshes.upper_vertex_index_lookup[link_name],
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


if __name__ == "__main__":
    # test_torch_kinematics(
    #     ros_package="lbr_description",
    #     xacro_path="urdf/med7/med7.xacro",
    #     root_link_name="lbr_link_0",
    #     end_link_name="lbr_link_7",
    # )
    test_torch_kinematics_on_mesh(
        ros_package="lbr_description",
        xacro_path="urdf/med7/med7.xacro",
        root_link_name="lbr_link_0",
        end_link_name="lbr_link_7",
    )
    test_torch_kinematics_on_mesh(
        ros_package="xarm_description",
        xacro_path="urdf/xarm_device.urdf.xacro",
        root_link_name="link_base",
        end_link_name="link7",
    )
    # test_diff_kinematics()
