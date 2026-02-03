import pytest
import pyvista as pv
import torch

from roboreg.core import TorchKinematics, TorchMeshContainer
from roboreg.io import URDFParser, apply_mesh_origins, load_meshes
from roboreg.util import RegistrationVisualizer, from_homogeneous


@pytest.mark.skip(reason="To be fixed.")
def test_visualize_point_cloud():
    meshes = TorchMeshContainer(
        meshes=load_meshes(
            {
                "link_0": "test/assets/lbr_med7_r800/description/meshes/collision/link_0.stl"
            }
        ),
        device="cpu",
    )

    # visialize point cloud
    points = from_homogeneous(meshes.vertices).numpy().squeeze()

    pl = pv.Plotter()

    # black background
    pl.background_color = [0, 0, 0]

    pl.add_points(points)
    pl.show(auto_close=False)
    pl.close()


@pytest.mark.skip(reason="To be fixed.")
def test_visalize_multi_color_point_cloud():
    meshes = TorchMeshContainer(
        meshes=load_meshes(
            {
                "link_0": "test/assets/lbr_med7_r800/description/meshes/collision/link_0.stl",
                "link_1": "test/assets/lbr_med7_r800/description/meshes/collision/link_1.stl",
            }
        ),
        device="cpu",
    )

    # visialize point cloud
    link_points = {}
    link_colors = {}
    for link_name in meshes.names:
        print(link_name)
        link_points[link_name] = (
            from_homogeneous(
                meshes.vertices[
                    :,
                    meshes.lower_vertex_index_lookup[
                        link_name
                    ] : meshes.upper_vertex_index_lookup[link_name],
                ]
            )
            .numpy()
            .squeeze()
        )

        link_colors[link_name] = torch.ones(
            link_points[link_name].shape[0], 4, dtype=torch.float32
        )

    link_colors[meshes.names[0]][..., 0] = 0.0
    link_colors[meshes.names[1]][..., 1] = 0.0

    pl = pv.Plotter()

    # black background
    pl.background_color = [0, 0, 0]

    for link_name in meshes.names:
        pl.add_points(
            link_points[link_name],
            rgba=True,
            scalars=link_colors[link_name],
            show_scalar_bar=False,
        )
    pl.show(auto_close=False)
    pl.close()


@pytest.mark.skip(reason="To be fixed.")
def test_visualize_robot():
    device = "cpu"
    root_link_name = "lbr_link_0"
    end_link_name = "lbr_link_7"

    # parse URDF
    urdf_parser = URDFParser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")

    # load meshes
    meshes = TorchMeshContainer(
        meshes=apply_mesh_origins(
            meshes=load_meshes(
                urdf_parser.mesh_paths_from_ros_registry(
                    root_link_name=root_link_name, end_link_name=end_link_name
                )
            ),
            origins=urdf_parser.mesh_origins(
                root_link_name=root_link_name, end_link_name=end_link_name
            ),
        ),
        device=device,
    )

    # instantiate kinematics
    kinematics = TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )

    # calculate kinematics with random joint state
    torch.manual_seed(42)
    q = torch.rand(kinematics.chain.n_joints)
    q_min, q_max = kinematics.chain.get_joint_limits()
    q_min, q_max = torch.tensor(q_min, device=device), torch.tensor(
        q_max, device=device
    )
    q = q_min + (q_max - q_min) * q

    ht_lookup = kinematics.forward_kinematics(q)

    # apply kinematics
    clone_vertices = meshes.vertices.clone()
    for link_name, ht in ht_lookup.items():
        clone_vertices[
            :,
            meshes.lower_vertex_index_lookup[
                link_name
            ] : meshes.upper_vertex_index_lookup[link_name],
        ] = torch.matmul(
            clone_vertices[
                :,
                meshes.lower_vertex_index_lookup[
                    link_name
                ] : meshes.upper_vertex_index_lookup[link_name],
            ],
            ht.transpose(-1, -2),
        )

    # visialize point cloud
    points = from_homogeneous(clone_vertices).numpy().squeeze()

    pl = pv.Plotter()

    # black background
    pl.background_color = [0, 0, 0]

    pl.add_points(points)
    pl.show(auto_close=False)
    pl.close()


@pytest.mark.skip(reason="To be fixed.")
def test_registration_visualizer() -> None:
    visualizer = RegistrationVisualizer()

    visualizer(
        mesh_vertices=[torch.rand(100, 3), torch.rand(100, 3)],
        observed_vertices=[torch.rand(100, 3), torch.rand(100, 3)],
        HT=None,
    )


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_visualize_point_cloud()
    test_visalize_multi_color_point_cloud()
    test_visualize_robot()
    test_registration_visualizer()
