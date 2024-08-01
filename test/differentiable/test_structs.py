import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from roboreg.differentiable.structs import TorchMeshContainer


def test_torch_robot_mesh() -> None:
    torch_robot_mesh = TorchMeshContainer(
        mesh_paths={
            "link_0": "test/data/lbr_med7/mesh/link_0.stl",
            "link_1": "test/data/lbr_med7/mesh/link_1.stl",
        }
    )
    print(torch_robot_mesh.per_mesh_vertex_count)
    print(torch_robot_mesh.lower_index_lookup)
    print(torch_robot_mesh.upper_index_lookup)
    print(torch_robot_mesh.get_mesh_vertices("link_0").shape)
    print(torch_robot_mesh.get_mesh_vertices("link_1").shape)


if __name__ == "__main__":
    test_torch_robot_mesh()
