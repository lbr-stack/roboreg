import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roboreg.differentiable.data_structures import TorchRobotMesh


def test_torch_robot_mesh() -> None:
    torch_robot_mesh = TorchRobotMesh(
        ["test/data/lbr_med7/mesh/link_0.stl", "test/data/lbr_med7/mesh/link_1.stl"]
    )
    print(torch_robot_mesh.per_link_vertex_count)
    print(torch_robot_mesh.lower_indices)
    print(torch_robot_mesh.upper_indices)
    print(torch_robot_mesh.get_link_vertices(0))
    print(torch_robot_mesh.get_link_vertices(1))


if __name__ == "__main__":
    test_torch_robot_mesh()
