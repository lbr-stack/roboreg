import torch

from roboreg.core import Robot, TorchKinematics, TorchMeshContainer
from roboreg.io import load_robot_data_from_urdf_file


def test_robot() -> None:
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot_data = load_robot_data_from_urdf_file(
        urdf_path="test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf",
        collision=True,
    )

    mesh_container = TorchMeshContainer(
        meshes=robot_data.meshes,
        batch_size=batch_size,
        device=device,
    )
    kinematics = TorchKinematics(
        urdf=robot_data.urdf,
        root_link_name=robot_data.root_link_name,
        end_link_name=robot_data.end_link_name,
        device=device,
    )
    robot = Robot(
        mesh_container=mesh_container,
        kinematics=kinematics,
    )

    assert robot.device == torch.device(device), "Robot device mismatch."
    assert (
        robot.configured_vertices.shape == robot.mesh_container.vertices.shape
    ), "Configured vertices shape mismatch."
    q = torch.zeros(batch_size, robot.kinematics.chain.n_joints, device=device)
    robot.configure(q=q)
    try:
        q = torch.zeros(batch_size - 1, robot.kinematics.chain.n_joints, device=device)
        robot.configure(q=q)
    except ValueError:
        pass


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_robot()
