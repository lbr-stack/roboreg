import pytest
import torch

from roboreg.differentiable import Robot
from roboreg.io import URDFParser


@pytest.mark.skip(reason="To be fixed.")
def test_robot() -> None:
    urdf_parser = URDFParser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    collision = True
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot = Robot.from_urdf_parser(
        urdf_parser=urdf_parser,
        root_link_name=urdf_parser.link_names_with_meshes(collision=collision)[0],
        end_link_name=urdf_parser.link_names_with_meshes(collision=collision)[-1],
        collision=collision,
        batch_size=batch_size,
        device=device,
        target_reduction=0.0,
    )

    q = torch.zeros(batch_size, robot.kinematics.chain.n_joints, device=device)
    robot.configure(q=q)
    print(robot.configured_vertices)


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_robot()
