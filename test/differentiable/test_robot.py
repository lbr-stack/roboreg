import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import torch

from roboreg.differentiable import Robot
from roboreg.io import URDFParser


def test_robot() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    visual = False
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot = Robot(
        urdf_parser=urdf_parser,
        root_link_name=urdf_parser.link_names_with_meshes(visual=visual)[0],
        end_link_name=urdf_parser.link_names_with_meshes(visual=visual)[-1],
        visual=visual,
        batch_size=batch_size,
        device=device,
        target_reduction=0.0,
    )

    q = torch.zeros(batch_size, robot.kinematics.chain.n_joints, device=device)
    robot.configure(q=q)
    print(robot.configured_vertices)


if __name__ == "__main__":
    test_robot()
