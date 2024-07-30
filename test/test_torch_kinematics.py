import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from roboreg.differentiable.kinematics import TorchKinematics
from roboreg.io import URDFParser


def test_torch_robot_mesh() -> None:
    urdf_parser = URDFParser()
    urdf = urdf_parser.urdf_from_ros_xacro(
        ros_package="lbr_description", xacro_path="urdf/med7/med7.xacro"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_kinematics = TorchKinematics(
        urdf=urdf, root_link_name="link_0", end_link_name="link_ee", device=device
    )
    q = torch.zeros(7, device=device).unsqueeze(0)

    # copy q (batch)
    q = torch.cat([q, q], dim=0)
    result = torch_kinematics.forward_kinematics(q)
    print(result)


if __name__ == "__main__":
    test_torch_robot_mesh()
