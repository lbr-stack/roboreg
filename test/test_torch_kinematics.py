import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import xacro
from ament_index_python import get_package_share_directory

from roboreg.differentiable.kinematics import TorchKinematics


def test_torch_robot_mesh() -> None:
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.xacro"
        )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_kinematics = TorchKinematics(
        urdf=urdf, end_link_name="link_ee", root_link_name="link_0", device=device
    )
    q = torch.zeros(7, device=device).unsqueeze(0)

    # copy q (batch)
    q = torch.cat([q, q], dim=0)
    result = torch_kinematics.forward_kinematics(q)
    print(result)


if __name__ == "__main__":
    test_torch_robot_mesh()
