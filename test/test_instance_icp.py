import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from common import load_data

from roboreg.instance_icp import InstanceICP
from roboreg.o3d_robot import O3DRobot


def test_instance_icp():
    observed_xyzs, mesh_xyzs = load_data(idcs=[0, 1], scan=False, visualize=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to numpy
    for i in range(len(observed_xyzs)):
        observed_xyzs[i] = torch.from_numpy(observed_xyzs[i]).to(
            dtype=torch.float32, device=device
        )
        mesh_xyzs[i] = torch.from_numpy(mesh_xyzs[i]).to(
            dtype=torch.float32, device=device
        )

    instance_icp = InstanceICP()
    HT = instance_icp(observed_xyzs, mesh_xyzs)


if __name__ == "__main__":
    test_instance_icp()
