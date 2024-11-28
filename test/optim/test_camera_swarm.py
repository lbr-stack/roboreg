import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import torch

from roboreg.optim import CameraSwarmOptimizer


def test_camera_swarm_optimizer() -> None:
    cam_swarm = CameraSwarmOptimizer()
    cam_swarm._fitness_function(torch.rand(10, 7))


if __name__ == "__main__":
    test_camera_swarm_optimizer()
