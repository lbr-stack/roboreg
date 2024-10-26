import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import numpy as np

from roboreg.util.points import clean_xyz


def test_clean_xyz() -> None:
    # no batch dimension
    points = np.random.rand(256, 256, 3)
    mask = np.random.rand(256, 256)
    mask = mask > 0.5

    clean_points = clean_xyz(xyz=points, mask=mask)
    if clean_points.shape[0] != mask.sum():
        raise ValueError(
            f"Expected {mask.sum()} points, got {clean_points.shape[0]} points."
        )
    if clean_points.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {clean_points.shape[1]} channels.")


if __name__ == "__main__":
    test_clean_xyz()
