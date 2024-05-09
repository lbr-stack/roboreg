import os

from roboreg.util import parse_camera_info


def test_parse_camera_info() -> None:
    path = "test/data/lbr_med7/high_res"
    file = "left_camera_info.yaml"
    height, width, intrinsic_matrix = parse_camera_info(os.path.join(path, file))

    print(height)
    print(width)
    print(intrinsic_matrix)


if __name__ == "__main__":
    test_parse_camera_info()
