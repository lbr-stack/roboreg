import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roboreg.io import URDFParser, find_files, parse_camera_info, parse_hydra_data


def test_urdf_parser() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")
    print(urdf_parser.chain_link_names("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.raw_mesh_paths("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.ros_package_mesh_paths("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.mesh_origins("lbr_link_0", "lbr_link_ee"))
    print(urdf_parser.link_names_with_meshes(visual=False))
    print(urdf_parser.link_names_with_meshes(visual=True))


def test_find_files() -> None:
    path = "test/data/lbr_med7/zed2i"
    for mask_file in find_files(path, "mask_sam2_left_*.png"):
        print(mask_file)


def test_parse_camera_info() -> None:
    path = "test/data/lbr_med7/zed2i"
    file = "left_camera_info.yaml"
    height, width, intrinsic_matrix = parse_camera_info(os.path.join(path, file))

    print(height)
    print(width)
    print(intrinsic_matrix)


def test_parse_hydra_data() -> None:
    joint_states, masks, depths = parse_hydra_data(
        "test/data/lbr_med7/zed2i",
        joint_states_pattern="joint_states_*.npy",
        mask_pattern="mask_sam2_left_*.png",
        depth_pattern="depth_*.npy",
    )
    print(len(joint_states))
    print(len(masks))
    print(len(depths))
    print(joint_states[0].shape)
    print(masks[0].shape)
    print(depths[0].shape)


if __name__ == "__main__":
    test_urdf_parser()
    test_find_files()
    test_parse_camera_info()
    test_parse_hydra_data()
