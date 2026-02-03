import argparse
from pathlib import Path


def validate_urdf_source(parser: argparse.ArgumentParser, args: argparse.Namespace):
    r"""Validate mutually exclusive URDF source options."""
    urdf_provided = args.urdf_path is not None
    ros_provided = args.ros_package is not None
    xacro_provided = args.xacro_path is not None

    # check if both methods provided
    if urdf_provided and (ros_provided or xacro_provided):
        parser.error(
            "Cannot specify --urdf-path together with --ros-package or --xacro-path. "
            "Use either --urdf-path OR (--ros-package + --xacro-path)."
        )

    # check if ROS method incomplete
    if ros_provided and not xacro_provided:
        parser.error("--ros-package requires --xacro-path")

    if xacro_provided and not ros_provided:
        parser.error("--xacro-path requires --ros-package")

    # check if nothing provided
    if not urdf_provided and not ros_provided:
        parser.error(
            "Must specify URDF source: either --urdf-path OR "
            "(--ros-package + --xacro-path)"
        )

    # validate file exists if using urdf-path
    if urdf_provided:
        urdf_path = Path(args.urdf_path)
        if not urdf_path.exists():
            parser.error(f"URDF file not found: {urdf_path}")
