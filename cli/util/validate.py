from pathlib import Path
from typing import Optional

import typer


def validate_urdf_source(
    urdf_path: Optional[Path],
    ros_package: Optional[str],
    xacro_path: Optional[str],
) -> None:
    r"""Validate mutually exclusive URDF source options."""
    urdf_provided = urdf_path is not None
    ros_provided = ros_package is not None
    xacro_provided = xacro_path is not None

    def _error(message: str) -> None:
        typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(code=2)

    # check if both methods provided
    if urdf_provided and (ros_provided or xacro_provided):
        _error(
            "Cannot specify --urdf-path together with --ros-package or --xacro-path. "
            "Use either --urdf-path OR (--ros-package + --xacro-path)."
        )

    # check if ROS method incomplete
    if ros_provided and not xacro_provided:
        _error("--ros-package requires --xacro-path")

    if xacro_provided and not ros_provided:
        _error("--xacro-path requires --ros-package")

    # check if nothing provided
    if not urdf_provided and not ros_provided:
        _error(
            "Must specify URDF source: either --urdf-path OR "
            "(--ros-package + --xacro-path)"
        )

    # validate file exists if using urdf-path
    if urdf_provided and not urdf_path.exists():
        _error(f"URDF file not found: {urdf_path}")
