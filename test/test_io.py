import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roboreg.io import URDFParser


def test_urdf_parser() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")
    urdf_parser.link_names("link_0", "link_ee")
    print(urdf_parser.robot.links)


if __name__ == "__main__":
    test_urdf_parser()
