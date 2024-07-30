import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from roboreg import differentiable as rrd
from roboreg.io import URDFParser


def test_pipeline() -> None:
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro("lbr_description", "urdf/med7/med7.xacro")

    root_link_name = "link_0"
    end_link_name = "link_ee"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate scene objects
    mesh = rrd.structs.TorchRobotMesh(
        mesh_paths=urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name, end_link_name=end_link_name
        ),
        device=device,
    )
    kinematics = rrd.kinematics.TorchKinematics(
        urdf=urdf_parser.urdf,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        device=device,
    )
    renderer = rrd.rendering.NVDiffRastRenderer(device=device)

    # generate scene
    scene = rrd.scene.RobotScene(mesh=mesh, kinematics=kinematics, renderer=renderer)

    # # parse mesh as torch robot mesh: read mesh paths from urdf

    # # instantiate kinematics
    # self._kinematics = TorchKinematics(
    #     urdf=urdf,
    #     root_link_name=root_link_name,
    #     end_link_name=end_link_name,
    #     device=device,
    # )

    # # instantiate renderer
    # urdf_parser = URDFParser()
    # urdf_parser.from_urdf(urdf=urdf)

    # self._mesh_paths = []
    # self._mesh = TorchRobotMesh(self._mesh_paths)
    # # intrinsics: torch.Tensor,

    # # intrinsics: torch.Tensor,
    # # camera_pose: torch.Tensor,
    # pass

    # mesh_paths: List[trimesh.Geometry] = [
    #     trimesh.load(f"test/data/lbr_med7/mesh/link_{idx}.stl") for idx in range(8)
    # ]

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch_robot_mesh = TorchRobotMesh(mesh_paths=mesh_paths, device=device)

    # try:
    #     cnt = 0

    #     #### init transforms !!!!!!!!!
    #     fk = FK()
    #     HTS = fk.initial_meshes_transform()
    #     # for idx, ht in enumerate(HTS):
    #     #     if idx < 7:
    #     #         torch_robot_mesh.link_vertices = torch.matmul(
    #     #             torch_robot_mesh.link_vertices(idx),
    #     #             torch.from_numpy(ht).float().to(device).t(),
    #     #         )
    #     q = np.random.rand(7) * 2 - 1
    #     print(q)
    #     HTS = fk.compute(q)

    #     # apply HTS to pos

    #     # pos = torch.matmul(pos, DUMMY_HT.t())
    #     for idx, ht in enumerate(HTS):
    #         if idx < 7:
    #             torch_robot_mesh.set_link_vertices(
    #                 idx,
    #                 torch.matmul(
    #                     torch_robot_mesh.get_link_vertices(idx),
    #                     torch.from_numpy(ht).float().to(device).t(),
    #                 ),
    #             )

    #     DUMMY_HT = torch.from_numpy(
    #         tf.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32")
    #     ).to(device)
    #     DUMMY_HT[3, 2] = 0.1
    #     torch_robot_mesh.vertices = torch.matmul(
    #         torch_robot_mesh.vertices, DUMMY_HT.t()
    #     )

    #     render = NVDiffRastRenderer(device=device)

    #     while True:
    #         cnt += 1
    #         DUMMY_HT = torch.from_numpy(
    #             tf.euler_matrix(0.0, 0.05, 0.0).astype("float32")
    #         ).to(device)
    #         torch_robot_mesh.vertices = torch.matmul(
    #             torch_robot_mesh.vertices, DUMMY_HT.t()
    #         )
    #         color = render.constant_color(
    #             torch_robot_mesh.vertices,
    #             torch_robot_mesh.faces,
    #             [512, 512],
    #             [1.0, 1.0, 0.0],
    #         )

    #         # display render
    #         color = color.detach().cpu().numpy()
    #         cv2.imshow("render", color[0])
    #         cv2.waitKey(1)
    # except KeyboardInterrupt:
    #     pass


if __name__ == "__main__":
    test_pipeline()
