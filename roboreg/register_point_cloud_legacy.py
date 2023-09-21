import os
import time
from typing import Tuple

from trimesh import registration
import cv2
import numpy as np
import pycpd
import pyvista as pv
import xacro
from ament_index_python import get_package_share_directory
from meshify_robot import MeshifyRobot
from simpleicp import PointCloud, SimpleICP


def clean_data(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    idcs = np.isfinite(x)
    x = x[idcs]
    y = y[idcs]
    z = z[idcs]
    return x, y, z


def load_points(
    point_cloud_prefix: str, x_path: str, y_path: str, z_path: str, rgba_path: str
):
    x = np.load(os.path.join(point_cloud_prefix, x_path))
    y = np.load(os.path.join(point_cloud_prefix, y_path))
    z = np.load(os.path.join(point_cloud_prefix, z_path))
    rgba = np.load(os.path.join(point_cloud_prefix, rgba_path))
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

    x, y, z = clean_data(x, y, z)
    return x, y, z, rgb


def numpy_to_pyvista(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    cloud = pv.PolyData(points)
    return cloud


def animate():
    point_cloud_prefix = "/home/martin/Dev/zed_ws/records/point_cloud"
    kinematics_state_prefix = "/home/martin/Dev/zed_ws/records/kinematics"
    idx = 0

    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )
    meshify_robot = MeshifyRobot(urdf)

    step = 10
    for i in range(0, 180, step):
        # process joint states
        q = np.load(os.path.join(kinematics_state_prefix, f"position_{idx}.npy"))
        meshes = meshify_robot.transformed_meshes(q)
        meshify_robot.plot_point_clouds(meshes)

        # process point cloud
        x, y, z, rgb = load_points(
            point_cloud_prefix,
            f"x_{idx}.npy",
            f"y_{idx}.npy",
            f"z_{idx}.npy",
            f"rgba_{idx}.npy",
        )
        cloud = numpy_to_pyvista(x, y, z)

        plotter = pv.Plotter()
        plotter.add_points(cloud, scalars=rgb)
        plotter.show()
        plotter.clear()
        idx += step


def sub_sample(data: np.ndarray, N: int, max_mode: bool = False):
    if max_mode:
        N = max(min(N, data.shape[0]), data.shape[0])
    indices = np.random.choice(data.shape[0], N, replace=False)
    sampled_points = data[indices]
    return sampled_points


# kuka kcl max_depth = 1.0
# kuka kcl high_offset = 0.3


def cut_off(
    data: np.ndarray, max_depth: float = 1.0, hight_offset: float = 0.3
) -> np.ndarray:
    data = data[data[:, 0] < max_depth]  # axes might have different order!!!
    data = data[data[:, 2] > data[:, 2].min() + hight_offset]
    return data


def segment_bounding_box(
    point_cloud: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
):
    x_idcs = np.logical_and(point_cloud[:, 0] >= x_min, point_cloud[:, 0] < x_max)
    y_idcs = np.logical_and(point_cloud[:, 1] >= y_min, point_cloud[:, 1] < y_max)
    z_idcs = np.logical_and(point_cloud[:, 2] >= z_min, point_cloud[:, 2] < z_max)
    idcs = np.logical_and(np.logical_and(x_idcs, y_idcs), z_idcs)
    return point_cloud[idcs]


def visualize(source: np.ndarray, target: np.ndarray) -> None:
    # visualize X, Y
    plotter = pv.Plotter()
    plotter.add_points(
        pv.PolyData(source),
        scalars=np.full_like(source, 255),
    )
    plotter.add_points(
        pv.PolyData(target),
        scalars=np.full_like(target, 0),
    )
    plotter.add_axes_at_origin()
    plotter.show()


def cpd_registration():
    point_cloud_prefix = "/home/martin/Dev/zed_ws/records/point_cloud"
    kinematics_state_prefix = "/home/martin/Dev/zed_ws/records/kinematics"
    idx = 0

    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )
    meshify_robot = MeshifyRobot(urdf)

    step = 10
    idx = 100

    # process joint states
    q = np.load(os.path.join(kinematics_state_prefix, f"position_{idx}.npy"))
    meshes = meshify_robot.transformed_meshes(q)

    # process point cloud
    x, y, z, rgb = load_points(
        point_cloud_prefix,
        f"x_{idx}.npy",
        f"y_{idx}.npy",
        f"z_{idx}.npy",
        f"rgba_{idx}.npy",
    )

    stereo_cloud = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # cut-off (very trivial segmentation)
    stereo_cloud = cut_off(stereo_cloud)

    mesh_cloud = meshify_robot.meshes_to_point_cloud(meshes)
    mesh_cloud = meshify_robot.remove_inner_points(mesh_cloud, alpha=0.1)

    # sub-sample N points
    N = 4000
    stereo_cloud = sub_sample(stereo_cloud, N)
    mesh_cloud = sub_sample(mesh_cloud, N)

    print("start registration")
    registration = pycpd.RigidRegistration(X=mesh_cloud, Y=stereo_cloud)
    start = time.time()
    registration.register()
    Y = registration.transform_point_cloud(registration.Y)
    print(f"registration took {time.time() - start} seconds")

    # visualize X, Y
    visualize(Y, registration.X)


def homogenous_point_cloud_sampling(point_cloud: np.ndarray, N: int) -> np.ndarray:
    # define a bounding box
    min_x, max_x = point_cloud[:, 0].min(), point_cloud[:, 0].max()
    min_y, max_y = point_cloud[:, 1].min(), point_cloud[:, 1].max()
    min_z, max_z = point_cloud[:, 2].min(), point_cloud[:, 2].max()

    # sample points
    x = np.random.uniform(min_x, max_x, N)
    y = np.random.uniform(min_y, max_y, N)
    z = np.random.uniform(min_z, max_z, N)

    # find corresponding points in point cloud
    sampled_points = []
    for i in range(N):
        dist = np.linalg.norm(point_cloud - np.array([x[i], y[i], z[i]]), axis=1)
        idx = np.argmin(dist)
        sampled_points.append(point_cloud[idx])

    return np.array(sampled_points)


def icp_registration():
    prefix = "/media/martin/Samsung_T5/23_07_04_faros_integration_week_measurements/faros_integration_week_kuka_left"
    # parameter
    bounding_box_params = {
        "x_min": 1.0,
        "x_max": 1.5,
        "y_min": -0.5,
        "y_max": 0.5,
        "z_min": -0.34,
        "z_max": 0.5,
    }
    idx = 740

    point_cloud_prefix = f"{prefix}/point_cloud"
    kinematics_state_prefix = f"{prefix}/kinematics"

    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )
    meshify_robot = MeshifyRobot(urdf)

    # process joint states
    q = np.load(os.path.join(kinematics_state_prefix, f"position_{idx}.npy"))
    meshes = meshify_robot.transformed_meshes(q)

    # process point cloud
    x, y, z, rgb = load_points(
        point_cloud_prefix,
        f"x_{idx}.npy",
        f"y_{idx}.npy",
        f"z_{idx}.npy",
        f"rgba_{idx}.npy",
    )

    stereo_cloud = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # cut-off (very trivial segmentation)
    # stereo_cloud = cut_off(stereo_cloud)
    stereo_cloud = segment_bounding_box(stereo_cloud, **bounding_box_params)

    mesh_cloud = meshify_robot.meshes_to_point_cloud(meshes)
    mesh_cloud = mesh_cloud[mesh_cloud[..., 0] < 0]
    # mesh_cloud = meshify_robot.remove_inner_points(mesh_cloud, alpha=0.1)

    # sub-sample N points
    # N = mesh_cloud.shape[0]
    N = 3000
    stereo_cloud = sub_sample(stereo_cloud, N)
    # mesh_cloud = sub_sample(mesh_cloud, N, False)
    mesh_cloud = homogenous_point_cloud_sampling(mesh_cloud, N)
    print(mesh_cloud)

    # move to com
    pre_translation = mesh_cloud.mean(axis=0, keepdims=True) - stereo_cloud.mean(axis=0, keepdims=True)
    stereo_cloud += pre_translation

    H_pre = np.eye(4)
    H_pre[:3, 3] = pre_translation

    print(H_pre)

    visualize(stereo_cloud, mesh_cloud)

    def run_icp(
        pc_fix: PointCloud,
        pc_mov: PointCloud,
        max_iter=100,
        max_overlap_distance=np.inf,
        min_change: float = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        print("start registration")
        icp = SimpleICP()
        icp.add_point_clouds(pc_fix=pc_fix, pc_mov=pc_mov)
        start = time.time()
        H, pc_mov_transformed, rigid_body_tf_params, distance_resituals = icp.run(
            max_iterations=max_iter,
            neighbors=10,
            max_overlap_distance=max_overlap_distance,
            min_change=min_change,
        )
        print(f"registration took {time.time() - start} seconds")
        return H, pc_mov_transformed

    # fine-tune
    H_0, pc_stereo_transformed = run_icp(
        PointCloud(mesh_cloud, columns=["x", "y", "z"]),
        PointCloud(stereo_cloud, columns=["x", "y", "z"]),
        max_iter=100,
        max_overlap_distance=np.inf,
    )
    visualize(pc_stereo_transformed, mesh_cloud)
    H_1, pc_stereo_transformed = run_icp(
        PointCloud(mesh_cloud, columns=["x", "y", "z"]),
        PointCloud(pc_stereo_transformed, columns=["x", "y", "z"]),
        max_iter=100,
        max_overlap_distance=0.2,
    )
    visualize(pc_stereo_transformed, mesh_cloud)
    H_2, pc_stereo_transformed = run_icp(
        PointCloud(mesh_cloud, columns=["x", "y", "z"]),
        PointCloud(pc_stereo_transformed, columns=["x", "y", "z"]),
        max_iter=100,
        max_overlap_distance=0.1,
    )
    visualize(pc_stereo_transformed, mesh_cloud)
    H_3, pc_stereo_transformed = run_icp(
        PointCloud(mesh_cloud, columns=["x", "y", "z"]),
        PointCloud(pc_stereo_transformed, columns=["x", "y", "z"]),
        max_iter=100,
        max_overlap_distance=0.05,
    )
    visualize(pc_stereo_transformed, mesh_cloud)

    print("H_0:\n", H_0)
    print("H_1:\n", H_1)
    print("H_2:\n", H_2)
    print("H_3:\n", H_3)
    H = H_3 @ H_2 @ H_1 @ H_0 @ H_pre
    print(H)

    # write H to file
    np.save("homogeneous_transform.npy", H)

    print(pc_stereo_transformed.shape)
    print(mesh_cloud.shape)


def trimesh_icp():
    prefix = "/media/martin/Samsung_T5/23_07_04_faros_integration_week_measurements/faros_integration_week_kuka_left"
    left_params = {
        "x_min": 1.0,
        "x_max": 1.5,
        "y_min": -0.5,
        "y_max": 0.5,
        "z_min": -0.34,
        "z_max": 0.5,
    }
    point_cloud_prefix = f"{prefix}/point_cloud"
    kinematics_state_prefix = f"{prefix}/kinematics"
    idx = 0

    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )
    meshify_robot = MeshifyRobot(urdf)

    step = 10
    idx = 10
    # idx = 0

    # process joint states
    q = np.load(os.path.join(kinematics_state_prefix, f"position_{idx}.npy"))
    meshes = meshify_robot.transformed_meshes(q)

    # process point cloud
    x, y, z, rgb = load_points(
        point_cloud_prefix,
        f"x_{idx}.npy",
        f"y_{idx}.npy",
        f"z_{idx}.npy",
        f"rgba_{idx}.npy",
    )

    stereo_cloud = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # cut-off (very trivial segmentation)
    # stereo_cloud = cut_off(stereo_cloud)
    stereo_cloud = segment_bounding_box(stereo_cloud, **left_params)

    mesh_cloud = meshify_robot.meshes_to_point_cloud(meshes)

    # sub-sample N points
    # N = mesh_cloud.shape[0]
    N = 3000
    stereo_cloud = sub_sample(stereo_cloud, N)
    # mesh_cloud = sub_sample(mesh_cloud, N, False)
    mesh_cloud = homogenous_point_cloud_sampling(mesh_cloud, N)

    ### registration
    def run_icp(stereo_cloud, mesh_cloud):
        pl = pv.Plotter()
        pl.add_points(pv.PolyData(mesh_cloud), scalars=np.full_like(mesh_cloud, 255))
        pl.show(interactive_update=True)
        for i in range(1, 1000):
            tranform, stereo_cloud, cost = registration.icp(
                stereo_cloud, mesh_cloud, max_iterations=1
            )

            print(cost)

            pl.clear()

            pl.add_points(
                pv.PolyData(mesh_cloud), scalars=np.full_like(mesh_cloud, 255)
            )
            pl.add_points(
                pv.PolyData(stereo_cloud), scalars=np.full_like(stereo_cloud, 0)
            )
            pl.update()

    run_icp(stereo_cloud, mesh_cloud)


def main():
    # animate()
    icp_registration()
    # cpd_registration()
    # trimesh_icp()


if __name__ == "__main__":
    main()
