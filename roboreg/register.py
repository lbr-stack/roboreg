import copy

import numpy as np
import open3d as o3d


def clean_xyz(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r"""

    Args:
        xyz: Point cloud of HxWx3.
        mask: Mask for the point cloud.

    Returns:
        The cleaned point cloud of shape Nx3.
    """
    # mask the cloud
    clean_xyz = np.where(mask[..., None], xyz, np.nan)
    # remove nan
    clean_xyz = clean_xyz[~np.isnan(clean_xyz).any(axis=2)]
    return clean_xyz


class O3DRegister:
    def __init__(self, observed_xyz: np.ndarray, mesh_xyz) -> None:
        self._observed_xyz_pcd = o3d.geometry.PointCloud()
        self._mesh_xyz_pcd = o3d.geometry.PointCloud()

        self.observed_xyz_pcd = observed_xyz  # calls into setter
        self.mesh_xyz_pcd = mesh_xyz
        self._transformation = None
        observed_com = self.observed_xyz_pcd.get_center()
        mesh_com = self.mesh_xyz_pcd.get_center()
        self._trans_init = np.identity(4)
        self._trans_init[:3, 3] = mesh_com - observed_com
        print("Initial transformation is:")
        print(self._trans_init)

    def draw_registration_result(self):
        observed_xyz_pcd = copy.deepcopy(self.observed_xyz_pcd)
        mesh_xyz_pcd = copy.deepcopy(self.mesh_xyz_pcd)
        observed_xyz_pcd.paint_uniform_color([1, 0.706, 0])
        mesh_xyz_pcd.paint_uniform_color([0, 0.651, 0.929])
        observed_xyz_pcd.transform(self._transformation)
        o3d.visualization.draw_geometries([observed_xyz_pcd, mesh_xyz_pcd])

    @property
    def observed_xyz_pcd(self) -> o3d.geometry.PointCloud:
        return self._observed_xyz_pcd

    @observed_xyz_pcd.setter
    def observed_xyz_pcd(self, observed_xyz: np.ndarray):
        self._observed_xyz_pcd.points = o3d.utility.Vector3dVector(observed_xyz)

    @property
    def mesh_xyz_pcd(self) -> o3d.geometry.PointCloud:
        return self._mesh_xyz_pcd

    @mesh_xyz_pcd.setter
    def mesh_xyz_pcd(self, mesh_xyz: np.ndarray):
        self._mesh_xyz_pcd.points = o3d.utility.Vector3dVector(mesh_xyz)

    def register(self) -> None:
        raise NotImplementedError


class ICPRegister(O3DRegister):
    def __init__(self, observed_xyz: np.ndarray, mesh_xyz: np.ndarray) -> None:
        super().__init__(observed_xyz, mesh_xyz)

    def register(self, threshold: float = 0.02) -> None:
        print("Running point-to-point ICP.")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.observed_xyz_pcd,
            self.mesh_xyz_pcd,
            threshold,
            self._trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        print("Transformation is:")
        print(reg_p2p.transformation)
        self._transformation = reg_p2p.transformation
