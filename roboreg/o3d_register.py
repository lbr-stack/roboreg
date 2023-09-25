import copy

import numpy as np
import open3d as o3d


class O3DRegister:
    def __init__(self, observed_xyz: np.ndarray, mesh_xyz) -> None:
        self._observed_xyz_pcd = o3d.geometry.PointCloud()
        self._mesh_xyz_pcd = o3d.geometry.PointCloud()

        self.observed_xyz_pcd = observed_xyz  # calls into setter
        self.mesh_xyz_pcd = mesh_xyz
        self._transformation = None
        self._trans_init = np.identity(4)
        self.center_initial_transform()
        print("Initial transformation is:")
        print(self._trans_init)

    def draw_registration_result(self):
        observed_xyz_pcd = copy.deepcopy(self.observed_xyz_pcd)
        mesh_xyz_pcd = copy.deepcopy(self.mesh_xyz_pcd)
        observed_xyz_pcd.paint_uniform_color([1, 0.706, 0])
        mesh_xyz_pcd.paint_uniform_color([0, 0.651, 0.929])
        observed_xyz_pcd.transform(self._transformation)
        o3d.visualization.draw_geometries([observed_xyz_pcd, mesh_xyz_pcd])

    def center_initial_transform(self) -> None:
        observed_com = self.observed_xyz_pcd.get_center()
        mesh_com = self.mesh_xyz_pcd.get_center()
        self._trans_init[:3, 3] = mesh_com - observed_com

    @property
    def observed_xyz_pcd(self) -> o3d.geometry.PointCloud:
        return self._observed_xyz_pcd

    @observed_xyz_pcd.setter
    def observed_xyz_pcd(self, observed_xyz: np.ndarray):
        self._observed_xyz_pcd.points = o3d.utility.Vector3dVector(
            copy.deepcopy(observed_xyz)
        )

    @property
    def mesh_xyz_pcd(self) -> o3d.geometry.PointCloud:
        return self._mesh_xyz_pcd

    @mesh_xyz_pcd.setter
    def mesh_xyz_pcd(self, mesh_xyz: np.ndarray):
        self._mesh_xyz_pcd.points = o3d.utility.Vector3dVector(copy.deepcopy(mesh_xyz))

    def register(self) -> None:
        raise NotImplementedError


class GlobalRegistration(O3DRegister):
    # http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    def __init__(self, observed_xyz: np.ndarray, mesh_xyz: np.ndarray) -> None:
        super().__init__(observed_xyz, mesh_xyz)

    def register(self, voxel_size: float = 0.01) -> None:
        (
            observed_xyz_pcd_down,
            observed_xyz_pcd_fpfh,
        ) = self._preprocess_point_cloud(self.observed_xyz_pcd, voxel_size)

        (
            mesh_xyz_pcd_down,
            mesh_xyz_pcd_fpfh,
        ) = self._preprocess_point_cloud(self.mesh_xyz_pcd, voxel_size)

        distance_threshold = voxel_size * 1.5
        print("RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                observed_xyz_pcd_down,
                mesh_xyz_pcd_down,
                observed_xyz_pcd_fpfh,
                mesh_xyz_pcd_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        )

        self._transformation = result.transformation

    def _preprocess_point_cloud(
        self, pcd: o3d.geometry.PointCloud, voxel_size: float
    ) -> o3d.geometry.PointCloud:
        print("Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print("Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        radius_feature = voxel_size * 5
        print("Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        return pcd_down, pcd_fpfh


class ICPRegister(O3DRegister):
    # http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    def __init__(self, observed_xyz: np.ndarray, mesh_xyz: np.ndarray) -> None:
        super().__init__(observed_xyz, mesh_xyz)

    def register(self, threshold: float = 1.0) -> None:
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


class RobustICPRegister(O3DRegister):
    # http://www.open3d.org/docs/release/tutorial/pipelines/robust_kernels.html#Robust-ICP
    def __init__(self, observed_xyz: np.ndarray, mesh_xyz: np.ndarray) -> None:
        super().__init__(observed_xyz, mesh_xyz)

    def register(self, threshold: float = 1.0, sigma: float = 0.1) -> None:
        self.observed_xyz_pcd.estimate_normals()
        self.mesh_xyz_pcd.estimate_normals()

        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_p2l = o3d.pipelines.registration.registration_icp(
            self.observed_xyz_pcd, self.mesh_xyz_pcd, threshold, self._trans_init, p2l
        )
        print("Transformation is:")
        print(reg_p2l.transformation)
        self._transformation = reg_p2l.transformation
