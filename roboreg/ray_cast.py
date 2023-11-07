import open3d as o3d

from roboreg.o3d_robot import O3DRobot


class RayCastRobot:
    def __init__(self, robot: O3DRobot) -> None:
        self.robot = robot

    def cast(
        self,
        fov_deg: float,
        center: o3d.core.Tensor,
        eye: o3d.core.Tensor,
        up: o3d.core.Tensor,
        width_px: int,
        height_px: int,
    ) -> o3d.t.geometry.PointCloud:
        """
        Cast rays from a pinhole camera and visualize the result.

        refer doc: http://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html#open3d.t.geometry.RaycastingScene.create_rays_pinhole
        """
        scene = o3d.t.geometry.RaycastingScene()
        for mesh in self.robot.meshes:
            scene.add_triangles(mesh)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=fov_deg,
            center=center,
            eye=eye,
            up=up,
            width_px=width_px,
            height_px=height_px,
        )
        ans = scene.cast_rays(rays)

        # cloud
        hit = ans["t_hit"].isfinite()
        points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape(
            (-1, 1)
        )
        pcd = o3d.t.geometry.PointCloud(points)
        return pcd

    def cast_ht(
        self,
        intrinsic_matrix: o3d.core.Tensor,
        extrinsic_matrix: o3d.core.Tensor,
        width_px: int,
        height_px: int,
    ) -> o3d.t.geometry.PointCloud:
        """
        Cast rays from a pinhole camera and visualize the result.

        refer doc: http://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html#open3d.t.geometry.RaycastingScene.create_rays_pinhole
        """
        scene = o3d.t.geometry.RaycastingScene()
        for mesh in self.robot.meshes:
            scene.add_triangles(mesh)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            width_px=width_px,
            height_px=height_px,
        )
        ans = scene.cast_rays(rays)

        # cloud
        hit = ans["t_hit"].isfinite()
        points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape(
            (-1, 1)
        )
        pcd = o3d.t.geometry.PointCloud(points)
        return pcd
