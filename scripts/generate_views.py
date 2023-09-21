import os

import numpy as np
import open3d as o3d
from ament_index_python import get_package_share_directory

path = get_package_share_directory("lbr_description")
links = [os.path.join(path, f"meshes/med7/collision/link_{i}.stl") for i in range(8)]


meshes = [o3d.io.read_triangle_mesh(link) for link in links]
meshes = [mesh.compute_vertex_normals() for mesh in meshes]

# view mesh
# o3d.visualization.draw_geometries(meshes)

pcd = o3d.geometry.PointCloud()

for mesh in meshes:
    pcd += mesh.sample_points_poisson_disk(1000)

# https://dl.acm.org/doi/abs/10.1145/1275808.1276407?casa_token=Mwn049sRLKgAAAAA:XxHEhPWb9dMoZ31Um78h1Bb1Oweg-Th3YBzQx3zeD-jG5gMJex3SNv5-wmxhywdgDP56MLKvuNQ
# http://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html#Hidden-point-removal


diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
)

# view pcd
# o3d.visualization.draw_geometries([pcd])


print("Define parameters used for hidden_point_removal")
camera = [1, 0, 0]
radius = diameter * 100

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

print("Visualize result")
pcd = pcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([pcd])
