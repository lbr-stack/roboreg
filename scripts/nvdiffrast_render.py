# import os
# import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math

import kaolin as kal
import nvdiffrast
import torch
from matplotlib import pyplot as plt

# from roboreg.util import parse_camera_info

# height, width, _ = parse_camera_info("test/data/lbr_med7/high_res/left_camera_info.yaml") # height divisible by 8 ???? -> try kaolin for rendering, not nvdiffrast
height, width = 480, 640

glctx = nvdiffrast.torch.RasterizeCudaContext(device="cuda")

# Load a specific obj instead
OBJ_PATH = "test/data/lbr_med7/mesh/link_0.obj"
mesh = kal.io.obj.import_mesh(OBJ_PATH, with_materials=True, triangulate=True)

# Batch, move to GPU and center and normalize vertices in the range [-0.5, 0.5]
mesh = mesh.to_batched().cuda()
mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices, normalize=True)
print(mesh.to_string(print_stats=True))

# Use a single diffuse color as backup when map doesn't exist (and face_uvs_idx == -1)
mesh.uvs = torch.nn.functional.pad(mesh.uvs, (0, 0, 0, 1))
mesh.face_uvs_idx[mesh.face_uvs_idx == -1] = mesh.uvs.shape[1] - 1

cam = kal.render.camera.Camera.from_args(
    eye=torch.tensor([2.0, 0.0, 0.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=math.pi * 45 / 180,
    width=width,
    height=height,
    device="cuda",
)


def render(mesh, cam):
    vertices_camera = cam.extrinsics.transform(mesh.vertices)

    # Create a fake W (See nvdiffrast documentation)
    proj = cam.projection_matrix()[None]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(vertices_camera)[..., None]

    vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)

    rast = nvdiffrast.torch.rasterize(  # height divisible by 8 ???? -> try kaolin for rendering, not nvdiffrast
        glctx, vertices_clip, mesh.faces.int(), (cam.height, cam.width), grad_db=False
    )
    rast0 = torch.flip(rast[0], dims=(1,))
    return (rast0[:, :, :, -1:] != 0).squeeze()


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)
plt.imshow(render(mesh, cam).cpu())
plt.waitforbuttonpress()
