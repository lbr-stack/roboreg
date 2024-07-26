from typing import List

import torch
import trimesh
from torch import FloatTensor

import nvdiffrast.torch as dr


class DiffBot:
    r"""Differentiable robot."""

    def __init__(self, meshes: List[str]) -> None:
        pass

    def render(self, intrinsics: FloatTensor, pose: FloatTensor) -> FloatTensor:
        pass

    def fk(self, q: FloatTensor) -> None:
        pass


class DiffBotModule(torch.nn.Module):
    f"""Differentiable robot module."""

    def __init__(self, meshes: List[str]) -> None:
        super().__init__()
        self._robot = DiffBot(meshes)

    def forward(self, intrinsics: FloatTensor, pose: FloatTensor) -> FloatTensor:
        pass


if __name__ == "__main__":
    #### robot:
    ### - render(intrinsics, pose)
    ### - fk(q)

    # Step 0: Load mesh and visualize
    mesh_0: trimesh.Geometry = trimesh.load("test/data/lbr_med7/mesh/link_0.stl")
    mesh_1: trimesh.Geometry = trimesh.load("test/data/lbr_med7/mesh/link_1.stl")
    meshes = [mesh_0, mesh_1]

    # mesh = trimesh.load("test/data/xarm/mesh/link_base.STL")
    # mesh_0.show()

    # apply different transforms per mesh

    # TODO: mesh -> meshes + FK

    # Step 1: Render sample mesh (dae / stl)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = torch.cat(
        [
            torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
            for mesh in meshes
        ],
        dim=0,
    )

    # xyz -> xyz1
    pos = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1)
    pos = pos.unsqueeze(0)

    # FK: N_links x 4 x 4 and applied to range of vertices
    # # TODO: projections (range mode) including FK
    pos_idx = torch.cat(
        [torch.tensor(mesh.faces, device=device, dtype=torch.int32) for mesh in meshes],
        dim=0,
    )

    # create ranges
    ranges = []
    previous_range = 0
    for mesh in meshes:
        ranges.append([previous_range, previous_range + len(mesh.faces) - 1])
        previous_range += len(mesh.faces)
    ranges = torch.tensor(ranges, device="cpu", dtype=torch.int32)

    ## target resolution, target color,....

    ctx = dr.RasterizeCudaContext(device=device)
    rast, _ = dr.rasterize(ctx, pos, pos_idx, resolution=[512, 512], ranges=ranges)
    col = torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=torch.float32).unsqueeze(
        0
    )
    col_idx = torch.zeros_like(pos_idx)
    color, _ = dr.interpolate(col, rast, col_idx)
    color = dr.antialias(color, rast, pos, pos_idx)

    # display rasterized image
    import cv2

    rast = rast.detach().cpu().numpy()
    print(rast.shape)
    cv2.imshow("rasterized", rast[0])
    cv2.waitKey()

    # display interpolation
    color = color.detach().cpu().numpy()
    print(color.shape)
    cv2.imshow("interpolated", color[0])
    cv2.waitKey()
