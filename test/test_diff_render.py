from typing import List
import numpy as np

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
    # meshes = [mesh_1]
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

    # create ranges
    lengths = [len(mesh.vertices) for mesh in meshes]
    lengths = torch.tensor(lengths, device="cpu", dtype=torch.int32)

    # offsets
    offsets = torch.tensor(
        [0] + list(np.cumsum(lengths)[:-1]), device=device, dtype=torch.int32
    )

    print("offsets")
    print(offsets)
    print("lenghts")
    print(lengths)

    for idx, mesh in enumerate(meshes):
        print(f"***************************{idx}")
        print(mesh.faces)

        offset_idcs = torch.add(
            torch.tensor(meshes[idx].faces, device=device, dtype=torch.int32),
            offsets[idx],
        )
        print(offset_idcs)

    # FK: N_links x 4 x 4 and applied to range of vertices
    # # TODO: projections (range mode) including FK
    pos_idx = torch.cat(  ### offset these indices by ranges
        [
            torch.add(
                torch.tensor(meshes[i].faces, device=device, dtype=torch.int32),
                offsets[i],
            )
            for i in range(len(meshes))
        ],
        dim=0,
    )

    # offsets = torch.cat(
    #     [
    #         torch.tensor(
    #             [offsets[i]] * len(meshes[i].faces),
    #             device=device,
    #             dtype=torch.int32,
    #         )
    #         for i in range(len(meshes))
    #     ]
    # )
    print("************************+")
    print(offsets.shape)

    # replicate offsets Nx1 -> Nx3
    offsets = torch.cat([offsets.unsqueeze(1)] * 3, dim=1)

    print("offset")
    print(pos_idx)

    # pos_idx = pos_idx + offsets.unsqueeze(1)

    ## target resolution, target color,....

    # some dummy transform
    import transformations
    import cv2

    try:
        cnt = 0
        pos = pos.unsqueeze(0)

        DUMMY_HT = torch.from_numpy(
            transformations.euler_matrix(np.pi / 2, 0.0, 0.0).astype("float32")
        ).to(device)
        pos = torch.matmul(pos, DUMMY_HT.t())
        while True:
            cnt += 1
            DUMMY_HT = torch.from_numpy(
                transformations.euler_matrix(0.0, 0.05, 0.0).astype("float32")
            ).to(device)
            pos = torch.matmul(pos, DUMMY_HT.t())

            ctx = dr.RasterizeCudaContext(device=device)
            rast, _ = dr.rasterize(
                ctx,
                pos,
                pos_idx,
                resolution=[512, 512],
                ranges=torch.tensor(
                    [0, 4038], device="cpu", dtype=torch.int32
                ).unsqueeze(0),
            )
            col = torch.tensor(
                [[1.0, 1.0, 1.0]], device=device, dtype=torch.float32
            ).unsqueeze(0)
            col_idx = torch.zeros_like(pos_idx)
            color, _ = dr.interpolate(col, rast, col_idx)
            color = dr.antialias(color, rast, pos, pos_idx)

            # display rasterized image

            # rast = rast.detach().cpu().numpy()
            # cv2.imshow("rasterized", rast[0])
            # cv2.waitKey()

            # display interpolation
            color = color.detach().cpu().numpy()
            print(color.shape)
            cv2.imshow("interpolated", color[0])
            cv2.waitKey()
    except KeyboardInterrupt:
        pass
