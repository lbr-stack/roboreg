from typing import List, Tuple

import nvdiffrast.torch as dr
import torch


class NVDiffRastRenderer:
    r"""Simple renderer using nvdiffrast. Supports constant color
    rendering that bypasses the interpolation step. Adds functionality
    so that resolutions that are non-divisible by 8 can be rendered,
    refer https://github.com/NVlabs/nvdiffrast/issues/193#issuecomment-2250239862.
    """

    _device: torch.device
    _ctx: dr.RasterizeCudaContext

    def __init__(self, device: torch.device = "cuda") -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        self._device = device
        self._ctx = dr.RasterizeCudaContext(device=self._device)

    def scale_clip_vertices(
        self,
        clip_vertices: torch.Tensor,
        resolution: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        # find scaled resolution that is divisible by 8
        scaled_resolution = [(r + 7) // 8 * 8 for r in resolution]

        # scale vertices
        scaling_factor = torch.tensor(
            [
                1,
                resolution[0] / scaled_resolution[0],
                resolution[1] / scaled_resolution[1],
                1,
            ],
            dtype=clip_vertices.dtype,
            device=clip_vertices.device,
        )
        return clip_vertices * scaling_factor, scaled_resolution

    def constant_color(
        self,
        clip_vertices: torch.Tensor,
        faces: torch.Tensor,
        resolution: List[int],
        color: List[float] = [1.0],
    ) -> torch.Tensor:
        # scale clip vertices
        scaled_clip_vertices, scaled_resolution = self.scale_clip_vertices(
            clip_vertices, resolution
        )

        # render
        rast, _ = dr.rasterize(
            self._ctx, scaled_clip_vertices, faces, scaled_resolution
        )
        # the nvdiffrast interpolation is not required for this simple case
        # simply assign a constant color where there is a triange, i.e. id != 0
        col = torch.tensor(color, dtype=torch.float32, device=self._device).unsqueeze(0)
        render = torch.where(
            rast[..., -1].unsqueeze(-1) != 0, col, torch.zeros_like(col)
        )
        render = dr.antialias(render, rast, scaled_clip_vertices, faces)

        # center crop to original resolution
        return self._center_crop(render, resolution)

    def _center_crop(self, x: torch.Tensor, resolution) -> torch.Tensor:
        left = (x.shape[1] - resolution[0]) // 2
        right = left + resolution[0]
        top = (x.shape[2] - resolution[1]) // 2
        bottom = top + resolution[1]
        return x[:, left:right, top:bottom, :]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def ctx(self) -> dr.RasterizeCudaContext:
        return self._ctx
