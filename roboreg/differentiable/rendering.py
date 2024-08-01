from typing import List

import nvdiffrast.torch as dr
import torch


class NVDiffRastRenderer:
    _device: torch.device
    _ctx: dr.RasterizeCudaContext

    def __init__(self, device: torch.device = "cuda") -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        self._device = device
        self._ctx = dr.RasterizeCudaContext(device=self._device)

    def constant_color(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        resolution: List[int],
        color: List[float] = [1.0],
    ) -> torch.Tensor:
        rast, _ = dr.rasterize(self._ctx, vertices, faces, resolution)
        # the nvdiffrast interpolation is not required for this simple case
        # simply assign a constant color where there is a triange, i.e. id != 0
        col = torch.tensor(color, device=self._device, dtype=torch.float32).unsqueeze(0)
        render = torch.where(
            rast[..., -1].unsqueeze(-1) != 0, col, torch.zeros_like(col)
        )
        return dr.antialias(render, rast, vertices, faces)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def ctx(self) -> dr.RasterizeCudaContext:
        return self._ctx
