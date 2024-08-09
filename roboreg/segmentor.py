from typing import Any

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import SamPredictor, sam_model_registry


class Segmentor(object):
    _model: Any
    _device: str

    def __init__(self, device: str) -> None:
        self._device = device

    def __call__(self, img: np.ndarray) -> Any:
        raise NotImplementedError


class Sam2Segmentor(Segmentor):
    def __init__(
        self, model_id: str = "facebook/sam2-hiera-large", device: str = "cuda"
    ) -> None:
        super().__init__(device=device)
        self._model = SAM2ImagePredictor.from_pretrained(model_id)

    def __call__(
        self, img: np.ndarray, input_points: np.ndarray, input_labels: np.ndarray
    ) -> Any:
        self._model.set_image(img)
        with torch.inference_mode(), torch.autocast(self._device, dtype=torch.bfloat16):
            masks, _, _ = self._model.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
        return masks[0]


class SamSegmentor(Segmentor):
    def __init__(self, checkpoint: str, model_type: str, device: str = "cuda") -> None:
        super().__init__(device=device)
        self._sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self._sam.to(device=device)
        self._model = SamPredictor(self._sam)

    def __call__(
        self, img: np.ndarray, input_points: np.ndarray, input_labels: np.ndarray
    ) -> np.ndarray:
        self._model.set_image(img)
        with torch.no_grad():
            masks, _, _ = self._model.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
        return masks[0]
