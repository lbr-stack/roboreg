from typing import Any

import numpy as np
from segment_anything import SamPredictor, sam_model_registry


class Segmentor(object):
    model: Any

    def __call__(self, img: np.ndarray) -> Any:
        raise NotImplementedError


class SamSegmentor(Segmentor):
    def __init__(
        self, sam_checkpoint: str, model_type: str, device: str = "cuda"
    ) -> None:
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.model = SamPredictor(self.sam)

    def __call__(
        self, img: np.ndarray, input_points: np.ndarray, input_labels: np.ndarray
    ) -> np.ndarray:
        self.model.set_image(img)
        masks, scores, logits = self.model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        return masks[0]
