import cv2
import numpy as np

from roboreg.registration.image.solver import OptimizationState
from roboreg.util.viz import overlay_mask


class RenderOverlayCallback:
    def __init__(
        self,
        images: dict[str, list[np.ndarray]],
        every_n_iterations: int = 1,
    ) -> None:
        self._images = images
        self._every_n_iterations = every_n_iterations

    def __call__(self, state: OptimizationState) -> None:
        if state.iteration % self._every_n_iterations != 0:
            return
        for camera_name, render in state.renders.items():
            images = self._images.get(camera_name)
            if images is None:
                continue
            image = images[0]
            mask = render[0].detach().cpu().numpy().squeeze()
            mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            if image.shape[:2] != mask.shape:
                image = cv2.resize(
                    image,
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            overlay = overlay_mask(
                img=image,
                mask=mask,
                mode="r",
                scale=1.0,
            )
            cv2.imshow(f"Render overlay: {camera_name}", overlay)
        cv2.waitKey(1)
