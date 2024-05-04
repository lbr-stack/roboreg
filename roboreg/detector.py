from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np


class Detector(ABC):
    def __init__(self, buffer_size: int) -> None:
        self.points = []
        self.labels = []
        self.buffer_size = buffer_size

    def clear(self) -> None:
        self.points = []
        self.labels = []

    @abstractmethod
    def detect(self, img: np.ndarray) -> Tuple[List, List]:
        raise NotImplementedError


class OpenCVDetector(Detector):
    def __init__(self, buffer_size: int = 3) -> None:
        super().__init__(buffer_size)

    def _on_mouse(self, event, x, y, flags, param):
        if len(self.points) >= self.buffer_size:
            self.points.pop(0)
            self.labels.pop(0)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)
            print(
                f"Added point {x}, {y}. Total: {len(self.points)} of {self.buffer_size}."
            )

    def detect(self, img: np.ndarray) -> Tuple[List, List]:
        cv2.namedWindow("detect")
        cv2.setMouseCallback("detect", self._on_mouse)
        while len(self.points) < self.buffer_size:
            try:
                cv2.imshow("detect", img)
                cv2.waitKey(10)

                # draw points
                if len(self.points) > 0:
                    cv2.circle(
                        img,
                        (self.points[-1][0], self.points[-1][1]),
                        5,
                        (255, 255, 0),
                        -1,
                    )
            except KeyboardInterrupt:
                break
        cv2.destroyAllWindows()
        return self.points, self.labels
