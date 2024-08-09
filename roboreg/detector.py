from abc import ABC, abstractmethod
from csv import DictReader, DictWriter
from typing import List, Tuple

import cv2
import numpy as np


class SampleParserMixin:
    def __init__(self) -> None:
        self._fieldnames = ["x", "y", "label"]

    def read(self, path: str) -> Tuple[List[List[int]], List[int]]:
        samples = []
        labels = []
        with open(path, "r") as f:
            reader = DictReader(f, fieldnames=self._fieldnames)
            next(reader)  # skip header
            for row in reader:
                samples.append(
                    [int(row[self._fieldnames[0]]), int(row[self._fieldnames[1]])]
                )
                labels.append(int(row[self._fieldnames[2]]))
        return samples, labels

    def write(self, path: str, samples: List[List[int]], labels: List[int]) -> None:
        with open(path, "w") as f:
            writer = DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for sample, label in zip(samples, labels):
                writer.writerow(
                    {
                        self._fieldnames[0]: sample[0],
                        self._fieldnames[1]: sample[1],
                        self._fieldnames[2]: label,
                    }
                )


class Detector(ABC, SampleParserMixin):
    def __init__(self, n_positive_samples: int, n_negative_samples: int) -> None:
        super().__init__()
        self._positive_samples = []
        self._negative_samples = []
        self._n_positive_samples = n_positive_samples
        self._n_negative_samples = n_negative_samples

    def clear(self) -> None:
        self._positive_samples = []
        self._negative_samples = []

    @abstractmethod
    def detect(self, img: np.ndarray) -> Tuple[List, List]:
        raise NotImplementedError

    @property
    def positive_samples(self) -> List[List[int]]:
        return self._positive_samples

    @positive_samples.setter
    def positive_samples(self, value: List[List[int]]) -> None:
        self._positive_samples = value

    @property
    def negative_samples(self) -> List[List[int]]:
        return self._negative_samples

    @negative_samples.setter
    def negative_samples(self, value: List[List[int]]) -> None:
        self._negative_samples = value

    @property
    def samples(self) -> List[List[int]]:
        return self._positive_samples + self._negative_samples

    @property
    def labels(self) -> List[int]:
        return [1] * len(self._positive_samples) + [0] * len(self._negative_samples)


class OpenCVDetector(Detector):
    def __init__(
        self, n_positive_samples: int = 3, n_negative_samples: int = 3
    ) -> None:
        super().__init__(
            n_positive_samples=n_positive_samples, n_negative_samples=n_negative_samples
        )

    def _on_mouse(self, event, x, y, flags, param):
        if (
            event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY
        ):  # bitwise and for flags: https://stackoverflow.com/questions/32210066/mouse-callback-event-flags-in-python-opencv-osx
            print(f"event: {event}, flags: {flags}, CTRL: {cv2.EVENT_FLAG_CTRLKEY}")
            if len(self._negative_samples) >= self._n_negative_samples:
                print(
                    f"Already added {len(self._negative_samples)} of {self._n_negative_samples}  negative samples."
                )
                return
            self._negative_samples.append([x, y])
            print(
                f"Added negative sample {x}, {y}. Total: {len(self._negative_samples)} of {self._n_negative_samples}."
            )
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"event: {event}, flags: {flags}, CTRL: {cv2.EVENT_FLAG_CTRLKEY}")
            if len(self._positive_samples) >= self._n_positive_samples:
                print(
                    f"Already added {len(self._positive_samples)} of {self._n_positive_samples} positive samples. Use CTRL + Left Click to add negative samples."
                )
                return
            self._positive_samples.append([x, y])
            print(
                f"Added positive sample {x}, {y}. Total: {len(self._positive_samples)} of {self._n_positive_samples}."
            )
            return

    def detect(self, img: np.ndarray) -> Tuple[List, List]:
        cv2.namedWindow("detect")
        cv2.setMouseCallback("detect", self._on_mouse)
        img_cpy = img.copy()
        while (
            len(self._positive_samples) < self._n_positive_samples
            or len(self._negative_samples) < self._n_negative_samples
        ):
            try:
                cv2.imshow("detect", img_cpy)
                cv2.waitKey(10)

                # draw samples
                if len(self._positive_samples) > 0:
                    cv2.circle(
                        img_cpy,
                        (self._positive_samples[-1][0], self._positive_samples[-1][1]),
                        5,
                        (255, 255, 0),
                        -1,
                    )
                if len(self._negative_samples) > 0:
                    cv2.circle(
                        img_cpy,
                        (self._negative_samples[-1][0], self._negative_samples[-1][1]),
                        5,
                        (0, 255, 255),
                        -1,
                    )
            except KeyboardInterrupt:
                break
        cv2.destroyAllWindows()
        return self.samples, self.labels
