from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import rich
from torch.utils.data import Dataset

from .filesystem import find_files


class MonocularDataset(Dataset):
    def __init__(
        self,
        images_path: Union[Path, str],
        image_pattern: str,
        joint_states_path: Union[Path, str],
        joint_states_pattern: str,
    ):
        self._image_files = find_files(images_path, image_pattern)
        self._joint_states_files = find_files(joint_states_path, joint_states_pattern)

        rich.print("Found the following files:")
        rich.print(f"Images: {[f.name for f in self._image_files]}")
        rich.print(f"Joint states: {[f.name for f in self._joint_states_files]}")

        if len(self._image_files) != len(self._joint_states_files):
            raise ValueError(
                f"Number of images '{len(self._image_files)}' and joint states '{len(self._joint_states_files)}' do not match."
            )

        if len(self._image_files) == 0:
            raise ValueError("No images found.")

        if len(self._joint_states_files) == 0:
            raise ValueError("No joint states found.")

        for image_file, joint_states_file in zip(
            self._image_files, self._joint_states_files
        ):
            image_index = image_file.stem.split("_")[-1]
            joint_states_index = joint_states_file.stem.split("_")[-1]
            if image_index != joint_states_index:
                raise ValueError(
                    f"Image file index '{image_file.name}' and joint states file index '{joint_states_file.name}' do not match."
                )

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        image_file = self._image_files[idx]
        joint_states_file = self._joint_states_files[idx]
        image = cv2.imread(image_file)
        joint_states = np.load(joint_states_file)
        return image, joint_states, image_file.name
