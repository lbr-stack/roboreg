import pathlib
from typing import List


def find_files(path: str, pattern: str = "img_*.png") -> List[str]:
    path = pathlib.Path(path)
    image_paths = list(path.glob(pattern))
    return [image_path.name for image_path in image_paths]
