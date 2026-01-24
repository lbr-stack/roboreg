from pathlib import Path
from typing import List, Union


def find_files(path: Union[Path, str], pattern: str = "image_*.png") -> List[Path]:
    """Find files in a directory.

    Args:
        path (Union[Path, str]): Path to the directory.
        pattern (str): Pattern to match. Must include '_{number}.ext' format.

    Returns:
        List[Path]: Sorted file paths.
    """
    path = Path(path)
    file_paths = list(path.glob(pattern))
    return sorted(file_paths, key=lambda x: int(x.stem.split("_")[-1]))
