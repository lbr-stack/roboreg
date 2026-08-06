from pathlib import Path

import numpy as np


def save_extrinsics(
    extrinsics_file: Path | str,
    extrinsics: np.ndarray,
) -> None:
    r"""Save extrinsics to a NumPy or CSV file."""
    extrinsics_file = Path(extrinsics_file)
    suffix = extrinsics_file.suffix.lower()

    if extrinsics.shape != (4, 4):
        raise ValueError(
            f"Expected extrinsics with shape (4, 4), got {extrinsics.shape}."
        )

    if suffix == ".npy":
        np.save(extrinsics_file, extrinsics)
    elif suffix == ".csv":
        np.savetxt(extrinsics_file, extrinsics, delimiter=",")
    else:
        raise ValueError(
            f"Unsupported extrinsics file type '{suffix}'. "
            "Expected '.npy' or '.csv'."
        )
