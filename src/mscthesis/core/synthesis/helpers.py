from __future__ import annotations

from pathlib import Path

import numpy as np

from ...utilities.log import log_call


@log_call()
def get_sample_seed(base_seed: int, sample_id: str) -> int:
    """
    Get a unique deterministic seed for a given sample ID based on a base seed.

    Args:
        base_seed (int): The base seed for random number generation.
        sample_id (str): The unique identifier for the sample.

    Returns:
        int: A unique seed for the sample.
    """
    return base_seed + 17 * 31 * 53 * int(sample_id)


@log_call()
def save_voxel_model(
    voxels: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]], file_path: str | Path
) -> None:
    """
    Save a voxel model to a binary .npy file.

    Args:
        voxels (np.ndarray): 3D numpy array representing the voxel model.
        filename (str | Path): The output filename for the .npy file.
    """
    np.save(file_path, voxels)
