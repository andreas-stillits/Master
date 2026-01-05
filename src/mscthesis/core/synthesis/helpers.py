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


@log_call()
def initialize_meshgrid(
    plug_aspect: float,
    planar_resolution: int,
    axial_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize a 3D meshgrid for voxel generation.

    Args:
        planar_resolution (int): Number of voxels along the x and y axes.
        axial_resolution (int): Number of voxels along the z axis.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Meshgrid arrays for X, Y, Z coordinates.
    """
    x = np.linspace(-plug_aspect, plug_aspect, planar_resolution)
    y = np.linspace(-plug_aspect, plug_aspect, planar_resolution)
    z = np.linspace(0, 1, axial_resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


@log_call()
def metadata_nonoverlapping_spheres(
    random_seed: int, centers: np.ndarray, radii: np.ndarray, voxels: np.ndarray
) -> dict[str, float | int]:
    """
    Generate metadata for a voxel model non-overlapping spheres.

    Args:
        random_seed (int): Seed used for random number generation.
        centers (np.ndarray): Array of cell centers of shape (N, 3).
        radii (np.ndarray): Array of cell radii of shape (N,).
        voxels (np.ndarray): The generated voxel model of shape (X, Y, Z).

    Returns:
        dict[str, float | int]: Metadata dictionary.
    """
    metadata: dict[str, float | int] = {}
    metadata["random_seed"] = random_seed
    metadata["num_cells_placed"] = len(centers)
    metadata["min_radius"] = float(np.min(radii)) if len(radii) > 0 else 0.0
    metadata["mean_radius"] = float(np.mean(radii)) if len(radii) > 0 else 0.0
    metadata["max_radius"] = float(np.max(radii)) if len(radii) > 0 else 0.0
    metadata["mean_cell_volume"] = (
        float(np.mean([(4 / 3) * np.pi * r**3 for r in radii]))  # type: ignore
        if len(radii) > 0
        else 0.0
    )
    metadata["mean_porosity"] = 1.0 - float(np.sum(voxels) / voxels.size)
    metadata["std_porosity"] = float(
        np.std(1.0 - np.sum(voxels, axis=(0, 1)) / (voxels.shape[0] * voxels.shape[1]))
    )
    return metadata
