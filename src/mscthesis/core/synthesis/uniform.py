from __future__ import annotations

from typing import Any

import numpy as np

from ...utilities.log import log_call
from .helpers import (
    get_sample_seed,
    initialize_meshgrid,
)


@log_call()
def _metadata(
    random_seed: int, centers: np.ndarray, radii: np.ndarray, voxels: np.ndarray
) -> dict[str, Any]:
    """
    Generate metadata for a voxel model non-overlapping spheres.

    Args:
        random_seed (int): Seed used for random number generation.
        centers (np.ndarray): Array of cell centers of shape (N, 3).
        radii (np.ndarray): Array of cell radii of shape (N,).
        voxels (np.ndarray): The generated voxel model of shape (X, Y, Z).

    Returns:
        dict[str, Any]: Metadata dictionary.
    """
    metadata: dict[str, Any] = {}
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
    metadata["status"] = "success"
    return metadata


@log_call()
def generate_voxels_from_seed(
    random_seed: int,
    resolution: int,
    plug_aspect: float,
    num_cells: int,
    min_radius: float,
    max_radius: float,
    min_separation: float,
    max_attempts: int,
) -> tuple[np.ndarray[tuple[int, int, int]], dict[str, Any]]:
    """
    Generate uniform swiss cheese voxel models for a given seed.

    Args:
        seed (int): Seed for random number generation.
        resolution (int): Number of voxels along each axis.
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        num_cells (int): Number of cells (spheres) to place in the model.
        min_radius (float): Minimum radius of the cells.
        max_radius (float): Maximum radius of the cells.
        min_separation (float): Minimum separation distance between cells.
        max_attempts (int): Maximum attempts to place each cell without overlap.

    Returns:
        np.ndarray: 3D numpy array of shape (planar_resolution, planar_resolution, resolution)
        with uint8 values, where 1 indicates presence of tissue (cell)
        and 0 indicates airspace.
        dict[str, Any]: Metadata dictionary.
    """

    np.random.seed(random_seed)

    voxels, (X, Y, Z) = initialize_meshgrid(plug_aspect, resolution)

    # initialize cell lists and determine placement boundaries
    centers = np.zeros((num_cells, 3))
    radii = np.zeros((num_cells,))
    max_r = plug_aspect - max_radius - min_separation
    min_z = max_radius + min_separation
    max_z = 1 - max_radius - min_separation

    if max_r <= 0:
        raise ValueError(
            f"Cell size {max_radius} and separation {min_separation} too large for given plug aspect {plug_aspect}."
        )

    # placement of cells
    for i in range(num_cells):
        attempts = 0
        while attempts < max_attempts:
            # draw random cell center
            center = np.array(
                [
                    np.random.uniform(-max_r, max_r),
                    np.random.uniform(-max_r, max_r),
                    np.random.uniform(min_z, max_z),
                ]
            )

            # enforce cyllindrical boundary
            if np.linalg.norm(center[:2]) > max_r:
                attempts += 1
                continue

            # draw random cell radius and check for overlaps
            radius = np.random.uniform(min_radius, max_radius)
            if np.all(
                np.linalg.norm(centers[:i] - center, axis=1)
                > (radii[:i] + radius + min_separation)
            ):
                centers[i] = center
                radii[i] = radius
                break
            attempts += 1

        else:  # executed only if while loop is not stopped by break - then we dont attempt to place any further spheres
            break  # break out of the for loop

        # compute distance field and update voxels
        distance = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )
        voxels |= (distance <= radius).astype(np.uint8)

    # remove cells where radius is still zero (not placed)
    centers = centers[radii > 0]
    radii = radii[radii > 0]

    metadata = _metadata(
        random_seed,
        centers,
        radii,
        voxels,
    )

    return voxels, metadata


@log_call()
def generate_voxels_from_sample_id(
    sample_id: str,
    base_seed: int,
    resolution: int,
    plug_aspect: float,
    num_cells: int,
    min_radius: float,
    max_radius: float,
    min_separation: float,
    max_attempts: int,
) -> tuple[np.ndarray[tuple[int, int, int]], dict[str, Any]]:
    """
    Generate uniform swiss cheese voxel models for a given sample ID.

    Args:
        sample_id (str): Unique identifier for the sample. Mappable to int.
        base_seed (int): Base seed for random number generation.
        resolution (int): Number of voxels along each axis.
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        num_cells (int): Number of cells (spheres) to place in the model.
        min_radius (float): Minimum radius of the cells.
        max_radius (float): Maximum radius of the cells.
        min_separation (float): Minimum separation distance between cells.
        max_attempts (int): Maximum attempts to place each cell without overlap.

    Returns:
        np.ndarray: 3D numpy array of shape (planar_resolution, planar_resolution, resolution)
        with uint8 values, where 1 indicates presence of tissue (cell)
        and 0 indicates airspace.
        dict[str, Any]: Metadata dictionary.
    """

    # calulate and fix sample seed
    random_seed = get_sample_seed(base_seed, sample_id)

    voxels, metadata = generate_voxels_from_seed(
        random_seed,
        resolution,
        plug_aspect,
        num_cells,
        min_radius,
        max_radius,
        min_separation,
        max_attempts,
    )

    return voxels, metadata
