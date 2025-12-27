from __future__ import annotations

import numpy as np

from ...utilities.log import log_call
from .helpers import get_sample_seed


def _initialize_meshgrid(
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
def generate_uniform_swiss_voxels(
    sample_id: str,
    base_seed: int,
    resolution: int,
    plug_aspect: float,
    num_cells: int,
    min_radius: float,
    max_radius: float,
    min_separation: float,
    max_attempts: int,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    """
    Generate uniform swiss cheese voxel models for a list of sample IDs.

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
    """

    # calulate and fix sample seed
    np.random.seed(get_sample_seed(base_seed, sample_id))

    # scale planar resolution to have isotropic sampling of space
    planar_resolution = int(2 * plug_aspect * resolution)

    # create meshgrid and empty voxels
    X, Y, Z = _initialize_meshgrid(plug_aspect, planar_resolution, resolution)
    voxels = np.zeros(
        (planar_resolution, planar_resolution, resolution), dtype=np.uint8
    )

    # initialize cell lists and determine placement boundaries
    centers = []
    radii = []
    max_xy = plug_aspect - max_radius - min_separation
    min_z = max_radius + min_separation
    max_z = 1 - max_radius - min_separation

    # placement of cells
    for _ in range(num_cells):
        attempts = 0
        while attempts < max_attempts:
            # draw random cell center
            center = np.array(
                [
                    np.random.uniform(-max_xy, max_xy),
                    np.random.uniform(-max_xy, max_xy),
                    np.random.uniform(min_z, max_z),
                ]
            )

            # enforce cyllindrical boundary
            if np.linalg.norm(center[:2]) > max_xy:
                attempts += 1
                continue

            # draw random cell radius and check for overlaps
            radius = np.random.uniform(min_radius, max_radius)
            if all(
                np.linalg.norm(center - placed_center)
                > (radius + placed_radius + min_separation)
                for placed_center, placed_radius in zip(centers, radii, strict=False)
            ):
                centers.append(center)
                radii.append(radius)
                break
            attempts += 1

        else:  # executed only if while loop is not stopped by break - then we dont attempt to place any further spheres
            break  # break out of the for loop

        # compute distance field and update voxels
        distance = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )
        voxels |= (distance <= radius).astype(
            np.uint8
        )  # set tissue voxels to 1 within mask

    return voxels
