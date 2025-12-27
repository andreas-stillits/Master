from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from ..utilities.log import log_call


@log_call()
def load_voxels_from_npy(file_path: str | Path) -> np.ndarray:
    """
    Load a 3D voxel grid from a .npy file.
    Args:
        file_path (str | Path): Path to the .npy file containing the voxel grid.
    Returns:
        np.ndarray: A 3D numpy array representing the voxel grid.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    elif not file_path.suffix == ".npy":
        raise ValueError("The file must be a .npy file.")

    return np.load(file_path)


@log_call()
def visualize_voxels(voxels: np.ndarray, material_id: int = 1) -> None:
    """
    Visualize a 3D voxel grid using Open3D.
    Args:
        voxels (np.ndarray): A 3D numpy array representing the voxel grid.
        material_id (int): The number ID identifying voxels to color
    """
    points = np.argwhere(voxels == material_id)
    if points.size == 0:
        raise ValueError("No voxels found with the specified material_id.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    return
