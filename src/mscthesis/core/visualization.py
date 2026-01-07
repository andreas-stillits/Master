from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from ..utilities.log import log_call


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
    o3d.visualization.draw_geometries([pcd])  # type: ignore[reportAttributeAccessIssue]
    return
