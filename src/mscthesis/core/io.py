from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from ..utilities.log import log_call


@log_call()
def load_voxels(file_path: str | Path) -> np.ndarray:
    """
    Load a 3D voxel grid from a .npy file.

    Args:
        filepath (str | Path): Path to the .npy file containing the voxel grid.

    Returns:
        np.ndarray: The loaded 3D voxel grid.
    """
    voxels = np.load(file_path)
    return voxels


@log_call()
def save_voxels(voxels: np.ndarray, file_path: str | Path) -> None:
    """
    Save a voxel model to a binary .npy file.

    Args:
        voxels (np.ndarray): 3D numpy array representing the voxel model.
        filename (str | Path): The output filename for the .npy file.
    """
    np.save(file_path, voxels)
    return


@log_call()
def load_surface_mesh(file_path: str | Path) -> o3d.geometry.TriangleMesh:
    """
    Load a surface mesh from a file.

    Args:
        file_path (str | Path): Path to the mesh file.
    Returns:
        o3d.geometry.TriangleMesh: The loaded surface mesh.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        raise IOError(f"Failed to read mesh from {file_path}")
    return mesh


@log_call()
def save_surface_mesh(mesh: o3d.geometry.TriangleMesh, file_path: str | Path) -> None:
    """
    Save a surface mesh to a file.

    Args:
        mesh (o3d.geometry.TriangleMesh): The surface mesh to save.
        file_path (str | Path): The output filename for the mesh file.
    """
    written = o3d.io.write_triangle_mesh(file_path, mesh)
    if not written:
        raise IOError(f"Failed to write mesh to {file_path}")
    return
