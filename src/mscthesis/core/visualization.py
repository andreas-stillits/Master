from __future__ import annotations

from pathlib import Path

import gmsh
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


@log_call()
def visualize_surface_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    """
    Visualize a 3D mesh using Open3D.
    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to visualize.
    """
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], point_show_normal=True, mesh_show_wireframe=True)  # type: ignore[reportAttributeAccessIssue]
    return


@log_call()
def visualize_volumetric_mesh(mesh_path: str | Path) -> None:
    """
    Visualize a volumetric mesh from a given file path using GMSH's built-in GUI
    Args:
        mesh_path (str | Path): The file path to the volumetric mesh.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Mesh From File")
    gmsh.merge(str(mesh_path))
    gmsh.fltk.run()
    gmsh.finalize()
    return
