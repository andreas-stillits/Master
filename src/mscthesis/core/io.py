from __future__ import annotations

from pathlib import Path
from typing import Any

import adios4dolfinx as a4x
import gmsh
import numpy as np
import open3d as o3d
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.mesh import Mesh
from mpi4py import MPI

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


# monkey patch for silent GMSH
_original_initialize = gmsh.initialize


def _quiet_initialize(*args, **kwargs):
    """Initialize GMSH without printing to stdout."""
    _original_initialize(*args, **kwargs)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)


@log_call()
def load_volumetric_mesh(file_path: str | Path) -> tuple[Mesh, Any, Any]:
    """
    Load a volumetric mesh from a Gmsh file.

    Args:
        file_path (str | Path): Path to the Gmsh .msh file.
    Returns:
        tuple[Mesh, Any, Any]: The loaded volumetric mesh, cell tags, and facet tags.
    """
    # override the gmsh.initialize() call in gmshio to suppress output
    gmsh.initialize = _quiet_initialize
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        file_path, MPI.COMM_SELF, 0, gdim=3
    )
    return mesh, cell_tags, facet_tags


@log_call()
def save_fem_solution(
    solution: fem.Function,
    mesh: Mesh,
    cell_tags: Any,
    facet_tags: Any,
    file_path: str | Path,
) -> None:
    """
    Save FEniCSx solution as a .bp file
    Args:
        solution (fem.Function): The FEniCSx solution to save.
        mesh (Mesh): The mesh on which the solution is defined.
        cell_tags (Any): The cell tags associated with the mesh.
        facet_tags (Any): The facet tags associated with the mesh.
    """
    a4x.write_mesh(file_path, mesh)
    a4x.write_meshtags(file_path, mesh, cell_tags, meshtag_name="cell_tags")
    a4x.write_meshtags(file_path, mesh, facet_tags, meshtag_name="facet_tags")
    a4x.write_function(file_path, solution, name="solution")
    return


@log_call()
def load_fem_solution(file_path: str | Path) -> tuple[fem.Function, Mesh, Any, Any]:
    """
    Load a FEniCSx solution from a .bp file
    Args:
        file_path (str | Path): The path to the .bp file containing the solution.
    Returns:
        tuple[fem.Function, Mesh, Any, Any]: The loaded solution, mesh, cell tags, and facet tags.
    """
    mesh = a4x.read_mesh(file_path, MPI.COMM_SELF)
    cell_tags = a4x.read_meshtags(file_path, mesh, meshtag_name="cell_tags")
    facet_tags = a4x.read_meshtags(file_path, mesh, meshtag_name="facet_tags")
    functionspace = fem.functionspace(mesh, ("Lagrange", 1))
    solution = fem.Function(functionspace)
    a4x.read_function(file_path, solution, name="solution")
    return solution, mesh, cell_tags, facet_tags
