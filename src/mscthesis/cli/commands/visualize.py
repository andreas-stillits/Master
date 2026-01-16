from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from ...core.io import load_surface_mesh, load_voxels
from ...core.visualization import (
    visualize_surface_mesh,
    visualize_volumetric_mesh,
    visualize_voxels,
)
from ...utilities.paths import ProjectPaths, resolve_existing_samples_file
from ..shared import derive_cli_flags_from_config


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to visualize the contents of a file via file extension"""
    rank = comm.Get_rank()

    paths: ProjectPaths = ProjectPaths(args.config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()

    if rank == 0:
        file_path: Path = resolve_existing_samples_file(
            paths, args.file_path, ".npy", ".stl", ".msh"
        )

        # visualize based on file extension
        if file_path.suffix == ".npy":
            voxels = load_voxels(file_path)
            visualize_voxels(voxels, material_id=1)

        elif file_path.suffix == ".stl":
            mesh = load_surface_mesh(file_path)
            visualize_surface_mesh(mesh)

        elif file_path.suffix == ".msh":
            visualize_volumetric_mesh(file_path)

        else:
            raise ValueError(
                f"Unsupported file extension '{file_path.suffix}' for visualization."
            )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    cmd_name = "visualize"
    parser = subparsers.add_parser(
        cmd_name,
        description="Visualize file content based on file extension",
        help="Visualize the contents of a file based on its extension",
        epilog="msc visualize [options] <file_path>",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the file to visualize (e.g., .npy for voxel grids)",
    )
    parser = derive_cli_flags_from_config(parser, cmd_name)
    parser.set_defaults(cmd=_cmd)
