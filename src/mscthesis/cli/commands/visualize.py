from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from ...core.visualization import load_voxels_from_npy, visualize_voxels
from ...utilities.checks import verify_existence
from ...utilities.paths import expand_samples_path
from ..shared import derive_cli_flags_from_config


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to visualize the contents of a file via file extension"""
    rank = comm.Get_rank()

    if rank == 0:
        file_path: Path = expand_samples_path(
            args.config.behavior.storage_root, args.file_path
        )

        # verify existence of file
        verify_existence(file_path)

        # visualize based on file extension
        if file_path.suffix == ".npy":
            voxels = load_voxels_from_npy(file_path)
            visualize_voxels(voxels, material_id=1)

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
