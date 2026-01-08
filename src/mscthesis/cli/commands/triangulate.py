from __future__ import annotations

import argparse

from mpi4py import MPI

from ...config.declaration import TriangulationConfig
from ...core.io import load_voxels, save_surface_mesh
from ...core.meshing.triangulation import triangulate_voxels
from ...utilities.paths import determine_target_directory, get_voxel_file_path
from ..shared import (
    add_target_directory_argument,
    derive_cli_flags_from_config,
    document_command_execution,
    interpret_sample_input,
)

CMD_NAME = "triangulate"
STORAGE_FOLDERNAME = "triangulation"


def _execute_single_sample_id(
    args: argparse.Namespace, sample_id: str, size: int
) -> None:
    """Execute synthesis for a single sample ID"""
    # get resolved config
    cmdconfig: TriangulationConfig = args.config.triangulate

    voxel_input_path = get_voxel_file_path(
        args.config.behavior.storage_root,
        sample_id,
    )
    voxels = load_voxels(voxel_input_path)

    # generate voxel model
    surface_mesh, metadata = triangulate_voxels(
        voxels,
        cmdconfig.smoothing_iterations,
        cmdconfig.decimation_target,
        cmdconfig.shrinkage_tolerance,
    )

    # save voxel model to disk
    target_directory = determine_target_directory(
        args.config.behavior.storage_root,
        sample_id,
        STORAGE_FOLDERNAME,
        args.target_dir,
    )
    filename = "surface_mesh.stl"
    file_path = target_directory / filename

    save_surface_mesh(surface_mesh, file_path)

    document_command_execution(
        args.config,
        target_directory,
        CMD_NAME,
        size,
        sample_id,
        inputs={"voxel_model": str(voxel_input_path.expanduser().resolve())},
        outputs={"surface_mesh": str(file_path.expanduser().resolve())},
        metadata=metadata,
    )

    return


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to generate a surface mesh from a voxel model using marching cubes"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    sample_ids = interpret_sample_input(
        args.config.behavior.storage_root,
        args.sample_input,
        args.config.behavior.sample_id_digits,
    )

    # early exit if less samples than workers - also cathes the case of zero samples:
    if rank > len(sample_ids) or len(sample_ids) == 0:
        return

    # distribute sample IDs among workers
    assigned_sample_ids = sample_ids[rank::size]
    for sample_id in assigned_sample_ids:
        _execute_single_sample_id(args, sample_id, size)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="generate a surface mesh from a voxel model using marching cubes",
        help="generate a surface mesh from a voxel model using marching cubes",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_input",
        type=str,
        help="Either a valid sample ID or path to a text file containing sample IDs (one per line)",
    )
    add_target_directory_argument(parser)
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
