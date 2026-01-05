from __future__ import annotations

import argparse

from mpi4py import MPI

from ...config.declaration import TriangulationConfig
from ...core.meshing.triangulation import save_triangulated_mesh, triangulate_voxels
from ..shared import (
    add_target_directory_argument,
    derive_cli_flags_from_config,
    determine_target_directory,
    document_command_execution,
    interpret_sample_input,
)

CMD_NAME = "triangulate"


def _execute_single_sample_id(
    args: argparse.Namespace, sample_id: str, size: int
) -> None:
    """Execute triangulation for a single sample ID"""
    # get resolved config
    cmdconfig: TriangulationConfig = args.config.triangulate

    # load voxel model from disk
    """ 
    OKAY I AM STOPPING HERE MID DEVELOPMENT TO GO TO BED.
    WE NEED TO LOAD THE VOXEL MODEL FROM DISK HERE.
    DETERMINE_TARGET_DIRECTORY MIGHT HAVE BECOME TOO RIGID USING CONFIG AND CMD_NAME.
    IMPLEMENT IN UTILITIES.PATHS INSTEAD OF IN CONFIG (IMMUTABLE).

    ALSO: UNIFORM AND TRINGULATION CORE CODE NEEDS GLUE - THERE SHOULD BE A SEPARATE WORK FUNCTION.
          THEN SEPARATE MANIFEST FUNCTION, AND A WRAPPER TO STITCH THEM - MAYBE JUST CMD DEFINITION.
    
    """

    # generate triangulated surface mesh
    mesh, metadata = triangulate_voxels()

    # save mesh to disk
    target_directory = determine_target_directory(
        args.config,
        CMD_NAME,
        sample_id,
        args.target_dir,
    )

    filename = "surface_mesh.stl"
    file_path = target_directory / filename

    save_triangulated_mesh(mesh, file_path)

    document_command_execution(
        args.config,
        target_directory,
        CMD_NAME,
        size,
        sample_id,
        inputs={},
        outputs={"surface_mesh": str(file_path.expanduser().resolve())},
        metadata=metadata,
        status="success",
    )

    return


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to generate a uniform swiss cheese voxel model"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    sample_ids = interpret_sample_input(
        args.sample_input, args.config.behavior.sample_id_digits
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
    """Register the synthesize uniform voxel model command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="generate a uniform swiss cheese voxel model",
        help="generate a uniform swiss cheese voxel model",
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
