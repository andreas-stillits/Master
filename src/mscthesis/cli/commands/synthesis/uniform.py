from __future__ import annotations

import argparse

from mpi4py import MPI

from ....config.declaration import UniformSynthesisConfig
from ....core.io import save_voxels
from ....core.synthesis.uniform import generate_voxels_from_sample_id
from ...shared import (
    add_target_directory_argument,
    derive_cli_flags_from_config,
    determine_target_directory,
    document_command_execution,
    interpret_sample_input,
)

CMD_NAME = "synthesize-uniform"
STORAGE_FOLDERNAME = "synthesis"


def _execute_single_sample_id(
    args: argparse.Namespace, sample_id: str, size: int
) -> None:
    """Execute synthesis for a single sample ID"""
    # get resolved config
    cmdconfig: UniformSynthesisConfig = args.config.synthesize_uniform

    # generate voxel model
    voxels, metadata = generate_voxels_from_sample_id(
        sample_id,
        cmdconfig.base_seed,
        cmdconfig.resolution,
        cmdconfig.plug_aspect,
        cmdconfig.num_cells,
        cmdconfig.min_radius,
        cmdconfig.max_radius,
        cmdconfig.min_separation,
        cmdconfig.max_attempts,
    )

    # save voxel model to disk
    target_directory = determine_target_directory(
        args.config,
        sample_id,
        STORAGE_FOLDERNAME,
        args.target_dir,
    )
    filename = "voxels.npy"
    file_path = target_directory / filename

    save_voxels(voxels, file_path)

    document_command_execution(
        args.config,
        target_directory,
        CMD_NAME,
        size,
        sample_id,
        inputs={},
        outputs={"voxel_model": str(file_path.expanduser().resolve())},
        metadata=metadata,
        status="success",
    )

    return


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to generate a uniform swiss cheese voxel model"""
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
