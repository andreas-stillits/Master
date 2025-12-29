from __future__ import annotations

import argparse
import time

from mpi4py import MPI

from ....config.declaration import UniformSynthesisConfig
from ....core.synthesis.helpers import save_voxel_model
from ....core.synthesis.uniform import generate_uniform_swiss_voxels
from ....utilities.checks import validate_sample_id
from ...shared import (
    add_target_directory_argument,
    derive_cli_flags_from_config,
    determine_target_directory,
    document_command_execution,
)

CMD_NAME = "synthesize-uniform"


def _cmd(args: argparse.Namespace) -> None:
    """Command to MPI batch generate a uniform swiss cheese voxel model"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Starting MPI batch uniform synthesis with {size} processes...")
        start = time.perf_counter()

    # get resolved config
    cmdconfig: UniformSynthesisConfig = args.config.synthesize_uniform

    # determine which sample ID this rank is responsible for
    with open(args.sample_id_file_path, "r") as f:
        sample_ids = [line.strip() for line in f if line.strip()]
        if rank >= len(sample_ids):
            print(f"Rank {rank} has no sample ID to process. Exiting.")
            return
        rank_sample_ids = sample_ids[rank::size]

    for sample_id in rank_sample_ids:

        # validate sample ID
        validate_sample_id(sample_id, args.config.behavior.sample_id_digits)

        # generate voxel model
        voxels = generate_uniform_swiss_voxels(
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
            CMD_NAME,
            sample_id,
            args.target_dir,
        )

        filename = "voxels.npy"
        file_path = target_directory / filename

        save_voxel_model(voxels, file_path)

        document_command_execution(
            args.config,
            target_directory,
            CMD_NAME,
            sample_id,
            inputs={},
            outputs={"voxel_model": str(file_path.expanduser().resolve())},
            metadata={},
            status="success",
        )

    if rank == 0:
        end = time.perf_counter()
        duration = end - start  # type: ignore
        print(f"MPI batch uniform synthesis completed in {duration:.2f} seconds.")
        print(f"Processed {len(sample_ids)} samples.")
        print(
            f"Estimated total execution time for single process: {duration * size:.2f} seconds."
        )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the mpibatch synthesize uniform voxel model command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="MPI batch command to generate a uniform swiss cheese voxel model",
        help="Generate a uniform swiss cheese voxel model using MPI batch processing",
        epilog="mpirun -n <num_proc> msc mpibatch synthesize-uniform [options] <sample_id_file_path>.txt",
    )
    parser.add_argument(
        "sample_id_file_path",
        type=str,
        help="A .txt file containing the sample IDs to batch over",
    )
    add_target_directory_argument(parser)
    # --- Here we would have to add a wrapping in a sample ID folder per ranks output, not spray all in the same dir directly
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
