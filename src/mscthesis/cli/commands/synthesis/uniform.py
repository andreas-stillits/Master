from __future__ import annotations

import argparse

from mpi4py import MPI

from ....config.declaration import ProjectConfig, UniformSynthesisConfig
from ....core.io import save_voxels
from ....core.synthesis.uniform import generate_voxels_from_sample_id
from ....utilities.paths import Paths
from ...shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    interpret_sample_input,
)

CMD_NAME = "synthesize-uniform"
STORAGE_FOLDERNAME = "synthesis"


def _execute_single_sample_id(
    paths: Paths, config: ProjectConfig, sample_id: str, size: int
) -> None:
    """Execute process for a single sample ID"""
    # get resolved config
    cmdconfig: UniformSynthesisConfig = config.synthesize_uniform

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
    sample_path = paths.sample(sample_id)
    sample_path.ensure_dir()

    process_paths = sample_path.synthesis()
    process_paths.ensure_dir()
    voxels_path = process_paths.voxels

    save_voxels(voxels, voxels_path)

    document_command_execution(
        process_paths,
        config,
        CMD_NAME,
        size,
        sample_id,
        inputs={},
        outputs={"voxel_model": str(voxels_path.expanduser().resolve())},
        metadata=metadata,
    )

    return


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command declaration"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    paths: Paths = Paths(args.config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_inventories_root()

    sample_ids = interpret_sample_input(
        paths,
        args.sample_input,
        args.config.behavior.sample_id_digits,
    )

    # early exit if less samples than workers - also cathes the case of zero samples:
    if rank >= len(sample_ids) or len(sample_ids) == 0:
        return

    # distribute sample IDs among workers
    assigned_sample_ids = sample_ids[rank::size]
    for sample_id in assigned_sample_ids:
        _execute_single_sample_id(paths, args.config, sample_id, size)

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
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
