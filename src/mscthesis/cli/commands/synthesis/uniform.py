from __future__ import annotations

import argparse

from mpi4py import MPI

from ....config.declaration import ProjectConfig, UniformSynthesisConfig
from ....core.io import save_voxels
from ....core.synthesis.uniform import generate_voxels_from_sample_id
from ....utilities.paths import ProjectPaths
from ...shared import (
    derive_cli_flags_from_config,
    distribute_command_execution,
    document_command_execution,
)

CMD_NAME = "synthesize-uniform"


def _execute_single_sample_id(
    paths: ProjectPaths, config: ProjectConfig, sample_id: str, size: int
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

    process_paths = paths.sample(sample_id).synthesis()
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
    return distribute_command_execution(args, comm, _execute_single_sample_id)


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
