from __future__ import annotations

import argparse

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
    """Command to generate a uniform swiss cheese voxel model"""

    # validate sample ID
    validate_sample_id(args.sample_id, args.config.behavior.sample_id_digits)

    # get resolved config
    cmdconfig: UniformSynthesisConfig = args.config.synthesize_uniform

    # generate voxel model
    voxels = generate_uniform_swiss_voxels(
        args.sample_id,
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
        args.sample_id,
        args.target_dir,
    )
    filename = "voxels.npy"
    file_path = target_directory / filename

    save_voxel_model(voxels, file_path)

    document_command_execution(
        args.config,
        target_directory,
        CMD_NAME,
        args.sample_id,
        inputs={},
        outputs={"voxel_model": str(file_path.expanduser().resolve())},
        metadata={},
        status="success",
    )

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
        "sample_id", type=str, help="Unique identifier for the generated sample"
    )
    add_target_directory_argument(parser)
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
