from __future__ import annotations

import argparse

from ....config.declaration import UniformSynthesisConfig
from ....core.io import save_voxels
from ....core.synthesis.uniform import generate_voxels_from_sample_id
from ...shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    setup_command,
)

CMD_NAME = "synthesize-uniform"


def _cmd(args: argparse.Namespace) -> None:
    """Command declaration"""
    paths, config, sample_id = setup_command(args)

    cmdconfig: UniformSynthesisConfig = config.synthesize_uniform

    # generate voxel model
    voxels, metadata = generate_voxels_from_sample_id(
        sample_id,
        cmdconfig.base_seed,
        cmdconfig.resolution,
        cmdconfig.plug_aspect,
        cmdconfig.num_cells,
        cmdconfig.radius,
        cmdconfig.separation,
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
        sample_id,
        inputs={},
        outputs={"voxel_model": str(voxels_path.expanduser().resolve())},
        metadata=metadata,
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
        "sample_id",
        type=str,
        help="A valid sample ID",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
