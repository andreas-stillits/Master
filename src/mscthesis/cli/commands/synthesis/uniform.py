from __future__ import annotations

import argparse

from ....config.declaration import UniformSynthesisConfig
from ....core.synthesis.helpers import save_voxel_model
from ....core.synthesis.uniform import generate_uniform_swiss_voxels
from ....utilities.checks import validate_sample_id
from ....utilities.paths import create_target_directory
from ...shared import derive_cli_flags_from_config


def _cmd(args: argparse.Namespace) -> None:
    """Command to generate a uniform swiss cheese voxel model"""

    # validate sample ID
    validate_sample_id(args.sample_id, args.config.behavior.sample_id_digits)

    # get resolved config
    config: UniformSynthesisConfig = args.config.synthesize_uniform

    voxels = generate_uniform_swiss_voxels(
        args.sample_id,
        config.base_seed,
        config.resolution,
        config.plug_aspect,
        config.num_cells,
        config.min_radius,
        config.max_radius,
        config.min_separation,
        config.max_attempts,
    )

    target = create_target_directory(
        args.config.behavior.storage_root,
        args.sample_id,
        config.storage_foldername,
    )
    file_path = target / "voxels.npy"
    save_voxel_model(voxels, file_path)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the synthesize uniform voxel model command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    cmd_name = "synthesize-uniform"
    parser = subparsers.add_parser(
        cmd_name,
        description="generate a uniform swiss cheese voxel model",
        help="generate a uniform swiss cheese voxel model",
        epilog=f"msc {cmd_name} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_id", type=str, help="Unique identifier for the generated sample"
    )
    parser = derive_cli_flags_from_config(parser, cmd_name)
    parser.set_defaults(cmd=_cmd)
