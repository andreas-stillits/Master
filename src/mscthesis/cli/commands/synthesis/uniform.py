from __future__ import annotations

import argparse

from ....config.declaration import UniformSynthesisConfig
from ....core.synthesis.helpers import save_voxel_model
from ....core.synthesis.uniform import generate_uniform_swiss_voxels
from ....utilities.paths import get_samples_path
from ...shared import derive_cli_flags_from_config


def _cmd(args: argparse.Namespace) -> None:
    """Command to copy the current settings to a specified file in JSON format."""

    # validate sample ID length
    required_chars = args.config.behavior.num_sample_id_chars
    if not len(args.sample_id) == required_chars:
        raise ValueError(
            f"Sample ID '{args.sample_id}' does not match required "
            f"length of {required_chars} characters."
        )

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

    samples_path = get_samples_path(args.config.behavior.storage_root)

    # create target: <storage_root>/<samples>/<sample_id>/synthesis/...
    target_dir = samples_path / args.sample_id / config.storage_foldername
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = target_dir / "voxels.npy"

    save_voxel_model(voxels, filename)

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
