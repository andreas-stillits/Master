from __future__ import annotations

import argparse
from pathlib import Path

from ....config.declaration import UniformSynthesisConfig
from ....core.synthesis.helpers import save_voxel_model
from ....core.synthesis.uniform import generate_uniform_swiss_voxels
from ....utilities.checks import validate_sample_id, verify_existence, verify_extension
from ....utilities.manifest import dump_manifest
from ....utilities.paths import create_target_directory
from ...shared import derive_cli_flags_from_config, dump_resolved_command_config

CMD_NAME = "synthesize-uniform"
DEFAULT_FILENAME = "voxels.npy"


def _cmd(args: argparse.Namespace) -> None:
    """Command to generate a uniform swiss cheese voxel model"""

    # validate sample ID
    validate_sample_id(args.sample_id, args.config.behavior.sample_id_digits)

    # get resolved config
    config: UniformSynthesisConfig = args.config.synthesize_uniform

    # generate voxel model
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

    # save voxel model to disk

    target_directory = (
        create_target_directory(
            args.config.behavior.storage_root,
            args.sample_id,
            config.storage_foldername,
        )
        if args.output_dir is None
        else Path(args.output_dir)
    )
    verify_existence(target_directory)

    filename = DEFAULT_FILENAME if args.filename is None else args.filename
    verify_extension(filename, ".npy")

    file_path = target_directory / filename
    save_voxel_model(voxels, file_path)

    # optionally dump resolved command-relevant config
    if not args.config.behavior.no_cmdconfig:
        dump_resolved_command_config(args.config, CMD_NAME, target_directory)

    # optionally dump manifest
    if not args.config.behavior.no_manifest:
        inputs: dict[str, str] = {}
        outputs: dict[str, str] = {"voxel_model": str(file_path.expanduser().resolve())}
        metadata: dict[str, str] = {}

        dump_manifest(
            target_directory,
            CMD_NAME,
            args.sample_id,
            inputs,
            outputs,
            metadata,
            "success",
            args.config.meta.project_version,
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
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Optional output directory (defaults to sample folder in storage root)",
    )
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        help=f"Optional filename for the voxel model (defaults to '{DEFAULT_FILENAME}')",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
