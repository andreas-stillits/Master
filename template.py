from __future__ import annotations

import argparse

from ....config.declaration import ProjectConfig
from ....utilities.checks import validate_sample_id
from ....utilities.manifest import dump_manifest
from ....utilities.paths import create_target_directory
from ...shared import derive_cli_flags_from_config, dump_resolved_command_config

CMD_NAME = "<...>"


def _cmd(args: argparse.Namespace) -> None:
    """Command to generate a uniform swiss cheese voxel model"""

    # validate sample ID
    validate_sample_id(args.sample_id, args.config.behavior.sample_id_digits)

    # get resolved config
    config: ProjectConfig = args.config

    # ====

    # command logic
    data = ...

    # ====

    # save to disk
    target_directory = create_target_directory(
        args.config.behavior.storage_root,
        args.sample_id,
        config.storage_foldername,
    )
    file_path = target_directory / "<filename>.<ext>"
    # save(data, file_path)

    # optionally dump resolved command-relevant config
    if not args.config.behavior.no_cmdconfig:
        dump_resolved_command_config(args.config, CMD_NAME, target_directory)

    # optionally dump manifest
    if not args.config.behavior.no_manifest:
        inputs: dict[str, str] = {}
        outputs: dict[str, str] = {"data": str(file_path.expanduser().resolve())}
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
    """<...>"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="<...>",
        help="<...>",
        epilog="<...>",
    )
    parser.add_argument(
        "sample_id", type=str, help="Unique identifier for the generated sample"
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
