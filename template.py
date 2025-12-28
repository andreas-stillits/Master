from __future__ import annotations

import argparse

from ....config.declaration import ProjectConfig
from ....utilities.checks import validate_sample_id
from ...shared import (
    add_target_directory_argument,
    add_filename_argument,
    derive_cli_flags_from_config,
    determine_target_and_file_path,
    document_command_execution,
)

CMD_NAME = "<...>"
DEFAULT_FILENAME = "<...>"


def _cmd(args: argparse.Namespace) -> None:
    """Command to generate a uniform swiss cheese voxel model"""

    # validate sample ID
    validate_sample_id(args.sample_id, args.config.behavior.sample_id_digits)

    # get resolved config
    config: ProjectConfig = args.config
    cmdconfig = config.<...> 

    # ====

    # command logic
    data = ...

    # ====

    # save voxel model to disk
    target_directory, file_path = determine_target_and_file_path(
        args,
        cmdconfig,
        DEFAULT_FILENAME,
        ".<exts>",
    )
    save(data, file_path)

    document_command_execution(
        args,
        target_directory,
        CMD_NAME,
        inputs={},
        outputs={"<description ...>": str(file_path.expanduser().resolve())},
        metadata={},
        status="success",
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
    add_target_directory_argument(parser)
    add_filename_argument(parser, DEFAULT_FILENAME)
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
