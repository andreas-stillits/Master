from __future__ import annotations

import argparse

from ..shared import derive_cli_flags_from_config


def _cmd(args: argparse.Namespace) -> None:
    """Command to copy the current settings to a specified file in JSON format."""
    # get resolved config
    # config: ProjectConfig = args.config

    # call library functions
    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    cmd_name = "test"
    parser = subparsers.add_parser(
        cmd_name,
        description="Luffe er lækker!",
        help="Og det er pointen!",
        epilog="siger det bare ...",
    )
    parser = derive_cli_flags_from_config(parser, cmd_name)
    parser.set_defaults(cmd=_cmd)
