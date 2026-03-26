from __future__ import annotations

import argparse

from ...config.declaration import SolutionConfig
from ...core.io import load_volumetric_mesh, save_fem_solution
from ..shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    setup_command,
)

CMD_NAME = "solve"


def _cmd(args: argparse.Namespace) -> None:
    """Command declaration"""
    paths, config, sample_id = setup_command(args)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="solve the diffusion problem for a given choice of parameters",
        help="solve the diffusion problem for a given choice of parameters",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="A valid sample ID",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
