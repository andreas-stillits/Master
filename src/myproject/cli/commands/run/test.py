"""Template for a cli command"""

from __future__ import annotations

import argparse
import numpy as np

from ....utilities.log import log_call
from ...shared import derive_cli_flags_from_config


class Dummy:
    name = "dummy"
    number = 42
    truth = False


@log_call()
def loading(x: np.ndarray) -> np.ndarray:
    return x + 1


@log_call()
def action(x: np.ndarray) -> np.ndarray:
    return x + 1


@log_call()
def saving(x: np.ndarray, dummy: Dummy) -> np.ndarray:
    return x, dummy


def _cmd(args: argparse.Namespace) -> None:
    """Command to copy the current settings to a specified file in JSON format."""
    # get resolved config
    # config: ProjectConfig = args.config
    x = np.linspace(1, 10, 100)
    x = loading(x)
    x = action(x)
    x, dummy = saving(x, Dummy())

    # call library functions
    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    cmd_name = "some_command"
    parser = subparsers.add_parser(
        cmd_name,
        description="some desription ...",
        help="some help messages",
        epilog="handy example usages ...",
    )
    parser = derive_cli_flags_from_config(parser, cmd_name)
    parser.set_defaults(cmd=_cmd)
