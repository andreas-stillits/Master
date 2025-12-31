from __future__ import annotations

import argparse

from mpi4py import MPI

from ....config.declaration import ProjectConfig
from ....config.helpers import load_config_from_file


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to print the resolved (user or project) config to stdout in JSON format."""
    config: ProjectConfig = args.config
    if not args.user:
        print("Current resolved project-level configuration:")
        print(config.dump_json())
    else:
        # default to user config in home directory
        print("Current user-level configuration:")
        # load user config from home directory and validate with pydantic model before printing
        print(
            ProjectConfig.model_validate(
                load_config_from_file(config.meta.user_config_path)
            ).dump_json()
        )


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'config [-u] show' command on the given subparsers object."""
    parser = subparsers.add_parser(
        "show",
        description="Show current config in JSON format.",
        help="Show the current config in JSON format. (default: project-level config, use -u for user-level config)",
        epilog="Example: msc config [-u] show \n",
    )

    parser.set_defaults(cmd=_cmd)
