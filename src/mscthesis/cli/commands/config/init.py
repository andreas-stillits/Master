from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from ....config.declaration import ProjectConfig


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    config: ProjectConfig = (
        args.config
    )  # always a defaults instance due to cli.main:main structure

    # get path to write to depending on args.user
    path: Path = (
        config.meta.project_config_path
        if not args.user
        else config.meta.user_config_path
    )

    # if file exists and user did not force override
    if path.is_file() and not args.force:
        print(
            f"Config file already exists at: {path}. Use --force to overwrite with defaults. "
        )
        return
    # else override with defaults
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.dump_json())
    print(f"Wrote default config to: {path}.")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'config init' subcommand on the given subparsers object."""
    parser = subparsers.add_parser(
        "init",  # command name
        description="Initialize a default config.json file in the user's home directory.",
        help="Initialize a default config.json file in the user's home directory.",
        epilog="Example (hard reset user-level config): msc config -u init --force",
    )
    parser.add_argument(
        "--force",  # optional flag to overwrite existing file and recreate hardcoded defaults
        action="store_true",
        help="Overwrite existing config.json file if it exists.",
    )
    parser.set_defaults(cmd=_cmd)  # associate 'cmd_init_settings' with args.cmd
