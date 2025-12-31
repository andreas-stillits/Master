from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from ....config.declaration import ProjectConfig


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to copy the current settings to a specified file in JSON format."""
    # get resolved config
    config: ProjectConfig = args.config
    # get output path from args (str -> Path)
    output_path = Path(args.output_path)
    # check that the parent directory exists
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Parent directory does not exist: {output_path.parent}. Provide a valid path."
        )
    # write config to output path in JSON format
    output_path.write_text(config.dump_json())


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'config copy <path>' command on the given subparsers object."""
    parser = subparsers.add_parser(
        "copy",
        description="Copy current resolved config in JSON format.",
        help="Copy the current resolved config in JSON format.",
        epilog="Example: msc config copy ./config_backup.json",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to output the copied configuration JSON file.",
    )

    parser.set_defaults(cmd=_cmd)
