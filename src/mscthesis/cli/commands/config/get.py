from __future__ import annotations

import argparse
from typing import Any

from ....config.declaration import ProjectConfig
from ....config.helpers import load_config_from_file


def _cmd(args: argparse.Namespace) -> None:
    """Command to get a specific configuration attribute via a dot-formated key."""
    # get resolved config
    config: ProjectConfig = args.config

    # get config as dictionary - project-level as default and user-level if args.user
    config_dict: dict[str, Any] = (
        config.model_dump()
        if not args.user
        else ProjectConfig.model_validate(
            load_config_from_file(config.meta.user_config_path)
        ).model_dump()
    )
    # walk the dictionary according to dot-formated key
    value = config_dict
    for part in args.key.split("."):
        if part not in value:
            raise ValueError(f"Config has no attribute '{args.key}'")
        value = value[part]
    print(f"Attribute '{args.key}' of type {type(value).__name__} has value: {value}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'config get <key>' command on the given subparsers object."""
    parser = subparsers.add_parser(
        "get",
        description="Get a specific configuration attribute via a dot-formatted key.",
        help="Get a specific configuration attribute via a dot-formatted key.",
        epilog="Example: msc config -u get behavior.log_level \n",
    )
    parser.add_argument(
        "key",
        type=str,
        help="Name of the configuration attribute to get - pass as a dot-separated string.",
    )

    parser.set_defaults(cmd=_cmd)
