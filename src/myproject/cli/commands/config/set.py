from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ....config.declaration import ProjectConfig
from ....config.helpers import load_config_from_file
from ...shared import parse_string_value


def _cmd(args: argparse.Namespace) -> None:
    """Command to set a configuration key to a specified value."""
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
    # get the path to write to depending on args.user
    path: Path = config.meta.project_config_path if not args.user else config.meta.user_config_path
    # walk config dict according to cli passed dot-string
    current = config_dict
    parts = args.key.split(".")  # separate keys
    # for all but the last key (used for assignment)
    for part in parts[:-1]:
        # if key not found or not pointing to a dictionary
        if part not in current or not isinstance(current[part], dict):
            raise ValueError(f"{args.key} is not a valid key")
        # update the dict current points to
        current = current[part]
    # assign the new value parsed as string through cli
    current[parts[-1]] = parse_string_value(args.value)  # Any | None if key not truthful

    try:
        config = ProjectConfig.model_validate(config_dict)  # try to validate updated config
    except Exception as exc:
        raise ValueError(
            f"Setting '{args.key}' to '{args.value}' results in invalid config: {exc}"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.dump_json())
    print(f"Set '{args.key}' to '{args.value}' in config file at: {path}.")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'config [-u] set <key> <value>' command on the given subparsers object."""
    parser = subparsers.add_parser(
        "set",
        description="Set a specific configuration attribute via a dot-formatted key.",
        help="Set a specific configuration attribute via a dot-formatted key.",
        epilog="Example: <myproject> config set behavior.log_level WARNING \n",
    )
    parser.add_argument(
        "key",
        type=str,
        help="Name of the config attribute to set.",
    )
    parser.add_argument(
        "value",
        type=str,
        help="Value to assign to the config attribute",
    )

    parser.set_defaults(cmd=_cmd)
