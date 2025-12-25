from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

from ..config.declaration import LogLevel, ProjectConfig
from ..config.helpers import deep_update


def initialize_parsers(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    """Add global CLI flags to the given parser. These flags will be available for all commands and subcommands.
        Also initialize the subparsers object and require a command to be given.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which global flags will be added.
    """
    # add global flags
    default_config_path = ProjectConfig().meta.project_config_path
    # add a flag to specify a project config file path
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config_path,
        help=f"Path to a config JSON file for process overrides (default: {default_config_path}).",
    )

    # initialize subparsers
    subparsers = parser.add_subparsers(
        title="commands",  # group listed commands under "commands" in help output
        dest="command",  # store chosen command name in args.command
    )
    subparsers.required = True  # error if no command is given

    return subparsers


def derive_cli_flags_from_config(
    parser: argparse.ArgumentParser, configname: str
) -> argparse.ArgumentParser:
    # init defaults and derive dictionary form for cli overrides
    defaults = ProjectConfig()

    if hasattr(defaults, configname):
        cmdconfig = getattr(defaults, configname)
        cli_overrides: dict[str, Any] = cmdconfig.model_dump()
        cli_hints = {key: "" for key in cli_overrides.keys()}
        if hasattr(cmdconfig, "cli_hints"):
            cli_hints = deep_update(cli_hints, cmdconfig.cli_hints)

        for key, value in cli_overrides.items():  # passes if empty {}
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):  # contract to defines bools as store_true flags
                parser.add_argument(flag, action="store_true", help=cli_hints.get(key, ""))
            else:
                # pick sensible type for argparse where possible
                if isinstance(value, Path):
                    argtype = Path
                elif isinstance(value, LogLevel):
                    argtype = LogLevel
                elif isinstance(value, int):
                    argtype = int
                elif isinstance(value, float):
                    argtype = float
                elif isinstance(value, str):
                    argtype = str
                else:
                    argtype = parse_string_value  # try to interpret complex types from string
                parser.add_argument(flag, type=argtype, default=value, help=cli_hints.get(key, ""))

    return parser


def assemble_cli_overrides(args: argparse.Namespace, defaults: ProjectConfig) -> dict[str, Any]:
    """Assemble CLI overrides from the given argparse.Namespace object."""
    defaults_dict = defaults.model_dump()
    keys = defaults_dict.keys()
    args_dict = dict(vars(args))
    args_keys = args_dict.keys()
    cli_overrides: dict[str, Any] = {}

    for configkey in ["behavior", args.command]:
        # assemble subdict for each config section
        if configkey in keys:
            cmd_defaults: dict[str, Any] = defaults_dict[configkey]
            subdict: dict[str, Any] = {}
            for cmdkey, cmdvalue in cmd_defaults.items():
                if cmdkey in args_keys:
                    value = args_dict[cmdkey]
                    # normalize Path / str comparisons
                    if isinstance(cmdvalue, Path) and value is not None:
                        value = Path(value)
                    # only include if different from defaults
                    if value != cmdvalue:
                        subdict[cmdkey] = value
            # if subdict is not empty, add to cli_overrides
            if subdict:
                cli_overrides[configkey] = subdict

    return cli_overrides


def parse_string_value(raw: str) -> Any:
    """Try to interpret a string representation, e.g. from CLI input."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return value
