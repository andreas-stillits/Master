from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config.declaration import LogLevel, ProjectConfig
from ..config.helpers import deep_update, filter_config_for_command
from ..utilities.checks import verify_existence, verify_extension
from ..utilities.manifest import dump_manifest
from ..utilities.paths import create_target_directory


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
    configname = configname.replace("-", "_")  # normalize possible dash usage

    if hasattr(defaults, configname):
        cmdconfig = getattr(defaults, configname)
        cli_overrides: dict[str, Any] = cmdconfig.model_dump()
        cli_hints = {key: "" for key in cli_overrides.keys()}
        if hasattr(cmdconfig, "cli_hints"):
            cli_hints = deep_update(cli_hints, cmdconfig.cli_hints)

        for key, value in cli_overrides.items():  # passes if empty {}
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):  # contract to defines bools as store_true flags
                parser.add_argument(
                    flag, action="store_true", help=cli_hints.get(key, "")
                )
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
                    argtype = (
                        parse_string_value  # try to interpret complex types from string
                    )
                parser.add_argument(
                    flag, type=argtype, default=value, help=cli_hints.get(key, "")
                )

    return parser


def assemble_cli_overrides(
    args: argparse.Namespace, defaults: ProjectConfig
) -> dict[str, Any]:
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


def add_io_arguments(parser: argparse.ArgumentParser, default_filename: str) -> None:
    """Add common I/O arguments to the given parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which I/O arguments will be added.
        default_filename (str): The default filename to use if none is provided.
    """
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory where output files will be saved (overrides default storage location).",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help=f"Filename for the output file (default: {default_filename}).",
    )
    return


def dump_resolved_command_config(
    config: ProjectConfig, command: str, target_directory: Path
) -> None:
    """
    Dump the resolved configuration for the given command to a file.
    Args:
        config (ProjectConfig): The resolved project configuration.
        command (str): The command name whose relevant configuration to dump.
        target_directory (Path): The directory where the configuration file will be saved.
    """
    command_config = filter_config_for_command(config, command)
    target_path = target_directory / config.meta.config_name
    target_path.write_text(json.dumps(command_config, indent=2, default=str))
    return


def determine_target_and_file_path(
    args: argparse.Namespace,
    cmdconfig: BaseModel,
    default_filename: str,
    *allowed_extensions: str,
) -> tuple[Path, Path]:
    """
    Determine the target directory and file path based on CLI arguments and command configuration.
    Args:
        args (argparse.Namespace): The parsed CLI arguments.
        cmdconfig (BaseModel): The command-specific configuration model.
        allowed_extensions (str): Allowed file extensions for the output filename.
    Returns:
        tuple[Path, Path]: A tuple containing the target directory and file path.
    Note:
        cmdconfig must expose 'storage_foldername'
    """
    # validate cmdconfig has required attribute
    if not hasattr(cmdconfig, "storage_foldername"):
        raise ValueError("cmdconfig must have a 'storage_foldername' attribute")

    # determine target directory
    target_directory = (
        create_target_directory(
            args.config.behavior.storage_root,
            args.sample_id,
            cmdconfig.storage_foldername,  # type: ignore
        )
        if args.output_dir is None
        else Path(args.output_dir)
    )

    verify_existence(target_directory)

    # determine file path
    filename = default_filename if args.filename is None else args.filename
    verify_extension(filename, *allowed_extensions)

    file_path = target_directory / filename

    return target_directory, file_path


def document_command_execution(
    args: argparse.Namespace,
    target_directory: Path,
    command_name: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    metadata: dict[str, Any],
    status: str,
) -> None:
    """
    Document the execution of a command by dumping the resolved configuration and manifest.
    Args:
        args (argparse.Namespace): The parsed CLI arguments.
        target_directory (Path): The directory where documentation files will be saved.
        command_name (str): The name of the executed command.
        inputs (dict[str, str]): A dictionary of input file paths.
        outputs (dict[str, str]): A dictionary of output file paths.
        metadata (dict[str, Any]): Additional metadata about the execution.
        status (str): The status of the execution (e.g., "success", "failure").
    """
    # optionally dump resolved command-relevant config
    if not args.config.behavior.no_cmdconfig:
        dump_resolved_command_config(args.config, command_name, target_directory)

    # optionally dump manifest
    if not args.config.behavior.no_manifest:
        dump_manifest(
            target_directory,
            command_name,
            args.sample_id,
            inputs,
            outputs,
            metadata,
            status,
            args.config.meta.project_version,
        )
    return
