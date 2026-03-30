from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

from ..config.declaration import LogLevel, ProjectConfig
from ..config.helpers import deep_update, dump_resolved_command_config
from ..utilities.ids import validate_sample_id
from ..utilities.manifest import dump_manifest
from ..utilities.paths import (
    ProcessPathsBase,
    ProjectPaths,
)


def initialize_parsers(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    """
    Add global CLI flags to the given parser. These flags will be available for all commands and subcommands.
    Also initialize the subparsers object and require a command to be given.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which global flags will be added.
    Returns:
        argparse._SubParsersAction[argparse.ArgumentParser]: The subparsers object for adding commands.
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
    """Derive CLI flags from the given command configuration name in ProjectConfig.
    Args:
        parser (argparse.ArgumentParser): The argument parser to which flags will be added.
        configname (str): The name of the command configuration in ProjectConfig.
    Returns:
        argparse.ArgumentParser: The updated argument parser with added flags.
    """
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
            if isinstance(value, bool):  # contract to define bools as store_true flags
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
    """
    Assemble CLI overrides from the given argparse.Namespace object.
    Args:
        args (argparse.Namespace): The parsed CLI arguments.
        defaults (ProjectConfig): The default project configuration.
    Returns:
        dict[str, Any]: A dictionary of CLI overrides to apply to the configuration.
    Notes:
        - Only arguments that differ from the defaults (coded) are included.
        - Supports nested configuration sections but only one level deep.
    """
    defaults_dict = defaults.model_dump()
    keys = defaults_dict.keys()
    args_dict = dict(vars(args))
    args_keys = args_dict.keys()
    cli_overrides: dict[str, Any] = {}

    # translate potential dash to underscore, consistent with config naming
    command_name = args.command.replace("-", "_") if args.command else ""

    for configkey in ["behavior", command_name]:
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
    """
    Try to interpret a string representation, e.g. from CLI input.
    Args:
        raw (str): The raw string input to interpret.
    Returns:
        Any: The interpreted value, or the original string if interpretation fails.
    """
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return value


def setup_command(args: argparse.Namespace) -> tuple[ProjectPaths, ProjectConfig, str]:
    """Perform common pre-command tasks such as validating sample ID and resolving paths.
    Args:
        args (argparse.Namespace): The parsed CLI arguments.
    Returns:
        paths (ProjectPaths): The resolved project paths.
        config (ProjectConfig): The resolved project configuration.
        sample_id (str): The validated sample ID from the CLI arguments.
    """
    paths: ProjectPaths = ProjectPaths(args.config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_inventories_root()
    paths.ensure_processes_root()
    paths.ensure_stats_root()

    config: ProjectConfig = args.config

    sample_id = args.sample_id.strip()
    sample_id = validate_sample_id(sample_id, config.behavior.sample_id_digits)

    return paths, config, sample_id


def document_command_execution(
    process_paths: ProcessPathsBase,
    config: ProjectConfig,
    command_name: str,
    sample_id: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """
    Document the execution of a command by dumping the resolved configuration and manifest.
    Args:
        process_paths (ProcessPathsBase): The process paths for the current sample.
        config (ProjectConfig): The resolved project configuration for the command execution.
        command_name (str): The name of the executed command.
        sample_id (str): The sample ID for which the command was executed.
        inputs (dict[str, str]): A dictionary of input names and their values/paths.
        outputs (dict[str, str]): A dictionary of output names and their values/paths.
        metadata (dict[str, Any]): A dictionary of additional metadata to include in the manifest.
    """
    # optionally dump resolved command-relevant config
    if not config.behavior.no_cmdconfig:
        dump_resolved_command_config(config, command_name, process_paths.config)

    # optionally dump manifest
    if not config.behavior.no_manifest:
        dump_manifest(
            process_paths.manifest,
            command_name,
            sample_id,
            inputs,
            outputs,
            metadata,
            config.meta.project_version,
        )
    return
