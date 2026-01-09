from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Callable

from mpi4py import MPI
from tqdm import tqdm

from ..config.declaration import LogLevel, ProjectConfig
from ..config.helpers import deep_update, filter_config_for_command
from ..utilities.ids import validate_sample_id
from ..utilities.manifest import dump_manifest
from ..utilities.paths import (
    ProcessPathsBase,
    ProjectPaths,
    resolve_existing_inventories_file,
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


def distribute_command_execution(
    args: argparse.Namespace, comm: MPI.Intracomm, execute_single_sample_id: Callable
) -> None:
    """Command declaration"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    paths: ProjectPaths = ProjectPaths(args.config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_inventories_root()

    sample_ids = interpret_sample_input(
        paths,
        args.sample_input,
        args.config.behavior.sample_id_digits,
    )

    # early exit if less samples than workers - also cathes the case of zero samples:
    if rank >= len(sample_ids) or len(sample_ids) == 0:
        return

    # distribute sample IDs among workers
    assigned_sample_ids = sample_ids[rank::size]
    if rank == 0 and size > 1:
        for sample_id in tqdm(assigned_sample_ids, desc="processing samples..."):
            execute_single_sample_id(paths, args.config, sample_id, size)
    else:
        for sample_id in assigned_sample_ids:
            execute_single_sample_id(paths, args.config, sample_id, size)

    return


def interpret_sample_input(
    paths: ProjectPaths, input: str, required_digits: int
) -> list[str]:
    """
    Interpret the sample input argument and return a list of sample IDs
    Args:
        input (str): The sample input, either a single sample ID or a path to a text file
                     containing multiple sample IDs (one per line).
        required_digits (int): The required length of each sample ID.
    Returns:
        list[str]: A list of valid sample IDs.
    """
    sample_ids: list[str] = []
    # check if input has .txt extension or begins with "@"
    if input.startswith("@") or input.endswith(".txt"):
        # expand path
        input_path = resolve_existing_inventories_file(paths, input, ".txt")
        # read sample IDs from file
        with open(input_path, "r") as f:
            for line in f:
                sample_id = line.strip()
                if sample_id and validate_sample_id(sample_id, required_digits):
                    sample_ids.append(sample_id)
    else:
        # treat input as single sample ID
        sample_id = input.strip()
        if validate_sample_id(sample_id, required_digits):
            sample_ids.append(sample_id)
    return sample_ids


def dump_resolved_command_config(
    config: ProjectConfig, command: str, target_path: Path
) -> None:
    """
    Dump the resolved configuration for the given command to a file.
    Args:
        config (ProjectConfig): The resolved project configuration.
        command (str): The command name whose relevant configuration to dump.
        target_path (Path): The path where the configuration file will be saved.
    """
    command_config = filter_config_for_command(config, command)
    target_path.write_text(json.dumps(command_config, indent=2, default=str))
    return


def document_command_execution(
    process_paths: ProcessPathsBase,
    config: ProjectConfig,
    command_name: str,
    num_processes: int,
    sample_id: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """
    Document the execution of a command by dumping the resolved configuration and manifest.
    Args:
        args (argparse.Namespace): The parsed CLI arguments.
        process_paths (ProcessPathsBase): The process paths where documentation files will be saved.
        command_name (str): The name of the executed command.
        inputs (dict[str, str]): A dictionary of input file paths.
        outputs (dict[str, str]): A dictionary of output file paths.
        metadata (dict[str, Any]): Additional metadata about the execution.
        status (str): The status of the execution (e.g., "success", "failure").
    """
    # optionally dump resolved command-relevant config
    if not config.behavior.no_cmdconfig:
        dump_resolved_command_config(config, command_name, process_paths.config)

    # optionally dump manifest
    if not config.behavior.no_manifest:
        dump_manifest(
            process_paths.manifest,
            command_name,
            num_processes,
            sample_id,
            inputs,
            outputs,
            metadata,
            config.meta.project_version,
        )
    return
