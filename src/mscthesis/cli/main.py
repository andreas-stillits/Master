from __future__ import annotations

import argparse
import time

from ..config.declaration import ProjectConfig
from ..config.helpers import build_project_config
from ..utilities.log import exit_program_log, setup_logging
from .commands.config import copy as config_copy
from .commands.config import get as config_get
from .commands.config import init as config_init
from .commands.config import set as config_set
from .commands.config import show as config_show
from .shared import (
    assemble_cli_overrides,
    derive_cli_flags_from_config,
    initialize_parsers,
)

# attempt to facilitate terminal auto completion for cli commands
try:
    import argcomplete
except ImportError:
    argcomplete = None


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the CLI."""
    # create global parser
    parser = argparse.ArgumentParser(
        prog="mscthesis", description="Master Thesis Command Line Interface"
    )
    parser = derive_cli_flags_from_config(
        parser, "behavior"
    )  # key must match the name of BehaviorConfig in ProjectConfig
    subparsers = initialize_parsers(parser)

    # === CONFIG COMMANDS ===
    # Wire in config related commands with an umbrella "config" command
    config_parser = subparsers.add_parser(
        "config",
        help="Commands related to project configuration setup and maintenance.",
    )
    config_parser.add_argument(
        "-u",
        "--user",
        action="store_true",
        help="Apply config commands to the user config in the home directory.",
    )
    config_subparsers = config_parser.add_subparsers(
        title="config_commands",
        dest="config_command",  # store chosen config command in args.config_command
    )
    config_subparsers.required = True
    # wire in possible config <subcommand>
    config_init.add_parser(config_subparsers)
    config_show.add_parser(config_subparsers)
    config_copy.add_parser(config_subparsers)
    config_get.add_parser(config_subparsers)
    config_set.add_parser(config_subparsers)
    # ... add more commands here that act as subcommands of "config ..."
    # ...
    # === OTHER COMMANDS ===

    # ... add more top-level commands here ...
    # either:
    #   <command>_parser = ...
    #   <command>_subparsers = ...
    #   <subcommand_module>.add_parser(<command>_subparsers)
    #
    # or directly:
    #   <command_module>.add_parser(subparsers) if no umbrella command is needed

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = _build_parser()

    # Enable tab completion if argcomplete is available (user must run: 'eval "$(register-python-argcomplete mscthesis)"'
    # in bash per session or add to .bashrc (I did))
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)

    # get default config
    defaults: ProjectConfig = ProjectConfig()

    if hasattr(args, "config_command") and args.config_command == "init":
        # execute init command
        args.config = defaults  # defaults
        args.cmd(args)
        return 0

    # if the user has not initialized a config file in their home directory, ask them to:
    if not defaults.meta.user_config_path.is_file():
        print(
            f"User config file not found at: {defaults.meta.user_config_path}. "
            "Please run 'msc config --user init' to initialize project configuration."
        )
        print(
            "NICETY: Optionally add the line 'eval '$(register-python-argcomplete mscthesis)'' "
            "to your shell profile (.bashrc/.zshrc) for cli autocompletion. "
        )
        return 1  # error code in the terminal
    # if project config has been initialized
    else:
        # resolve config after PROJECT (cwd) > USER (home) > DEFAULTS (code)
        config: ProjectConfig = build_project_config(
            args.config, overrides=assemble_cli_overrides(args, defaults)
        )  # here config points to a potential CLI passed path
        # overriding args.config to mean the resolved ProjectConfig instance
        args.config = config  # pass resolved config to args for commands to use
        b = config.behavior
        logger = setup_logging(
            b.storage_root / b.log_filename, b.log_level, b.quiet, b.no_log
        )

        # Call the function associated with the chosen command
        if hasattr(args, "cmd"):

            # commands should associate parser.set_defaults(cmd=...) with the cmd function
            start_time = time.perf_counter()

            args.cmd(args)

            duration = time.perf_counter() - start_time

            # log total command execution time
            exit_program_log(logger, duration)
        else:
            print(f"Unrecognized command: {argv}")
            parser.print_help()  # print top level help if not a valid command

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
