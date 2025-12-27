from __future__ import annotations

import functools
import inspect
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np

from ..config.declaration import LogLevel, ProjectConfig

# we use this "placeholder type" to make sure that typechecking can still resolve
# type hints even after decoration. It allows the function we decorate to pass through
# its information about what arguments it takes and what it returns.
# Had we directly used Callable[..., Any], the type checker would have overwritten e.g. Callable[[int,int], int]
GhostType = TypeVar(
    "GhostType", bound=Callable[..., Any]
)  # decorated object is always a callable (bound)
meta = ProjectConfig().meta  # get meta config for magic strings


def _summarize_value(value: Any, max_length: int = meta.log_summary_max_length) -> str:
    """Crude but practical value summarizer that acts as a Any -> str filter"""
    try:
        # render value in its string representation
        if isinstance(value, (int, float, bool)):
            text = str(value)
        elif isinstance(value, Path):
            # show as str but only last to parts of the path
            text = "..." + "/".join(value.parts[-2:])
        elif isinstance(value, np.ndarray):
            text = f"arr {value.shape} {value.dtype}"
        else:
            text = repr(value)
    except Exception:
        # substitute information about unreprability if applicable
        text = f"<unreprable {type(value).__name__}>"
    # if summary is longer than max_length, cut off and put "..."
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _summarize_args(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> dict[str, str]:
    """Describe the arguments passed to a function as strings"""
    # get the signature obj of the function
    sig = inspect.signature(func)
    try:
        # try to pass the args and kwargs to func and raise TypeError if not applicable
        bound = sig.bind_partial(*args, **kwargs)
        # fill in args, kwargs not assigned (default is empty tuple and empty dict respectively)
        bound.apply_defaults()
        # return dictionary of the individual args/kwargs identifier and assigned value
        return {key: _summarize_value(value) for key, value in bound.arguments.items()}
    except TypeError:
        # if bind_partial() has raised a TypeError, return dict of the raw args and kwargs passed
        return {"args": _summarize_value(args), "kwargs": _summarize_value(kwargs)}


def log_call(
    level: int = logging.INFO, include_result: bool = True
) -> Callable[[GhostType], GhostType]:
    """Decorator function to achieve logging at a certain log.level

    Args:
        level (int): the logging.LEVEL to execute at (default is INFO)
                       will short-circuit if the global loglevel is set higher
        include_result (bool): whether or not to represent the function output
    """

    def decorator(func: GhostType) -> GhostType:
        # format an informative name, e.g. core.io
        qualname = f".{func.__qualname__}"
        logger = logging.getLogger(func.__module__)  # get a per-target-module logger

        # wrapper itself
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not logger.isEnabledFor(level):
                # fast exit if logging is turned off
                return func(*args, **kwargs)

            # describe arguments
            call_args = _summarize_args(func, *args, **kwargs)
            # start timer
            start = time.perf_counter()
            # initial log entry: what level, what function, what inputs?
            logger.log(
                level,
                meta.log_call_start_msg,
                extra={
                    meta.log_call_func_key: qualname,
                    meta.log_call_details_key: f"args = {call_args}",
                },
            )
            # attempt to execute the function, handle if error
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                # if an error, stop the timer
                duration = time.perf_counter() - start
                # record that an error occured
                logger.error(
                    meta.log_call_error_msg,
                    exc_info=True,
                    extra={
                        meta.log_call_func_key: qualname,
                        meta.log_call_details_key: f"args = {call_args} | error_type = {type(exc).__name__} | error_msg = {str(exc)}",
                    },
                )
                raise  # re-raise the error
            # ===
            # if the function executed without error, stop timer
            duration = time.perf_counter() - start
            # commit log entry for end function call
            logger.log(
                level,
                meta.log_call_end_msg,
                extra={
                    meta.log_call_func_key: qualname,
                    meta.log_call_details_key: f"duration = {duration:.3f} s, \t\t results = {_summarize_value(result) if include_result else '-'}",
                },
            )
            return result

        return cast(GhostType, wrapper)

    return decorator


def setup_logging(
    log_filename: Path, log_level: LogLevel, quiet: bool, no_log: bool
) -> logging.Logger:
    """Set up logging configuration for the application.

    Args:
        log_filename (Path): The filename for the log file.
        log_level (LogLevel): The logging level to set.
        quiet (bool): If True, suppress console output.
        no_log (bool): If True, disable file logging.
    """
    import builtins

    level = builtins.getattr(
        logging, log_level
    )  # translate LogLevel enum to logging.LEVEL int
    func_key = meta.log_call_func_key
    details_key = meta.log_call_details_key
    #
    fmt = f"%(asctime)-8s | %(levelname)-8s | %({func_key})-32s | %({details_key})s"
    #
    datefmt = "%H:%M:%S"
    datefmt_full = "%Y-%m-%d %H:%M:%S"

    # clear existing handlers
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)

    # file handler
    if not no_log:
        log_filename.parent.mkdir(parents=True, exist_ok=True)
        sep_count = 80
        # get command line, but omit full path for the first argument (the binary)
        command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
        header_lines = [
            "_" * sep_count,
            f"Run started: {datetime.now().strftime(datefmt_full)}",
            f"Command line: {command}",
            "_" * sep_count,
            "",
        ]
        with log_filename.open("w", encoding="utf-8") as log_file:
            log_file.write("\n".join(header_lines))
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    # console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(console_handler)
    return root


def exit_program_log(logger: logging.Logger, duration: float) -> None:
    """Log program exit information including duration.

    Args:
        duration (float): The total duration of the program execution in seconds.
        logger (logging.Logger): The logger instance to use for logging.
    """
    logger.info(
        "program_exit",
        extra={
            meta.log_call_func_key: ".cli.main",
            meta.log_call_details_key: f"duration = {duration:.3f} s  in total",
        },
    )
