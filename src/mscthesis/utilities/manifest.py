from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .log import log_call


@log_call()
def dump_manifest(
    target_directory: Path,
    command_name: str,
    sample_id: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
    tool_version: str,
) -> None:
    """
    Dump a manifest JSON file summarizing the command execution and output contents.
    Args:
        target_directory (Path): Directory where the manifest file will be saved.
        command_name (str): Name of the command executed.
        sample_id (str): Identifier for the generated sample.
        inputs (dict[str, Any]): Dictionary of input file paths used.
        outputs (dict[str, Any]): Dictionary of output file paths generated.
        metadata (dict[str, Any]): Additional metadata to include in the manifest.
        success (bool): Status of the command execution.
        tool_version (str): Version of the tool used.
    """
    manifest_path = target_directory / "manifest.json"
    manifest: dict[str, Any] = {}
    manifest["generation_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    manifest["command"] = command_name
    manifest["sample_id"] = sample_id
    manifest["inputs"] = inputs
    manifest["outputs"] = outputs
    manifest["meta"] = metadata
    manifest["status"] = status
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    manifest["git_commit"] = git_commit
    manifest["tool"] = f"mscthesis version {tool_version}"

    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, default=str)

    return
