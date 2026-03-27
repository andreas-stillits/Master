from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.declaration import ProjectConfig
from .log import log_call
from .paths import ProjectPaths


@log_call()
def fetch_manifest_quantity(
    config: ProjectConfig, sample_id: str, process: str, *keys: str
) -> dict[str, Any]:
    """
    Fetch a quantity from the manifest of a given sample and process
    Args:
        config: ProjectConfig object containing the configuration of the project
        sample_id: ID of the sample to fetch the quantity from
        process: Name of the process to fetch the quantity from
        keys: Keys of the quantities to fetch from the manifest
    Returns:
        The value of the quantity specified by the key in the manifest
    Raises:
        ValueError: If any of the keys is not found in the manifest
    """
    paths: ProjectPaths = ProjectPaths(config.behavior.storage_root)
    manifest_path: Path = (
        paths.sample(sample_id).require_process(process).require_manifest()
    )
    quantities: dict[str, Any] = {}
    with open(manifest_path, "r") as f:
        manifest_dict = json.load(f)
        for key in keys:
            try:
                quantities[key] = manifest_dict["meta"][key]
            except KeyError as exc:
                raise ValueError(f"Key '{key}' not found in manifest") from exc

    return quantities
