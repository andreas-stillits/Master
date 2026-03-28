from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .log import log_call
from .paths import require_file


@log_call()
def fetch_manifest_quantities(manifest_path: Path, *keys: str) -> dict[str, Any]:
    """
    Fetch a quantity from the manifest of a given sample and process
    Args:
        manifest_path: Path to the manifest file
        keys: Keys of the quantities to fetch from the manifest
    Returns:
        The value of the quantity specified by the key in the manifest
    Raises:
        ValueError: If any of the keys is not found in the manifest
    """
    manifest_path = require_file(manifest_path)

    quantities: dict[str, Any] = {}

    with open(manifest_path, "r") as f:
        manifest_dict = json.load(f)
        for key in keys:
            try:
                quantities[key] = manifest_dict["meta"][key]
            except KeyError as exc:
                raise ValueError(f"Key '{key}' not found in manifest") from exc

    return quantities
