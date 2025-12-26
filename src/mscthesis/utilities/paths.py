from __future__ import annotations

from pathlib import Path

from .log import log_call


@log_call()
def get_samples_path(storage_root: Path) -> Path:
    """Get the path to the samples directory."""
    samples_path = storage_root / "samples"
    samples_path.mkdir(parents=True, exist_ok=True)
    return samples_path
