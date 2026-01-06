from __future__ import annotations

from pathlib import Path

from .log import log_call

SAMPLES_FOLDERNAME: str = "samples"
INVENTORIES_FOLDERNAME: str = "inventories"


@log_call()
def get_samples_path(storage_root: Path) -> Path:
    """
    Get the path to the samples directory
    Args:
        storage_root (Path): The root storage directory.
    Returns:
        Path: The path to the samples directory.
    """
    samples_path = storage_root / SAMPLES_FOLDERNAME
    samples_path.mkdir(parents=True, exist_ok=True)
    return samples_path


@log_call()
def expand_samples_path(storage_root: Path, relative_path: str) -> Path:
    """
    Derive full path within storage root given a relative path if signified by @.
    Else, return as asolute Path.
    Args:
        storage_root (Path): The root storage directory.
        relative_path (str): The relative path, potentially prefixed with '@' to indicate storage root.
    Returns:
        full_path (Path): The derived full path.
    """
    if relative_path.startswith("@"):
        relative_path = relative_path[1:]  # remove '@' prefix
        full_path = get_samples_path(storage_root) / relative_path
    else:
        full_path = Path(relative_path)
    return full_path


@log_call()
def get_inventory_path(storage_root: Path) -> Path:
    """
    Get the path to the inventory directory
    Args:
        storage_root (Path): The root storage directory.
    Returns:
        Path: The path to the inventory directory.
    """
    inventory_path = storage_root / INVENTORIES_FOLDERNAME
    inventory_path.mkdir(parents=True, exist_ok=True)
    return inventory_path


@log_call()
def expand_inventory_path(storage_root: Path, relative_path: str) -> str:
    """
    Derive full path within inventory directory given a relative path if signified by @.
    Else, return as absolute Path.
    Args:
        storage_root (Path): The root storage directory.
        relative_path (str): The relative path, potentially prefixed with '@' to indicate inventory directory.
    Returns:
        full_path (Path): The derived full path.
    """
    if relative_path.startswith("@"):
        relative_path = relative_path[1:]  # remove '@' prefix
        full_path = get_inventory_path(storage_root) / relative_path
    else:
        full_path = relative_path

    return str(full_path)


@log_call()
def create_target_directory(
    storage_root: Path, sample_id: str, storage_foldername: str
) -> Path:
    """
    Create and return the target directory for storing sample output data.
    Args:
        storage_root (Path): The root storage directory.
        sample_id (str): The unique identifier for the sample.
        storage_foldername (str): The folder name under which to store the sample data.
    Returns:
        Path: The path to the created target directory.
    """
    samples_path = get_samples_path(storage_root)
    target_dir = samples_path / sample_id / storage_foldername
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir
