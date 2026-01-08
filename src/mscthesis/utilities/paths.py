from __future__ import annotations

from pathlib import Path

from .checks import verify_existence
from .log import log_call

SAMPLES_FOLDERNAME: str = "samples"
INVENTORIES_FOLDERNAME: str = "inventories"


# ===== General utilities =====
def get_path(storage_root: Path, foldername: str) -> Path:
    """
    Get the path to a specified folder within the storage root.
    Args:
        storage_root (Path): The root storage directory.
        foldername (str): The folder name to retrieve the path for.
    Returns:
        Path: The path to the specified folder within the storage root.
    """
    target_path = storage_root / foldername
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def resolve_storage_shorthand(
    storage_root: Path, foldername: str, relative_path: str
) -> Path:
    """
    Resolve a relative path that may use '@' shorthand to indicate the storage root.
    Args:
        storage_root (Path): The root storage directory.
        relative_path (str): The relative path, potentially prefixed with '@'.
    Returns:
        Path: The resolved absolute path.
    """
    if relative_path.startswith("@"):
        relative_path = relative_path[1:]  # remove '@' prefix
        full_path = get_path(storage_root, foldername) / relative_path
    else:
        full_path = Path(relative_path)

    verify_existence(full_path)

    return full_path


# =============================


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
    return resolve_storage_shorthand(
        storage_root,
        SAMPLES_FOLDERNAME,
        relative_path,
    )


@log_call()
def expand_inventory_path(storage_root: Path, relative_path: str) -> Path:
    """
    Derive full path within inventory directory given a relative path if signified by @.
    Else, return as absolute Path.
    Args:
        storage_root (Path): The root storage directory.
        relative_path (str): The relative path, potentially prefixed with '@' to indicate inventory directory.
    Returns:
        full_path (Path): The derived full path.
    """
    return resolve_storage_shorthand(
        storage_root,
        INVENTORIES_FOLDERNAME,
        relative_path,
    )


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
    samples_path = get_path(storage_root, SAMPLES_FOLDERNAME)
    target_dir = samples_path / sample_id / storage_foldername
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


@log_call()
def determine_target_directory(
    storage_root: Path,
    sample_id: str,
    storage_foldername: str,
    override_target_dir: str | None = None,
) -> Path:
    """
    Determine the target directory for saving output files.
    Either under sample_id in storage root, or an override provided via CLI argument.
    Args:
        storage_root (Path): The root storage directory.
        sample_id (str): The sample ID for which the target directory is being determined.
        storage_foldername (str): The folder name under the storage root for this command.
        target_dir (Optional[str]): An optional target directory override provided via CLI argument.
    Returns:
        Path: The determined target directory path.
    """

    # determine target directory
    target_directory = (
        create_target_directory(
            storage_root,
            sample_id,
            storage_foldername,
        )
        if override_target_dir is None
        else override_target_dir
    )

    verify_existence(target_directory)

    return Path(target_directory)


@log_call()
def get_voxel_file_path(storage_root: Path, sample_id: str) -> Path:
    """
    Get the file path for the voxel data of a given sample ID.
    Args:
        storage_root (Path): The root storage directory.
        sample_id (str): The unique identifier for the sample.
    Returns:
        Path: The path to the voxel data file for the specified sample ID.
    """
    voxel_directory = determine_target_directory(
        storage_root,
        sample_id,
        "synthesis",
    )
    voxel_file_path = voxel_directory / "voxels.npy"
    verify_existence(voxel_file_path)

    return voxel_file_path
