from __future__ import annotations

from pathlib import Path

from .log import log_call


@log_call()
def validate_sample_id(sample_id: str, required_digits: int) -> bool:
    """
    Validate that the sample ID matches the required length.
    Args:
        sample_id (str): The sample ID to validate.
        required_digits (int): The required length of the sample ID.
    Raises:
        ValueError: If the sample ID does not match the required length.
    """
    # if not required number of digits
    if not len(sample_id) == required_digits:
        raise ValueError(
            f"Sample ID '{sample_id}' does not match required "
            f"length of {required_digits} characters."
        )
    # if not mappable to int
    try:
        int(sample_id)
    except ValueError as err:
        raise ValueError(
            f"Sample ID '{sample_id}' is not a valid integer string."
        ) from err

    return True


@log_call()
def verify_existence(path: str | Path) -> None:
    """
    Verify that the given file or directory exists.

    Args:
        path (str | Path): The path to verify.
    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    return


@log_call()
def verify_extension(file_path: str | Path, *valid_extensions: str) -> None:
    """
    Verify that the file has one of the valid extensions.
    Args:
        file_path (str | Path): The file path to verify.
        valid_extensions (Sequence[str]): Sequence of valid file extensions (including dot).
    Raises:
        ValueError: If the file does not have a valid extension.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"The file '{file_path}' does not have a valid extension. "
            f"Expected one of: {', '.join(valid_extensions)}"
        )
    return
