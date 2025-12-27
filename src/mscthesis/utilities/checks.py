from __future__ import annotations

from pathlib import Path


def validate_sample_id(sample_id: str, required_digits: int) -> None:
    """Validate that the sample ID matches the required length."""
    if not len(sample_id) == required_digits:
        raise ValueError(
            f"Sample ID '{sample_id}' does not match required "
            f"length of {required_digits} characters."
        )
    return


def verify_existence(path: str | Path) -> None:
    """Verify that the given file or directory exists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    return
