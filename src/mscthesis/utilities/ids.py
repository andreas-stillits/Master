from __future__ import annotations

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
            f"length of {required_digits} digits."
        )
    # if not mappable to int
    try:
        int(sample_id)
    except Exception as exc:
        raise ValueError(
            f"Sample ID '{sample_id}' is not a valid integer string."
        ) from exc

    return True
