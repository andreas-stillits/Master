from __future__ import annotations

import json
from pathlib import Path

from batching import BatchContext, format_sample_id

from mscthesis.utilities.paths import ProjectPaths

MAX_WORKERS: int = 15
COMMAND = "triangulate"

SAMPLE_ID_BASE: int = 0
SAMPLE_ID_TOP: int = 7
SAMPLE_ID_DIGITS: int = 5

MINIMUM_SURFACES_PER_SPHERE: int = 100
MINIMUM_SURFACES: int = 2_000

CLI_FLAGS: tuple[str] = ("--decimation-target",)


def _get_sample_id_list() -> list[str]:
    """Generate a list of sample IDs to be triangulated"""
    sample_id_list: list[str] = []
    for sample_index in range(SAMPLE_ID_BASE, SAMPLE_ID_TOP + 1):
        sample_id: str = format_sample_id(sample_index, SAMPLE_ID_DIGITS)
        sample_id_list.append(sample_id)
    return sample_id_list


def generate_workload() -> list[tuple]:
    """Generate a workload of samples to be synthesized"""
    workload: list[tuple[str, int]] = []

    sample_id_list = _get_sample_id_list()

    storage_root: Path = Path.home() / "coding/master/.treasury"
    paths: ProjectPaths = ProjectPaths(storage_root)

    for sample_id in sample_id_list:
        manifest_path: Path = paths.sample(sample_id).synthesis().require_manifest()
        with open(manifest_path, "r") as f:
            manifest_dict = json.load(f)
            num_cells = manifest_dict["meta"]["num_cells_placed"]
            decimation_target = int(
                max(MINIMUM_SURFACES_PER_SPHERE * num_cells, MINIMUM_SURFACES)
            )
            workload.append((sample_id, decimation_target))

    return workload


def main() -> int:
    ctx = BatchContext(
        max_workers=MAX_WORKERS,
        generator=generate_workload,
        command=COMMAND,
        cli_flags=CLI_FLAGS,
    )
    return ctx.run_batch()


if __name__ == "__main__":
    raise SystemExit(main())
