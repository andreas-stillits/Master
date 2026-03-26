from __future__ import annotations

import numpy as np
from batching import BatchContext, format_sample_id

MAX_WORKERS: int = 15
COMMAND = "synthesize-uniform"

SAMPLE_ID_BASE: int = 0
SAMPLE_ID_DIGITS: int = 5

RESOLUTION: int = 2
NUM_CELL_MIN_FRACTION: float = 0.2
CELL_RADII_MIN: float = 0.02
PLUG_RADII: list[float] = [0.14, 0.25]
SEPARATION: float = 0.005

CLI_FLAGS: tuple[str] = (
    "--plug-aspect",
    "--radius",
    "--num-cells",
    "--separation",
)


def _packing_bound(plug_radius: float, cell_radius: float) -> int:
    """Calculate the packing bound for a given plug and cell radius"""
    bound: float = (np.pi / 6) * (3 * plug_radius**2) / (4 * cell_radius**3)
    return int(np.floor(bound))


def generate_workload() -> list[tuple]:
    """Generate a workload of samples to be synthesized"""

    workload: list[tuple[str, float, float, int, float]] = []

    sample_index: int = SAMPLE_ID_BASE
    for plug_radius in PLUG_RADII:
        #
        cell_radii_max: float = np.min([plug_radius - SEPARATION, 0.5 - SEPARATION]) / 2
        for cell_radius in np.linspace(CELL_RADII_MIN, cell_radii_max, RESOLUTION):
            #
            packing_bound: int = _packing_bound(plug_radius, cell_radius)
            num_cells_min: int = int(np.ceil(packing_bound * NUM_CELL_MIN_FRACTION))
            for num_cells in np.linspace(
                num_cells_min, packing_bound, RESOLUTION, dtype=int
            ):
                #
                sample_id: str = format_sample_id(sample_index)
                workload.append(
                    (sample_id, plug_radius, cell_radius, num_cells, SEPARATION)
                )
                sample_index += 1

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
