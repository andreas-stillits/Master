from __future__ import annotations

import subprocess
from mpi4py import MPI

import numpy as np

SAMPLE_ID_BASE: int = 0
SAMPLE_ID_DIGITS: int = 5

RESOLUTION: int = 5
NUM_CELL_MIN_FRACTION: float = 0.2
CELL_RADII_MIN: float = 0.02
PLUG_RADII: list[float] = [0.14, 0.25]
SEPARATION: float = 0.005

PLUG_RADIUS_KEY: str = "--plug-aspect"
CELL_RADIUS_KEY: str = "--radius"
NUM_CELLS_KEY: str = "--num-cells"
SEPARATION_KEY: str = "--separation"


def _packing_bound(plug_radius: float, cell_radius: float) -> int:
    """Calculate the packing bound for a given plug and cell radius"""
    bound: float = (np.pi / 6) * (3 * plug_radius**2) / (4 * cell_radius**3)
    return int(np.floor(bound))


def _sample_id(sample_index: int) -> str:
    """Generate a sample ID string from a sample index"""
    return str(sample_index).zfill(SAMPLE_ID_DIGITS)


def _generate_workload() -> list:
    """Generate a workload of samples to be synthesized"""

    workload: list[tuple[str, float, float, int]] = []

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
                sample_id: str = _sample_id(sample_index)
                workload.append((sample_id, plug_radius, cell_radius, num_cells))
                sample_index += 1

    return workload


def _run_instance(
    sample_id: str, plug_radius: float, cell_radius: float, num_cells: int
) -> None:
    """Run a single instance of the synthesis process"""
    print(
        f"Running sample {sample_id} with plug radius {plug_radius}, cell radius {cell_radius}, and num cells {num_cells}",
        flush=True,
    )
    cmd = [
        "msc",
        "--quiet",
        "synthesize-uniform",
        PLUG_RADIUS_KEY,
        str(plug_radius),
        CELL_RADIUS_KEY,
        str(cell_radius),
        NUM_CELLS_KEY,
        str(num_cells),
        SEPARATION_KEY,
        str(SEPARATION),
        sample_id,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    """Main function to generate the workload and run the synthesis process"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    workload: list[tuple[str, float, float, int]] = _generate_workload()
    amount = len(workload)

    work = workload[rank:amount:size]

    for sample_id, plug_radius, cell_radius, num_cells in work:
        _run_instance(sample_id, plug_radius, cell_radius, num_cells)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
