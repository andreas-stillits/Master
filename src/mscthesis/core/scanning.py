from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import load_volumetric_mesh
from ..core.solvers import BaseSolver, MeshContext, SolverConfig
from ..utilities.log import log_call


@log_call()
def generate_workload(
    transport_range: tuple[float, float, int],
    absorption_range: tuple[float, float, int],
    compensation: float,
    assumed_geometry_factor: float = 2.0,
) -> list[tuple[float, float]]:
    """
    Generate a workload of (absorption, transport) pairs based on the specified ranges.
    Use logarithmic spacing for both absorption and transport values.
    Args:
        transport_range (float, float, int): A tuple (min, max, num)
        absorption_range (float, float, int): A tuple (min, max, num)
    Returns:
        workload (list[(float, float)]): A list of (chii, absorption) pairs of size N_c x N_a.
    """

    # tiny helper function
    def _get_logspace(start: float, stop: float, num: int) -> np.ndarray:
        return np.logspace(np.log10(start), np.log10(stop), num)

    def _get_chii(transport: float, absorption: float) -> float:
        return 1 - (1 - compensation) / (
            1 + transport / assumed_geometry_factor + transport / absorption
        )

    # get logspaced values
    transport_values = _get_logspace(*transport_range)
    absorption_values = _get_logspace(*absorption_range)

    # create all combinations of the two
    workload: list[tuple[float, float]] = []

    for absorption in absorption_values:
        for transport in transport_values:
            chii = _get_chii(transport, absorption)
            workload.append((chii, absorption, compensation))

    return workload


@log_call()
def run_batch(
    batch: list[tuple[float, float, float]],
    mesh_file: str | Path,
    solver_config: SolverConfig,
    solver_class: type[BaseSolver],
) -> list[dict[str, Any]]:
    """
    Worker function to run a batch of simulations.
    Args:
        mesh_file (str | Path): Path to the volumetric mesh file.
        batch (list[(float, float, float)]): A list of (absorption, transport, compensation) triples.
        solver_config (SolverConfig): Configuration for the solver.
        solver_class (type[BaseSolver]): The solver class to instantiate.
    Returns:
        results (list[dict[str, Any]]): A list of dictionaries containing the simulation results.
    OBS:
        To accomodate parallelization, each worker needs to load the mesh independently. Hence we pass the mesh path and not the loaded object.
    """
    results: list[dict[str, Any]] = []

    mesh_ctx: MeshContext = load_volumetric_mesh(mesh_file)

    solver = solver_class(
        solver_config,
        mesh_ctx,
    )

    for chii, absorption, compensation in batch:
        _, analysis = solver.solve_for(chii, absorption, compensation)
        result = {
            "chii": chii,
            "absorption": absorption,
            "compensation": compensation,
            **analysis,
        }
        results.append(result)

    return results
