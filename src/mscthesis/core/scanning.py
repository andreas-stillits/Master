from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import load_volumetric_mesh
from ..core.solvers import BaseSolver, MeshContext, SolverConfig
from ..utilities.log import log_call


@log_call()
def generate_workload(
    absorption_range: tuple[float, float, int],
    transport_range: tuple[float, float, int],
) -> list[tuple[float, float]]:
    """
    Generate a workload of (absorption, transport) pairs based on the specified ranges.
    Use logarithmic spacing for both absorption and transport values.
    Args:
        absorption_range (float, float, int): A tuple (min, max, num)
        transport_range (float, float, int): A tuple (min, max, num)
    Returns:
        workload (list[(float, float)]): A list of (absorption, transport) pairs of size N_a x N_t.
    """

    # tiny helper function
    def _get_logspace(start: float, stop: float, num: int) -> np.ndarray:
        return np.logspace(np.log10(start), np.log10(stop), num)

    # get logspaced values
    absorption_values = _get_logspace(*absorption_range)
    transport_values = _get_logspace(*transport_range)

    # create all combinations of the two
    workload: list[tuple[float, float]] = []

    for absorption in absorption_values:
        for transport in transport_values:
            workload.append((absorption, transport))

    return workload


@log_call()
def generate_batches_round_robin(
    workload: list[tuple[float, float]], num_batches: int
) -> list[list[tuple[float, float]]]:
    """
    Distribute workload roughly equally into num_batches batches using Round-Robin distribution.
    Args:
        workload (list[(float, float)]): A list of (absorption, transport) pairs.
        num_batches (int): The number of batches to distribute the workload into.
    Returns:
        batches (list[list[tuple]]): A list of batches, each containing a list of (absorption, transport) pairs.
    """
    batches: list[list[tuple[float, float]]] = [[] for _ in range(num_batches)]

    for i, work in enumerate(workload):
        batches[i % num_batches].append(work)

    return [batch for batch in batches if batch]  # avoid empty batches


@log_call()
def run_batch(
    mesh_file: str | Path,
    batch: list[tuple[float, float]],
    solver_config: SolverConfig,
    solver_class: type[BaseSolver],
) -> list[dict[str, Any]]:
    """
    Worker function to run a batch of simulations.
    Args:
        mesh_file (str | Path): Path to the volumetric mesh file.
        batch (list[(float, float)]): A list of (absorption, transport) pairs.
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

    for absorption, transport in batch:
        _, analysis = solver.solve_for(absorption, transport)
        result = {
            "absorption": absorption,
            "transport": transport,
            **analysis,
        }
        results.append(result)

    return results


@log_call()
def distribute_batches(
    batches: list[list[tuple[float, float]]],
    mesh_file: str | Path,
    solver_config: SolverConfig,
    solver_class: type[BaseSolver],
    max_workers: int,
) -> list[dict[str, Any]]:
    """
    Distribute the workload across multiple processes.
    Args:
        batches (list[list[tuple]]): A list of batches, each containing a list of (absorption, transport) pairs.
        mesh_file (str | Path): Path to the volumetric mesh file.
        solver_config (SolverConfig): Configuration for the solver.
        solver_class (type[BaseSolver]): The solver class to instantiate.
        max_workers (int): The maximum number of worker processes to use.
    Returns:
        results (list[dict[str, Any]]): A list of dictionaries containing the simulation results from all batches.
    """

    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_batch, mesh_file, batch, solver_config, solver_class)
            for batch in batches
        ]

        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)

    return results
