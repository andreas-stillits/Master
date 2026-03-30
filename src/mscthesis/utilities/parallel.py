from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable


def generate_batches_round_robin(
    workload: list[tuple],
    num_batches: int,
) -> list[list[tuple]]:
    """
    Distribute workload roughly equally into num_batches batches using Round-Robin distribution.
    Args:
        workload (list[(float, float)]): A list of (absorption, transport) pairs.
        num_batches (int): The number of batches to distribute the workload into.
    Returns:
        batches (list[list[tuple]]): A list of batches, each containing a list of (absorption, transport) pairs.
    """
    batches: list[list[tuple]] = [[] for _ in range(num_batches)]

    for i, work in enumerate(workload):
        batches[i % num_batches].append(work)

    return [batch for batch in batches if batch]  # avoid empty batches


def distribute(
    run_batch: Callable[..., list[dict[str, Any]]],
    batches: list[list[tuple]],
    workers: int,
    *args,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Simple function to parallize work on batches of type list[tuple]
    Any arguments required by run_batch may be passed after 'batches' and 'workers'
    OBS: to accomodate parallelization, there cannot be shared instances between run_batch calls
    e.g. if run_batch requires a loaded file, each call must load it independently
    Args:
        run_batch (Callable[[batch, ...], list[dict[str, Any]]]): function acting on a batch
        batches (list[list[tuple]]): collection of individual batches of type list[tuple]
        workers (int): number of multiprocesses to envoke
        *args, **kwargs: additional arguments for run_batch if not passed as partial
    Returns:
        list[dict[str, Any]]: a list of results in the order work is completed - not given
    """

    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_batch, batch, *args, **kwargs) for batch in batches
        ]

        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)

    return results
