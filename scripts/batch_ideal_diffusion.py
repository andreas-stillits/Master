from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mscthesis.config.declaration import ProjectConfig
from mscthesis.core.io import load_dataframe, load_volumetric_mesh, save_dataframe
from mscthesis.core.meshing.gmeshing import build_cylinder_model
from mscthesis.core.plotting.ideal_diffusion import plot_diffusion_results
from mscthesis.core.solvers import DiffusionSolver, DiffusionSolverConfig
from mscthesis.utilities.parallel import distribute
from mscthesis.utilities.paths import DiffusionPaths, ProjectPaths, require_file

MAX_WORKERS: int = 16
PLUG_ASPECT_MIN: float = 0.10
PLUG_ASPECT_MAX: float = 0.50
PLUG_RESOLUTION: int = 9
STOMATAL_ASPECT_MIN: float = 0.02
STOMATAL_ASPECT_MAX: float = 0.50
STOMATAL_RESOLUTION: int = 25
STOMATAL_EPSILON: float = 0.002

KSP_TYPE: str = "cg"
KSP_RTOL: float = 1e-8
PC_TYPE: str = "jacobi"
QUAD_DEGREE: int = 4
ORDER: int = 2

GLOBAL_RESOLUTION_FACTOR: float = 2.0
MIN_STOMATAL_FEATURE: float = 0.008
MIN_STOMATAL_DIST_FACTOR: float = 4.0
MAX_STOMATAL_DIST_FACTOR: float = 8.0
MIN_BOUNDARY_DIST_FACTOR: float = 2.0
MAX_BOUNDARY_DIST_FACTOR: float = 4.0
MIN_POINTS_BOUNDARY: int = 60
MAX_POINTS_BOUNDARY: int = 40
TOLERANCE: float = 1e-2

TRANSPORT = 1.0
TOP_CONCENTRATION = 0.1


def _generate_meshing_batches(
    diffusion_paths: DiffusionPaths, workload: np.ndarray
) -> list[list[tuple[float, Path]]]:
    return [
        [(aspect, diffusion_paths.get_mesh_file(aspect).resolve().expanduser())]
        for aspect in workload
    ]


def _generate_solution_batches(
    diffusion_paths: DiffusionPaths,
    plug_aspects: np.ndarray,
    stomatal_aspects: np.ndarray,
) -> list[list[tuple[float, float, Path]]]:
    batches = []
    for plug_aspect in plug_aspects:
        batch = []
        mesh_file = diffusion_paths.get_mesh_file(plug_aspect).resolve().expanduser()
        for stomatal_aspect in stomatal_aspects:
            if stomatal_aspect <= plug_aspect:
                batch.append((plug_aspect, stomatal_aspect, mesh_file))
        if batch:
            batches.append(batch)
    return batches


def _run_meshing_session(
    batch: list[tuple[float, Path]],
) -> list[dict[str, Any]]:
    plug_aspect, output_file = batch[0]
    metadata = build_cylinder_model(
        output_file,
        plug_aspect,
        GLOBAL_RESOLUTION_FACTOR,
        MIN_STOMATAL_FEATURE,
        MIN_STOMATAL_DIST_FACTOR,
        MAX_STOMATAL_DIST_FACTOR,
        MIN_BOUNDARY_DIST_FACTOR,
        MAX_BOUNDARY_DIST_FACTOR,
        MIN_POINTS_BOUNDARY,
        MAX_POINTS_BOUNDARY,
        TOLERANCE,
    )
    return [metadata]


def _run_solution_session(
    batch: list[tuple[float, float, Path]],
) -> list[dict[str, Any]]:
    results = []

    for item in batch:
        plug_aspect, stomatal_aspect, mesh_file = item
        mesh_ctx = load_volumetric_mesh(mesh_file)
        solver_config = DiffusionSolverConfig(
            stomatal_aspect,
            STOMATAL_EPSILON,
            KSP_TYPE,
            KSP_RTOL,
            PC_TYPE,
            QUAD_DEGREE,
            ORDER,
        )
        solver = DiffusionSolver(solver_config, mesh_ctx)
        solution, analysis = solver.solve_for(TOP_CONCENTRATION, TRANSPORT)
        results.append(
            {
                "plug_aspect": plug_aspect,
                "stomatal_aspect": stomatal_aspect,
                "top_concentration": TOP_CONCENTRATION,
                "transport": TRANSPORT,
                **analysis,
            }
        )

    return results


def main() -> int:
    parser = ArgumentParser(
        description="Batch meshing and solving for diffusion exploration"
    )
    parser.add_argument(
        "--no-meshing",
        action="store_true",
        help="Skip meshing phase and assume meshes already exist",
    )
    parser.add_argument(
        "--no-solving",
        action="store_true",
        help="Skip solving phase and assume solutions already exist",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root)
    paths.require_base()
    diffusion_paths: DiffusionPaths = paths.diffuse()
    meshing_range = np.linspace(PLUG_ASPECT_MIN, PLUG_ASPECT_MAX, PLUG_RESOLUTION)
    stomatal_range = np.linspace(
        STOMATAL_ASPECT_MIN, STOMATAL_ASPECT_MAX, STOMATAL_RESOLUTION
    )

    if not args.no_meshing:

        batches = _generate_meshing_batches(diffusion_paths, meshing_range)

        _ = distribute(
            _run_meshing_session,
            batches,
            MAX_WORKERS,
        )

    if not args.no_solving:
        batches = _generate_solution_batches(
            diffusion_paths, meshing_range, stomatal_range
        )

        results = distribute(
            _run_solution_session,
            batches,
            MAX_WORKERS,
        )

        dataframe = pd.DataFrame(results)
        dataframe.sort_values(["plug_aspect", "stomatal_aspect"], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        save_dataframe(dataframe, diffusion_paths.results)

    require_file(diffusion_paths.results)
    dataframe = load_dataframe(diffusion_paths.results)
    plot_diffusion_results(dataframe, diffusion_paths.plots)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
