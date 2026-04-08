from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import load_volumetric_mesh, save_fem_solution
from ..core.meshing.gmeshing import run_gmsh_session
from ..core.solvers import BaseSolver, MeshContext, SolverConfig
from ..utilities.paths import ValidationPaths

# implement validation pipeline


""" 
1. Mesh provided brep for a range of resolutions h (factor in (1.0, ...)) [save meshes] 
2. Compute CG1 and CG2 solutions for all resolutions [save solutions, gradients]
3. save as two dataframes 
4. generate 4 plots:
    a.  Show QoI(C_h) for CG1 and CG2 in the same plot [two panel: conc, flux]
    b.  Show discrepancy in J(C_h) and J(grad C_h) for CG1 and CG2 [two panel: conc, flux]
    c . Show convergence of QoI(C_h) towards QoI(C_h_min)_CG2 for CG1 and CG2 [two panel: conc, flux]
    d.  Show C_h similar for CG1 and CG2 in plane, but grad C_h different [4 panel]
5. save plots and exit
"""


def _insert_path_extension(path: Path, extension: float) -> Path:
    return path.with_name(f"{path.stem}_{extension:.2f}_{path.suffix}")


def copy_reference_files(sample_dir: Path, validation_paths: ValidationPaths) -> None:
    whitelisted_dirs = {"synthesis", "triangulation", "meshing"}
    reference_dir = validation_paths.ensure_reference_dir()
    for item in sample_dir.rglob("*"):
        if item.is_file() and any(
            part in item.relative_to(sample_dir).parts for part in whitelisted_dirs
        ):
            relative_path = item.relative_to(sample_dir)
            destination = reference_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.read_bytes())
    return


def prepare_batches(
    resolution_factor_max: float,
    resolution_factor_num: int,
    validation_paths: ValidationPaths,
) -> list[list[tuple[float, Path]]]:
    resolution_fators = np.logspace(
        0,
        np.log10(resolution_factor_max),
        resolution_factor_num,
    )
    mesh_path = validation_paths.ensure_meshes_dir() / "volumetric_mesh.msh"

    batches = [
        [
            (
                factor,
                _insert_path_extension(mesh_path, factor),
            )
        ]
        for factor in resolution_fators
    ]

    return batches


def meshing(
    batch: list[tuple[float, Path]],
    brep_path: Path,
    *args,
) -> list[dict[str, float]]:
    resolution_factor, mesh_path = batch[0]
    return [
        run_gmsh_session(
            brep_path,
            mesh_path,
            resolution_factor,
            *args,
        )
    ]


def solving(
    batch: list[tuple[float, Path]],
    SolverClass: type[BaseSolver],
    solver_config: SolverConfig,
    validation_paths: ValidationPaths,
    parameters: tuple[float, ...],
) -> list[dict[str, Any]]:
    resolution_factor, mesh_path = batch[0]
    # solve a given RD problem to obtain QoI dictionary
    mesh_ctx: MeshContext = load_volumetric_mesh(mesh_path)

    results: list[dict[str, Any]] = []

    for order in [1, 2]:
        solver_config.order = order

        solver = SolverClass(solver_config, mesh_ctx)
        solution, analysis = solver.solve_for(*parameters)

        solution_path = _insert_path_extension(
            validation_paths.get_solution_path(order), resolution_factor
        )

        save_fem_solution(solution, mesh_ctx, solution_path)

        qoi_metrics = {
            "resolution_factor": resolution_factor,
            "order": order,
            **{f"par{i}": param for i, param in enumerate(parameters)},
            **analysis,
        }

        results.append(qoi_metrics)
        del solver

    return results
