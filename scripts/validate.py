from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mscthesis.config.declaration import ProjectConfig
from mscthesis.config.helpers import build_project_config
from mscthesis.core.io import load_volumetric_mesh, save_dataframe, save_fem_solution
from mscthesis.core.meshing.gmeshing import run_gmsh_session
from mscthesis.core.plotting import plot_validation_results
from mscthesis.core.solvers import (
    BaseSolver,
    MeshContext,
    UniformSolver,
    UniformSolverConfig,
)
from mscthesis.utilities.parallel import distribute
from mscthesis.utilities.paths import ProjectPaths, require_file

# inputs
config_file = "/home/andreasstillits/coding/master/config.json"
sample_id = "00019"  # will be a brep_path instead
tag = "00019_3200"


@dataclass
class CmdConfig:
    no_meshing: bool
    absorption: float
    transport: float
    compensation: float
    workers: int
    stomatal_aspect: float
    stomatal_epsilon: float
    ksp_rtol: float
    order: int


cmdconfig = CmdConfig(
    no_meshing=True,
    absorption=5.0,
    transport=5.0,
    compensation=0.1,
    workers=16,
    stomatal_aspect=0.02,
    stomatal_epsilon=0.002,
    ksp_rtol=1e-8,
    quad_degree=4,
    order=2,
)

config_file = Path(config_file).resolve().expanduser()
config_file = require_file(config_file)

config: ProjectConfig = build_project_config(config_file)
paths = ProjectPaths(config.behavior.storage_root)


brep_path = paths.sample(sample_id).triangulation().require_brep()
brep_path = require_file(brep_path.resolve().expanduser())


# initialize directories
validation_paths = paths.validate(tag)
validation_root = validation_paths.ensure_dir()
meshes_dir = validation_paths.ensure_mesh_dir()
solutions_dir = validation_paths.ensure_solutions_dir()
plots_dir = validation_paths.ensure_plots_dir()


# save a copy to validation directory for reference
shutil.copy(brep_path, validation_root / "surface_mesh.brep")

# create meshes of varying resolution - parallelize if requested
resolution_factors: list[float] = [1.0, 1.50, 2.0, 4.0, 6.0, 8.0]


batches = [
    [(factor, meshes_dir / f"volumetric_mesh_{factor:.2f}_.msh")]
    for factor in resolution_factors
]

print("FINISHED SETUP")


def meshing(batch: list[tuple[float, Path]]) -> dict[str, Any]:
    resolution_factor, mesh_path = batch[0]
    metadata = run_gmsh_session(
        brep_path,
        mesh_path,
        resolution_factor,
        config.mesh.min_stomatal_feature,
        config.mesh.min_cellular_feature,
        config.mesh.min_stomatal_dist_factor,
        config.mesh.max_stomatal_dist_factor,
        config.mesh.min_cellular_dist_factor,
        config.mesh.max_cellular_dist_factor,
        config.mesh.min_boundary_dist_factor,
        config.mesh.max_boundary_dist_factor,
        config.mesh.min_points_boundary,
        config.mesh.max_points_boundary,
        config.mesh.boundary_margin_fraction,
        config.mesh.substomatal_cavity_margin_fraction,
        config.mesh.tolerance,
    )
    return metadata


if cmdconfig.no_meshing:
    print("SKIPPING MESHING")
else:
    metadatas = distribute(meshing, batches, cmdconfig.workers)
    print("FINISHED MESHING")


# for each mesh, solve a problem and save QoI metrics to a .json file
def solving(
    batch: list[tuple[float, Path]],
    SolverClass: type[BaseSolver],
    solver_config: UniformSolverConfig,
    absorption: float,
    transport: float,
    compensation: float,
) -> list[dict[str, Any]]:
    resolution_factor, mesh_path = batch[0]
    # solve a given RD problem to obtain QoI dictionary
    mesh_ctx: MeshContext = load_volumetric_mesh(mesh_path)

    solver = SolverClass(solver_config, mesh_ctx)
    solution, analysis = solver.solve_for(absorption, transport, compensation)

    solution_path = solutions_dir / f"solution_{resolution_factor:.2f}_.bp"
    save_fem_solution(solution, mesh_ctx, solution_path)

    qoi_metrics = {
        "resolution_factor": resolution_factor,
        "absorption": absorption,
        "transport": transport,
        "compensation": compensation,
        **analysis,
    }

    return [qoi_metrics]


solver_config = UniformSolverConfig(
    cmdconfig.stomatal_aspect,
    cmdconfig.stomatal_epsilon,
    cmdconfig.ksp_rtol,
    cmdconfig.quad_degree,
    cmdconfig.order,
)

qoi_metrics = distribute(
    solving,
    batches,
    cmdconfig.workers,
    UniformSolver,
    solver_config,
    cmdconfig.absorption,
    cmdconfig.transport,
    cmdconfig.compensation,
)

print("FINISHED SOLVING")

dataframe = pd.DataFrame(qoi_metrics)
dataframe.sort_values(["resolution_factor"]).reset_index(drop=True, inplace=True)
save_dataframe(dataframe, validation_paths.results)

print("FINISHED SAVING RESULTS")

# generate plots of convergence for QoI metrics as a function of resolution, save in /plots/
plot_validation_results(dataframe, plots_dir, show=True)

print("COMPLETED VALIDATION")
