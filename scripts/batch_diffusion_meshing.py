from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mscthesis.config.declaration import MeshingConfig, ProjectConfig
from mscthesis.config.helpers import dump_resolved_command_config
from mscthesis.core.meshing.gmeshing import build_cylinder_model
from mscthesis.utilities.parallel import distribute
from mscthesis.utilities.paths import DiffusionPaths, ProjectPaths

MAX_WORKERS: int = 9
ASPECT_MIN: float = 0.10
ASPECT_MAX: float = 0.50
RESOLUTION: int = 9


def _run_meshing_session(
    batch: list[tuple[float, Path]], config: ProjectConfig
) -> list[dict[str, Any]]:
    plug_aspect, output_file = batch[0]
    cmdconfig: MeshingConfig = config.mesh
    metadata = build_cylinder_model(
        output_file,
        plug_aspect,
        cmdconfig.global_resolution_factor,
        cmdconfig.min_stomatal_feature,
        cmdconfig.min_stomatal_dist_factor,
        cmdconfig.max_stomatal_dist_factor,
        cmdconfig.min_boundary_dist_factor,
        cmdconfig.max_boundary_dist_factor,
        cmdconfig.min_points_boundary,
        cmdconfig.max_points_boundary,
        cmdconfig.tolerance,
    )
    mesh_dir = output_file.parent
    config_path = mesh_dir / "config.json"
    dump_resolved_command_config(config, "mesh", config_path)

    return [metadata]


def main() -> int:

    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root)
    paths.require_base()
    diffusion_paths: DiffusionPaths = paths.diffuse()
    workload = np.linspace(ASPECT_MIN, ASPECT_MAX, RESOLUTION).tolist()

    batches = [[(aspect, diffusion_paths.get_mesh_file(aspect)) for aspect in workload]]

    _ = distribute(
        _run_meshing_session,
        batches,
        MAX_WORKERS,
        config,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
