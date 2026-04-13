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
STOMATAL_ASPECT_MIN: float = 0.02
STOMATAL_ASPECT_MAX: float = 0.50
RESOLUTION: int = 29


def _generate_workload(paths: ProjectPaths) -> list[tuple[float, Path]]:
    diffusion_paths: DiffusionPaths = paths.diffuse()
    workload = np.linspace(
        STOMATAL_ASPECT_MIN, STOMATAL_ASPECT_MAX, RESOLUTION
    ).tolist()
    return [(aspect, diffusion_paths.get_mesh_file(aspect)) for aspect in workload]


def _run_solution_session(
    batch: list[tuple[float, Path]], config: ProjectConfig
) -> list[dict[str, Any]]:
    return [{}]


def main() -> int:

    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root)
    paths.require_base()
    diffusion_paths: DiffusionPaths = paths.diffuse()
    batches = _generate_workload(paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
