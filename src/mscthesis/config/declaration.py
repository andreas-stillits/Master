from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel
from pydantic.config import ConfigDict

from ..utilities.log import LogLevel

# === CHOICES ===
""" 
The config class is ProjectConfig with subclasses for each domain of configuration:
- meta (naming)
- behavior
- core related commands

Each subclass should declare model_config
- extra defines how to react if the loaded json/dict has other keys than defined by the models
    - forbid: exposing a non-coded key raises error (good for typos)
    - ignore: silently drop unkown keys
    - allow: keep unknown keys around 
- json_schema_extra defines
    - expose (bool): should this model be exposed in json files? (user editable)
    - commands (list[str]): what commands depend on these settings? (useful for saving minimal configs)
        -> must add all commands that uses it (whitelisting)
        -> can be recognized though CLI via args.command or similar
"""


# === Configuration Models ===


class MetaConfig(BaseModel):
    """Meta configuration for naming and hardcoded paths"""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"expose": False, "commands": []}
    )

    project_name: str = "mscthesis"
    project_version: str = "0.1.0"
    config_name: str = "config.json"  # bound in utilities.paths as well
    user_config_path: Path = Path.home() / f".{project_name}" / config_name
    project_config_path: Path = Path.cwd() / config_name


class BehaviorConfig(BaseModel):
    """Configuration for behavior related settings"""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"expose": True, "commands": []}
    )

    storage_root: Path = Path.home() / "coding/master/.treasury"
    sample_id_digits: int = 5
    quiet: bool = False
    no_cmdconfig: bool = False
    no_manifest: bool = False
    no_log: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_filename: str = "run.log"

    # annotate help messages for cli overrides
    # type hinting as 'ClassVar' makes pydantic disregard it upon .model_dump()
    # Can handle abscent and partial declaration with fallback to ""
    cli_hints: ClassVar[dict[str, str]] = {
        "storage_root": "Path to storage root for I/O actions",
        "sample_id_digits": "Number of digits required for valid sample IDs",
        "quiet": "Flag to store as true and suppress console output",
        "no_cmdconfig": "Flag to store as true and skip saving command-specific config file",
        "no_manifest": "Flag to store as true and skip saving manifest file",
        "no_log": "Flag to store as true and skip saving log file",
        "log_level": "Logging level for console and file output, choose from: DEBUG, INFO, WARNING, ERROR, CRITICAL",
        "log_filename": "Filename for log output, stored in storage_root",
    }


class UniformSynthesisConfig(BaseModel):
    """Configuration for uniform swiss cheese voxel model synthesis"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["synthesize-uniform"]},
    )

    base_seed: int = 123456
    resolution: int = 100
    plug_aspect: float = 0.25
    num_cells: int = 100
    radius: float = 0.08
    separation: float = 0.01
    max_attempts: int = 10_000

    cli_hints: ClassVar[dict[str, str]] = {
        "base_seed": "Base seed for random number generation",
        "resolution": "Number of voxels along each axis",
        "plug_aspect": "Ratio of plug radius to plug thickness/height",
        "num_cells": "Number of cells (spheres) to place in the model",
        "radius": "Radius of the cells",
        "separation": "Minimum separation distance between cells and boundaries",
        "max_attempts": "Maximum attempts to place each cell without overlap",
    }


class TriangulationConfig(BaseModel):
    """Configuration for triangulation related settings"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["triangulate"]},
    )

    smoothing_iterations: int = 15
    decimation_target: int = 10_000
    shrinkage_tolerance: float = 0.10
    freecad_cmd: str = "freecadcmd-daily"
    freecad_script_path: str = (
        "/home/andreasstillits/coding/master/src/mscthesis/core/meshing/breping.py"
    )

    cli_hints: ClassVar[dict[str, str]] = {
        "smoothing_iterations": "Number of smoothing iterations to apply to the mesh",
        "decimation_target": "Target number of faces after decimation",
        "shrinkage_tolerance": "Maximum acceptable shrinkage ratio for area and volume",
        "freecad_cmd": "Command to run FreeCAD in command line mode",
        "freecad_script_path": "Path to the FreeCAD script for BREP export (shipped with mscthesis)",
    }


class MeshingConfig(BaseModel):
    """Configuration for meshing related settings"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["mesh"]},
    )

    global_resolution_factor: float = 2.0
    min_stomatal_feature: float = 0.008
    min_cellular_feature: float = 0.008
    min_stomatal_dist_factor: float = 4.0
    max_stomatal_dist_factor: float = 8.0
    min_cellular_dist_factor: float = 4.0
    max_cellular_dist_factor: float = 8.0
    min_boundary_dist_factor: float = 2.0
    max_boundary_dist_factor: float = 4.0
    min_points_boundary: int = 30
    max_points_boundary: int = 40
    boundary_margin_fraction: float = 0.05
    substomatal_cavity_margin_fraction: float = 0.05
    tolerance: float = 0.01

    cli_hints: ClassVar[dict[str, str]] = {
        "global_resolution_factor": "Factor to determine global meshing resolution based on plug dimensions",
        "min_stomatal_feature": "Minimum feature size for stomatal pores",
        "min_cellular_feature": "Minimum feature size for cellular structures",
        "min_stomatal_dist_factor": "Minimum distance factor for stomatal pores",
        "max_stomatal_dist_factor": "Maximum distance factor for stomatal pores",
        "min_cellular_dist_factor": "Minimum distance factor for cellular structures",
        "max_cellular_dist_factor": "Maximum distance factor for cellular structures",
        "min_boundary_dist_factor": "Minimum distance factor for outer boundary",
        "max_boundary_dist_factor": "Maximum distance factor for outer boundary",
        "min_points_boundary": "Minimum number of points to place along the outer boundary -> sets global max resolution",
        "max_points_boundary": "Maximum number of points to place along the outer boundary -> sets boundary min resolution",
        "boundary_margin_fraction": "Margin fraction for minimum distance to outer edges",
        "substomatal_cavity_margin_fraction": "Margin fraction for minimum distance to substomatal cavity (bottom of the plug)",
        "tolerance": "Tolerance for meshing operations",
    }


class ValidationConfig(BaseModel):
    """Configuration for validation related settings"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["validation"]},
    )

    tag: str = "newtest"
    no_meshing: bool = False
    no_solving: bool = False
    no_show: bool = False
    max_workers: int = 16
    resolution_factor_max: float = 4.0
    resolution_factor_num: int = 16
    problem_type: Literal["uniform", "diffusion"] = "uniform"
    parameters_uniform: tuple[float, ...] = (
        0.75,
        1.0,
        0.2,
    )  # chii, absorption, compensation
    parameters_diffusion: tuple[float, ...] = (0.75, 0.2)  # chii, chi_top
    stomatal_aspect: float = 0.04
    stomatal_epsilon: float = 0.02
    kappa: float = 1e8
    ksp_type: str = "cg"
    ksp_rtol: float = 1e-8
    pc_type: str = "jacobi"
    quad_degree: int = 4

    cli_hints: ClassVar[dict[str, str]] = {
        "tag": "Tag to identify the validation run, used for naming output directory",
        "no_meshing": "Flag to store as true and skip meshing step if meshes already exist",
        "no_solving": "Flag to store as true and skip solving step if solutions already exist",
        "no_show": "Flag to store as true and skip displaying plots after validation",
        "max_workers": "Max number of worker processes for parallel execution of meshing and solving",
        "resolution_factor_max": "Maximum factor for determining mesh resolution",
        "resolution_factor_num": "Number of resolution factors to test (logspaced between 1.0 and max)",
        "problem_type": "Type of problem to solve for validation, choose from: uniform, diffusion",
        "parameters_uniform": "Tuple of parameters for uniform problem (chii, absorption, compensation)",
        "parameters_diffusion": "Tuple of parameters for diffusion problem (chii, chi_top)",
        "stomatal_aspect": "Aspect ratio of the stomatal pore for validation problems",
        "stomatal_epsilon": "Smoothing parameter for the stomatal envelope function in validation problems",
        "kappa": "Penalty parameter for the Dirichlet condition in diffusion validation problems",
        "ksp_type": "KSP solver type for validation problems",
        "ksp_rtol": "Relative tolerance for the KSP solver in validation problems",
        "pc_type": "Preconditioner type for validation problems",
        "quad_degree": "Quadrature degree for finite element integration in validation problems",
    }


class UniformSolutionConfig(BaseModel):
    """Configuration for solver related settings"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["solve-uniform"]},
    )

    chii: float = 0.75
    absorption: float = 1.0
    compensation: float = 0.1
    stomatal_aspect: float = 0.04
    stomatal_epsilon: float = 0.02
    kappa: float = 1e8
    ksp_type: str = "cg"
    ksp_rtol: float = 1e-8
    pc_type: str = "jacobi"
    quad_degree: int = 4
    order: int = 2
    no_save: bool = False

    cli_hints: ClassVar[dict[str, str]] = {
        "chii": "Chii parameter for the uniform problem",
        "absorption": "Absorption balance",
        "compensation": "Boundary condition value for the mesophyll flux",
        "stomatal_aspect": "Aspect ratio of the stomatal pore",
        "stomatal_epsilon": "Smoothing parameter for the stomatal envelope function",
        "kappa": "Penalty parameter for the Dirichlet condition in diffusion validation problems",
        "ksp_type": "KSP solver type for problem solution",
        "ksp_rtol": "Relative tolerance for the KSP solver",
        "pc_type": "Preconditioner type for problem solution",
        "quad_degree": "Quadrature degree for finite element integration",
        "order": "Order of the finite element method",
        "no_save": "Whether to save the solution to a file (put flag for true to skip saving)",
    }


class ScanningConfig(BaseModel):
    """Configuration for scanning related settings"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["scan"]},
    )

    absorption_min: float = 0.01
    absorption_max: float = 100.0
    absorption_num: int = 10
    chii_min: float = 0.11
    chii_max: float = 0.99
    chii_num: int = 10
    compensation: float = 0.1
    stomatal_aspect: float = 0.04
    stomatal_epsilon: float = 0.02
    kappa: float = 1e8
    ksp_type: str = "cg"
    ksp_rtol: float = 1e-8
    pc_type: str = "jacobi"
    quad_degree: int = 4
    order: int = 2
    max_workers: int = 16

    cli_hints: ClassVar[dict[str, str]] = {
        "absorption_min": "Minimum absorption value for scanning",
        "absorption_max": "Maximum absorption value for scanning",
        "absorption_num": "Number of absorption values to scan",
        "chii_min": "Minimum substomatal conc value for scanning",
        "chii_max": "Maximum substomatal conc value for scanning",
        "chii_num": "Number of substomatal conc values to scan",
        "compensation": "Boundary condition value for the mesophyll flux",
        "stomatal_aspect": "Aspect ratio of the stomatal pore",
        "stomatal_epsilon": "Smoothing parameter for the stomatal envelope function",
        "kappa": "Penalty parameter for the Dirichlet condition in diffusion validation problems",
        "ksp_type": "KSP solver type for problem solution",
        "ksp_rtol": "Relative tolerance for the KSP solver",
        "pc_type": "Preconditioner type for problem solution",
        "quad_degree": "Quadrature degree for finite element integration",
        "order": "Order of the finite element method",
        "max_workers": "Maximum number of worker processes for parallel execution",
    }


class DiffusionConfig(BaseModel):
    """Configuration for diffusion related settings"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["diffuse"]},
    )

    diffusion_coefficient: float = 1.0

    cli_hints: ClassVar[dict[str, str]] = {
        "diffusion_coefficient": "Diffusion coefficient for the diffusion simulation",
    }


# Declaration of the umbrella config object
class ProjectConfig(BaseModel):
    """Main project configuration for mscthesis."""

    meta: MetaConfig = MetaConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    synthesize_uniform: UniformSynthesisConfig = UniformSynthesisConfig()
    triangulate: TriangulationConfig = TriangulationConfig()
    mesh: MeshingConfig = MeshingConfig()
    validation: ValidationConfig = ValidationConfig()
    solve_uniform: UniformSolutionConfig = UniformSolutionConfig()
    scan: ScanningConfig = ScanningConfig()

    # helper function for filtering after model_config.json_schema_extra.expose
    def _filter_config_for_exposure(self) -> dict[str, Any]:
        """Extract the configuration models that are marked for exposure"""

        # recursive helper function for nested BaseModel objects
        def _recurse(m: BaseModel) -> dict[str, Any] | None:
            """Resolve the BaseModel as a dictionary if marked for exposure"""
            # get exposure status
            model_config = getattr(m.__class__, "model_config", None)
            if model_config is None:
                return None
            extra = model_config.get("json_schema_extra") or {}
            exposed = extra.get("expose", False)  # default to False / None

            # if not the top level ProjectConfig, or not marked for exposure, skip
            if m.__class__ != self.__class__ and not exposed:
                return None

            # resolve recursively as dictionary
            result: dict[str, Any] = {}
            for name in m.__class__.model_fields.keys():
                value = getattr(m, name)
                # call back if attribute is itself a BaseModel
                if isinstance(value, BaseModel):
                    sub_result = _recurse(value)
                    if sub_result:
                        result[name] = sub_result
                else:
                    result[name] = value
            return result

        return _recurse(self) or {}  # default to empty dictionary

    # helper function for printing exposed config in JSON format
    def dump_json(self) -> str:
        return json.dumps(self._filter_config_for_exposure(), indent=2, default=str)
