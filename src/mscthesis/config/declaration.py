from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic.config import ConfigDict

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
    # magic strings and numbers -- underscores are to avoid name collision with internal logging fields
    log_summary_max_length: int = 32
    log_call_start_msg: str = "_entry"
    log_call_end_msg: str = "_exit"
    log_call_error_msg: str = "_error"
    log_call_func_key: str = "_func"
    log_call_details_key: str = "_details"


# tiner helper class for clearer errors when user alters default logging level
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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
        "quiet": "Flag to store as true and suppress console output",
    }


class UniformSynthesisConfig(BaseModel):
    """Configuration for uniform swiss cheese voxel model synthesis"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"expose": True, "commands": ["synthesize-uniform"]},
    )

    base_seed: int = 123456
    resolution: int = 64
    plug_aspect: float = 0.30
    num_cells: int = 100
    min_radius: float = 0.05
    max_radius: float = 0.15
    min_separation: float = 0.01
    max_attempts: int = 1000

    cli_hints: ClassVar[dict[str, str]] = {
        "base_seed": "Base seed for random number generation",
        "resolution": "Number of voxels along each axis",
        "plug_aspect": "Ratio of plug radius to plug thickness/height",
        "num_cells": "Number of cells (spheres) to place in the model",
        "min_radius": "Minimum radius of the cells",
        "max_radius": "Maximum radius of the cells",
        "min_separation": "Minimum separation distance between cells",
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

    boundary_margin_fraction: float = 0.05
    substomatal_cavity_margin_fraction: float = 0.15
    tolerance: float = 0.01
    minimum_resolution: float = 0.02
    maximum_resolution: float = 0.2
    minimum_distance: float = 0.05
    maximum_distance: float = 0.2
    inlet_base_resolution_factor: float = 2.0

    cli_hints: ClassVar[dict[str, str]] = {
        "boundary_margin_fraction": "Margin fraction for boundary refinement",
        "substomatal_cavity_margin_fraction": "Margin fraction for substomatal cavity refinement",
        "tolerance": "Tolerance for geometric operations",
        "minimum_resolution": "Minimum mesh element size",
        "maximum_resolution": "Maximum mesh element size",
        "minimum_distance": "Minimum distance for mesh sizing field",
        "maximum_distance": "Maximum distance for mesh sizing field",
        "inlet_base_resolution_factor": "Factor to scale minimum resolution at inlets",
    }


# Declaration of the umbrella config object
class ProjectConfig(BaseModel):
    """Main project configuration for mscthesis."""

    meta: MetaConfig = MetaConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    synthesize_uniform: UniformSynthesisConfig = UniformSynthesisConfig()
    triangulate: TriangulationConfig = TriangulationConfig()
    mesh: MeshingConfig = MeshingConfig()

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
