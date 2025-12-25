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

    model_config = ConfigDict(extra="forbid", json_schema_extra={"expose": False})

    project_name: str = "mscthesis"
    user_config_path: Path = Path.home() / f".{project_name}" / "config.json"
    project_config_path: Path = Path.cwd() / "config.json"
    # magic strings and numbers -- underscores are to avoid name collision with internal logging fields
    log_summary_max_length: int = 16
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
    quiet: bool = False
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


# some example config relating to a specific command
# class SomeCommandConfig(BaseModel):
#     """Configuration for process testing"""

#     model_config = ConfigDict(
#         extra="forbid", json_schema_extra={"expose": True, "commands": ["test"]}
#     )

#     enable_process_test: bool = False
#     test_data_path: Path = Path.home() / f".{MetaConfig().project_name}" / "test_data"
#     some_list: list[float] = [0.1, 0.2, 0.3]

#     cli_hints: ClassVar[dict[str, str]] = {
#         "enable_process_test": "Flag to store as true",
#         "test_data_path": "provide path",
#         "some_list": "a list of floats",
#     }


# Declaration of the umbrella config object
class ProjectConfig(BaseModel):
    """Main project configuration for mscthesis."""

    meta: MetaConfig = MetaConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    # some_command: SomeCommandConfig = SomeCommandConfig()

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
