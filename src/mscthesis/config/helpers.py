from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .declaration import ProjectConfig

# === Helper Functions ===


def filter_config_for_command(model: BaseModel, command: str) -> dict[str, Any]:
    """
    Extract configuration models relevant to a specific command.
    Args:
        model (BaseModel): The root configuration model.
        command (str): The command to filter for.
    Returns:
        dict[str, Any]: A dictionary containing only the relevant configuration for the command.
    Notes:
        - This function assumes that each Pydantic model may have a `model_config` attribute
            with a `json_schema_extra` dictionary that can contain a `commands` list.
    """

    def _recurse(m: BaseModel) -> dict[str, Any] | None:
        model_config = getattr(m.__class__, "model_config", None)
        if model_config is None:
            return None
        extra = model_config.get("json_schema_extra") or {}
        commands = extra.get("commands")

        if commands is not None and command not in commands:
            # this whole model is irrelevant
            return None

        result: dict[str, Any] = {}
        for name in m.__class__.model_fields.keys():
            value = getattr(m, name)
            if isinstance(value, BaseModel):
                sub_result = _recurse(value)
                if sub_result:
                    result[name] = sub_result
            else:
                result[name] = value
        return result

    return _recurse(model) or {}


def load_config_from_file(path: Path | None) -> dict[str, Any]:
    """
    Load configuration from a JSON file.
    Args:
        path (Path | None): The path to the JSON configuration file.
    Returns:
        dict[str, Any]: The loaded configuration as a dictionary.
    """
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively update a nested dictionary with another dictionary.
    Args:
        base    (dict[str, Any]): The original dictionary to be updated.
        updates (dict[str, Any]): The dictionary with updates.
    Returns:
        dict[str, Any]: The updated dictionary.
    """
    result = dict(base)
    # for all keys and values
    for key, value in updates.items():
        # if key points to another dictionary, recurse
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        # else update value
        else:
            result[key] = value
    return result


def build_project_config(
    path: Path, overrides: dict[str, Any] | None = None
) -> ProjectConfig:
    """
    Build the project configuration from defaults and file overrides.
    Can be called through API with a path pointing to a config.json file
    Args:
        path (Path): Path to the user-supplied configuration file.
        overrides (dict[str, Any] | None): Additional overrides to apply.
    Returns:
        ProjectConfig: The constructed project configuration.
    """
    # load default config and translate to dictionary
    defaults = ProjectConfig()
    config_dict = defaults.model_dump()

    # update with user config from home directory if present
    config_dict = deep_update(
        config_dict,
        load_config_from_file(
            defaults.meta.user_config_path
        ),  # empty {} if doesnt exist -> no updates
    )

    # update with user supplied config file via command line if present
    config_dict = deep_update(config_dict, load_config_from_file(path))

    # update with overrides if present
    if overrides:
        config_dict = deep_update(config_dict, overrides)
    return ProjectConfig.model_validate(config_dict)  # raises error if not valid
