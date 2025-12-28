from pathlib import Path

from mscthesis.config.declaration import ProjectConfig


def test_project_config_contract():
    pc = ProjectConfig()
    # model_dump should exist and return a mapping convertible to dict
    dumped = pc.model_dump()
    assert isinstance(dumped, dict)

    # meta.project_config_path exists and is a Path (used by CLI)
    meta = getattr(pc, "meta", None)
    assert meta is not None
    assert hasattr(meta, "project_config_path")
    assert isinstance(meta.project_config_path, Path)
