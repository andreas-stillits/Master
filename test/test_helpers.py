import copy
from pathlib import Path

from myproject.config.helpers import deep_update


def test_deep_update_basic_merge():
    a = {"x": 1, "nested": {"a": 10, "b": 20}}
    b = {"y": 2, "nested": {"b": 99, "c": 30}}
    result = deep_update(copy.deepcopy(a), b)
    # keys merged at top level
    assert result["x"] == 1
    assert result["y"] == 2
    # nested dict updated, 'b' replaced, 'a' preserved, 'c' added
    assert result["nested"]["a"] == 10
    assert result["nested"]["b"] == 99
    assert result["nested"]["c"] == 30


def test_deep_update_non_dict_overwrite():
    a = {"k": {"sub": 1}}
    b = {"k": "string"}
    result = deep_update(a, b)
    # non-dict replaces dict
    assert result["k"] == "string"


def test_deep_update_preserves_path_instances():
    a = {"path": Path("/tmp")}
    b = {"path": Path("/other")}
    result = deep_update(a, b)
    assert isinstance(result["path"], Path)
    assert result["path"] == Path("/other")
