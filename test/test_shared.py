import argparse
from pathlib import Path

import pytest

from myproject.cli import shared


class FakeCmdConfig:
    def __init__(self):
        # optional hints used by derive_cli_flags_from_config
        self.cli_hints = {
            "do_something": "toggle doing something",
            "count": "an integer count",
            "path": "a path",
        }

    def model_dump(self):
        # defaults used when deriving flags
        return {"do_something": False, "count": 3, "path": Path("/home")}


class FakeMeta:
    project_config_path = Path("/fake/path/config.json")


class FakeProjectConfig:
    def __init__(self):
        self.behavior = FakeCmdConfig()
        self.meta = FakeMeta()

    def model_dump(self):
        # structure expected by assemble_cli_overrides
        return {"behavior": {"do_something": False, "count": 3, "path": Path("/home")}}


def test_parse_string_value_literals_and_fallback():
    # lists and dicts are parsed by ast.literal_eval
    assert shared.parse_string_value("[1, 2, 3]") == [1, 2, 3]
    assert shared.parse_string_value("{'a': 1}") == {"a": 1}
    # non-literal falls back to raw string
    assert shared.parse_string_value("not_a_literal") == "not_a_literal"


def test_derive_flags_and_assemble_overrides(monkeypatch):
    # replace ProjectConfig used inside the module with our fake
    monkeypatch.setattr(shared, "ProjectConfig", FakeProjectConfig)

    parser = argparse.ArgumentParser()
    # derive flags from the fake 'behavior' config
    parser = shared.derive_cli_flags_from_config(parser, "behavior")

    # supply flag to toggle boolean, change int and path
    args = parser.parse_args(["--do-something", "--count", "5", "--path", "/tmp"])
    args.command = "run"  # simulate command context

    # flags parsed with expected types/values
    assert args.do_something is True
    assert args.count == 5
    # Path arguments are parsed to pathlib.Path by the code
    assert isinstance(args.path, Path)
    assert args.path == Path("/tmp")

    # assemble_cli_overrides should include only values that differ from defaults
    defaults = FakeProjectConfig()
    overrides = shared.assemble_cli_overrides(args, defaults)  # type: ignore

    assert "behavior" in overrides
    sub = overrides["behavior"]
    assert sub["do_something"] is True
    assert sub["count"] == 5
    assert sub["path"] == Path("/tmp")


def test_no_overrides_when_defaults(monkeypatch):
    # if no CLI args are provided, assembled overrides should be empty
    monkeypatch.setattr(shared, "ProjectConfig", FakeProjectConfig)

    parser = argparse.ArgumentParser()
    parser = shared.derive_cli_flags_from_config(parser, "behavior")

    args = parser.parse_args([])  # nothing provided -> all defaults
    args.command = "run"  # simulate command context
    defaults = FakeProjectConfig()
    overrides = shared.assemble_cli_overrides(args, defaults)  # type: ignore

    assert overrides == {}


def test_initialize_parsers_adds_config_flag_and_requires_subcommand(monkeypatch):
    # ensure the module uses the fake ProjectConfig for the default path
    monkeypatch.setattr(shared, "ProjectConfig", FakeProjectConfig)

    parser = argparse.ArgumentParser()
    subparsers = shared.initialize_parsers(parser)

    # subparsers object returned and marked required
    assert getattr(subparsers, "required", False) is True

    # parser should have a --config option with default from FakeProjectConfig.meta
    optmap = parser._option_string_actions
    assert "--config" in optmap
    config_action = optmap["--config"]
    assert config_action.default == FakeMeta.project_config_path

    # trying to parse with no subcommand should fail (required subparsers)
    with pytest.raises(SystemExit):
        parser.parse_args([])  # no command supplied
