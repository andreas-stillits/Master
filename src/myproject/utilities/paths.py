from __future__ import annotations

from pathlib import Path
from typing import Collection, Mapping, Sequence

from pydantic import BaseModel


class ProjectPathTree:
    """Class representing a general filesystem tree structure"""

    def __init__(
        self,
        root: Path,
        top_name: str,
        subdirs: Sequence[str] = (),
        allowed: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        self.root = root
        self.top_name = top_name
        self.subdirs = list(subdirs)
        self.allowed = (
            {name: set(values) for name, values in allowed.items()}
            if allowed is not None
            else {}
        )
        self.base = self.root / self.top_name

    def _validate_levels(self, levels: Sequence[str]) -> None:
        """Validate the provided levels against allowed values"""
        if len(levels) > len(self.subdirs):
            raise ValueError(
                f"Too many levels specified. Expected at most {len(self.subdirs)} levels."
            )

        for name, value in zip(
            self.subdirs, levels, strict=False
        ):  # strict: require equal lengths (true) or stop at shortest (false)
            allowed_values = self.allowed.get(name)
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(
                    f"Invalid value '{value}' for level '{name}'"
                    f"Allowed: {sorted(allowed_values)}"
                )

    def path(self, *levels: str) -> Path:
        """Get a path within the tree structure"""
        self._validate_levels(levels)
        return self.base.joinpath(*levels)

    def file(self, *levels: str, filename: str) -> Path:
        """Get a file path within the tree structure"""
        return self.path(*levels) / filename

    def subdir(self, *levels: str, dirname: str) -> Path:
        """Get a subdirectory path at a specific level"""
        return self.path(*levels) / dirname

    def ensure(self, *levels: str, subdirs: Sequence[str] = ()) -> None:
        """Ensure that the specified path and its subdirectories exist"""
        dir_path = self.path(*levels)
        dir_path.mkdir(parents=True, exist_ok=True)
        for subdir in subdirs:
            (dir_path / subdir).mkdir(exist_ok=True)


class ScanPaths(BaseModel):
    storage_root: Path
    scan_name: str
    cases: list[str]
    choices: list[str]

    def make_tree(self) -> ProjectPathTree:
        allowed = {
            "case": self.cases,
            "choice": self.choices,
        }
        return ProjectPathTree(
            root=self.storage_root,
            top_name=self.scan_name,
            subdirs=["case", "choice"],
            allowed=allowed,
        )


# class ProjectPaths(BaseModel):
#     temporal_scanning: ScanPaths = ScanPaths(
#         storage_root=config.behavior.storage_root,
#         scan_name="temporal_scanning",
#         cases=["caseA", "caseB", "caseC", "caseD", "caseE"],
#         choices=["Ca", "K", "gs"],
#     )
#     lateral_scanning: ScanPaths = ScanPaths(
#         storage_root=config.behavior.storage_root,
#         scan_name="lateral_scanning",
#         cases=["case1", "case2", "case3"],
#         choices=["X", "Y", "Z"],
#     )


"""
scan_paths = ScanConfig(
    storage_root=config.behavior.storage_root,
    scan_name="temporal_scanning",
    cases=["caseA", "caseB", "caseC", "caseD", "caseE"],
    choices=["Ca", "K", "gs"],
).make_tree()

scan_paths.ensure("caseA", "Ca", subdirs=["results", "logs"])



"""
