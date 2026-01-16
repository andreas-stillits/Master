"""
Cheat sheet

Initialize:

    paths = Paths(config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_inventories_root()

Read from sample id:

    input_path = paths.sample("00001").synthesis().require_voxels() # -> storage_root/samples/00001/synthesis/voxels.npy
    # verifies existence and extension

Read from relative path with '@' shorthand:
    input_path = resolve_existing_samples_file(paths, input, ".npy")
    # -> storage_root/samples/input (if input starts with '@')
    # -> input (if input is absolute path)
    # verifies existence and extension

Write from sample id:
    synthesis = paths.sample("00001").synthesis()
    synthesis.ensure_dir() # create sample dir and synthesis dir if missing
    voxels_path = synthesis.voxels --> storage_root/samples/00001/synthesis/voxels.npy
    # Write to voxels_path


"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ===== Require helpers (pure checks, no creation) =====


def require_dir(path: Path) -> Path:
    """
    Ensure that the given path exists and is a directory.
    Args:
        path (Path): The path to verify.
    Returns:
        Path: The verified directory path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    return path


def require_file(path: Path) -> Path:
    """
    Ensure that the given path exists and is a file.
    Args:
        path (Path): The path to verify.
    Returns:
        Path: The verified file path.
    """
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    return path


def require_extension(path: Path, *valid_extensions: str) -> Path:
    """
    Ensure that the given file path has one of the specified extensions.
    Args:
        path (Path): The file path to verify.
        valid_extensions (str): Valid file extensions (e.g., '.txt', '.json').
    Returns:
        Path: The verified file path with a valid extension.
    """
    normalized_extensions = [
        ext if ext.startswith(".") else f".{ext}" for ext in valid_extensions
    ]
    if path.suffix not in normalized_extensions:
        raise ValueError(
            f"File {path} does not have a valid extension: {normalized_extensions}"
        )
    return path


# ===== Ensure helper (creation if missing) =====


def ensure_dir(path: Path) -> Path:
    """
    Ensure that the given path exists as a directory, creating it if necessary.
    Args:
        path (Path): The directory path to ensure.
    Raises:
        NotADirectoryError: If the path exists but is not a directory.
    Returns:
        Path: The ensured directory path.
    """
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===== shorthand @ helpers =====


def resolve_samples_shorthand(paths: ProjectPaths, relative_path: str) -> Path:
    """
    Helper to resolve relative paths using '@' shorthand for samples root.
    If relative_path starts with '@', it is resolved relative to samples root.
    Otherwise, it is treated as an absolute path.
    Args:
        paths (ProjectPaths): The ProjectPaths dataclass instance containing root paths.
        relative_path (str): The relative path, potentially prefixed with '@'.
    Returns:
        Path: The resolved absolute path.
    """
    if relative_path.startswith("@"):
        rel = relative_path[1:].lstrip(
            "/\\"
        )  # remove '@' prefix and leading slashes to avoid absolute path interpretation
        # dont allow escaping via ".."
        path = (paths.samples / rel).resolve()
        root = paths.samples.resolve()
        if root not in path.parents and root != path:
            raise ValueError(
                f"Path '{relative_path}' escapes samples root '{paths.samples}'"
            )
        return path

    return Path(relative_path).expanduser().resolve()


def resolve_existing_samples_file(
    paths: ProjectPaths, relative_path: str, *valid_extensions: str
) -> Path:
    """
    Resolve a relative path that may use '@' shorthand to indicate the samples root,
    and ensure it exists as a file with a valid extension.
    Args:
        paths (ProjectPaths): The ProjectPaths dataclass instance containing root paths.
        relative_path (str): The relative path, potentially prefixed with '@'.
        valid_extensions (str): Valid file extensions (e.g., '.txt', '.json').
    Returns:
        Path: The resolved absolute path.
    """
    path = resolve_samples_shorthand(paths, relative_path)
    require_file(path)
    require_extension(path, *valid_extensions)
    return path


def resolve_inventories_shorthand(paths: ProjectPaths, relative_path: str) -> Path:
    """
    Helper to resolve relative paths using '@' shorthand for inventories root.
    If relative_path starts with '@', it is resolved relative to inventories root.
    Otherwise, it is treated as an absolute path.
    Args:
        paths (ProjectPaths): The ProjectPaths dataclass instance containing root paths.
        relative_path (str): The relative path, potentially prefixed with '@'.
    Returns:
        Path: The resolved absolute path.
    """
    if relative_path.startswith("@"):
        rel = relative_path[1:].lstrip(
            "/\\"
        )  # remove '@' prefix, leading slashes to avoid absolute path interpretation
        # dont allow escaping via ".."
        path = (paths.inventories / rel).resolve()
        root = paths.inventories.resolve()
        if root not in path.parents and root != path:
            raise ValueError(
                f"Path '{relative_path}' escapes inventories root '{paths.inventories}'"
            )
        return path

    return Path(relative_path).expanduser().resolve()


def resolve_existing_inventories_file(
    paths: ProjectPaths, relative_path: str, *valid_extensions: str
) -> Path:
    """
    Resolve a relative path that may use '@' shorthand to indicate the inventories root,
    and ensure it exists as a file with a valid extension.
    Args:
        paths (ProjectPaths): The ProjectPaths dataclass instance containing root paths.
        relative_path (str): The relative path, potentially prefixed with '@'.
        valid_extensions (str): Valid file extensions (e.g., '.txt', '.json').
    Returns:
        Path: The resolved absolute path.
    """
    path = resolve_inventories_shorthand(paths, relative_path)
    require_file(path)
    require_extension(path, *valid_extensions)
    return path


# ===== Structured path dataclasses =====


@dataclass(frozen=True)
class ProjectPaths:
    base: Path

    @property
    def samples(self) -> Path:
        return self.base / "samples"

    @property
    def inventories(self) -> Path:
        return self.base / "inventories"

    def sample(self, sample_id: str) -> SamplePaths:
        return SamplePaths(self, sample_id)

    # verification
    def require_base(self) -> Path:
        return require_dir(self.base)

    def ensure_samples_root(self) -> Path:
        self.require_base()
        return ensure_dir(self.samples)

    def ensure_inventories_root(self) -> Path:
        self.require_base()
        return ensure_dir(self.inventories)


@dataclass(frozen=True)
class SamplePaths:
    paths: ProjectPaths
    sample_id: str

    @property
    def dir(self) -> Path:
        return self.paths.samples / self.sample_id

    # typed convenience
    def synthesis(self) -> SynthesisPaths:
        return SynthesisPaths(self)

    def triangulation(self) -> TriangulationPaths:
        return TriangulationPaths(self)

    def meshing(self) -> MeshingPaths:
        return MeshingPaths(self)

    # verification

    def require_dir(self) -> Path:
        return require_dir(self.dir)

    def ensure_dir(self) -> Path:
        self.paths.ensure_samples_root()
        return ensure_dir(self.dir)


@dataclass(frozen=True)
class ProcessPathsBase:
    sample: SamplePaths
    name: str

    @property
    def dir(self) -> Path:
        return self.sample.dir / self.name

    @property
    def config(self) -> Path:
        return self.dir / "config.json"

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.json"

    # verification
    def require_dir(self) -> Path:
        self.sample.require_dir()
        return require_dir(self.dir)

    def require_config(self) -> Path:
        self.require_dir()
        require_file(self.config)
        return require_extension(self.config, ".json")

    def require_manifest(self) -> Path:
        self.require_dir()
        require_file(self.manifest)
        return require_extension(self.manifest, ".json")

    def ensure_dir(self) -> Path:
        self.sample.ensure_dir()
        return ensure_dir(self.dir)


@dataclass(frozen=True)
class SynthesisPaths(ProcessPathsBase):
    name: str = "synthesis"

    @property
    def voxels(self) -> Path:
        return self.dir / "voxels.npy"

    def require_voxels(self) -> Path:
        self.require_dir()
        require_file(self.voxels)
        return require_extension(self.voxels, ".npy")


@dataclass(frozen=True)
class TriangulationPaths(ProcessPathsBase):
    name: str = "triangulation"

    @property
    def mesh(self) -> Path:
        return self.dir / "surface_mesh.stl"

    @property
    def brep(self) -> Path:
        return self.dir / "surface_mesh.brep"

    def require_mesh(self) -> Path:
        self.require_dir()
        require_file(self.mesh)
        return require_extension(self.mesh, ".stl")

    def require_brep(self) -> Path:
        self.require_dir()
        require_file(self.brep)
        return require_extension(self.brep, ".brep")


@dataclass(frozen=True)
class MeshingPaths(ProcessPathsBase):
    name: str = "meshing"

    @property
    def mesh(self) -> Path:
        return self.dir / "volumetric_mesh.msh"

    def require_mesh(self) -> Path:
        self.require_dir()
        require_file(self.mesh)
        return require_extension(self.mesh, ".msh")
