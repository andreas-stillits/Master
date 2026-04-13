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

    @property
    def processes(self) -> Path:
        return self.base / "processes"

    @property
    def stats(self) -> Path:
        return self.base / "stats"

    @property
    def validation(self) -> Path:
        return self.base / "validation"

    @property
    def diffusion(self) -> Path:
        return self.base / "diffusion"

    def validate(self, tag: str) -> ValidationPaths:
        return ValidationPaths(self, tag)

    def diffuse(self) -> DiffusionPaths:
        return DiffusionPaths(self)

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

    def ensure_processes_root(self) -> Path:
        self.require_base()
        return ensure_dir(self.processes)

    def ensure_stats_root(self) -> Path:
        self.require_base()
        return ensure_dir(self.stats)

    def ensure_validation_root(self) -> Path:
        self.require_base()
        return ensure_dir(self.validation)

    def ensure_diffusion_root(self) -> Path:
        self.require_base()
        return ensure_dir(self.diffusion)


@dataclass(frozen=True)
class ValidationPaths:
    paths: ProjectPaths
    tag: str

    @property
    def dir(self) -> Path:
        return self.paths.validation / self.tag

    @property
    def results(self) -> Path:
        return self.dir / "results.csv"

    @property
    def config(self) -> Path:
        return self.dir / "config.json"

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.json"

    def require_dir(self) -> Path:
        return require_dir(self.dir)

    def ensure_dir(self) -> Path:
        self.paths.require_base()
        return ensure_dir(self.dir)

    def verify_tag(self) -> None:
        # nonempty
        if not self.tag:
            raise ValueError("Validation tag cannot be empty")
        # no special characters that are not allowed in file paths
        if any(c in self.tag for c in r'\/:*?"<>|'):
            raise ValueError(
                'Validation tag cannot contain any of the following characters: \\ / : * ? " < > |'
            )
        # no leading or trailing whitespace
        if self.tag != self.tag.strip():
            raise ValueError(
                "Validation tag cannot have leading or trailing whitespace"
            )
        # no leading or trailing dots
        if self.tag != self.tag.strip("."):
            raise ValueError("Validation tag cannot have leading or trailing dots")
        return

    def ensure_reference_dir(self) -> Path:
        self.ensure_dir()
        return ensure_dir(self.dir / "reference")

    def ensure_meshes_dir(self) -> Path:
        self.ensure_dir()
        return ensure_dir(self.dir / "meshes")

    def ensure_solutions_dir(self) -> Path:
        self.ensure_dir()
        solutions_dir = ensure_dir(self.dir / "solutions")
        ensure_dir(solutions_dir / "CG1")
        ensure_dir(solutions_dir / "CG2")
        return solutions_dir

    def ensure_plots_dir(self) -> Path:
        self.ensure_dir()
        return ensure_dir(self.dir / "plots")

    def ensure_all(self) -> Path:
        self.ensure_dir()
        self.ensure_reference_dir()
        self.ensure_meshes_dir()
        self.ensure_solutions_dir()
        self.ensure_plots_dir()
        return self.dir

    def get_solution_path(self, order: int) -> Path:
        if order == 1:
            return self.ensure_solutions_dir() / "CG1" / "solution.bp"
        elif order == 2:
            return self.ensure_solutions_dir() / "CG2" / "solution.bp"
        else:
            raise ValueError(f"Unsupported order: {order}")


@dataclass(frozen=True)
class DiffusionPaths:
    paths: ProjectPaths

    @property
    def dir(self) -> Path:
        return self.paths.diffusion

    @property
    def meshes(self) -> Path:
        return self.dir / "meshes"

    @property
    def results(self) -> Path:
        return self.dir / "results.csv"

    def require_dir(self) -> Path:
        return require_dir(self.dir)

    def ensure_dir(self) -> Path:
        self.paths.require_base()
        return ensure_dir(self.dir)

    def ensure_meshes_dir(self) -> Path:
        self.ensure_dir()
        return ensure_dir(self.meshes)

    def get_mesh_dir(self, aspect: float) -> Path:
        self.ensure_meshes_dir()
        return ensure_dir(self.meshes / f"aspect_{aspect:.2f}")

    def get_mesh_file(self, aspect: float) -> Path:
        return self.get_mesh_dir(aspect) / "volumetric_mesh.msh"

    def require_mesh_file(self, aspect: float) -> Path:
        mesh_file = self.get_mesh_file(aspect)
        require_file(mesh_file)
        return require_extension(mesh_file, ".msh")

    def require_results_file(self) -> Path:
        require_file(self.results)
        return require_extension(self.results, ".csv")


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

    def solving(self) -> SingleSolutionPaths:
        return SingleSolutionPaths(self)

    def scanning(self) -> ScanningPaths:
        return ScanningPaths(self)

    def diffusion(self) -> DiffusionPaths:
        return DiffusionPaths(self)

    # verification

    def require_dir(self) -> Path:
        return require_dir(self.dir)

    def ensure_dir(self) -> Path:
        self.paths.ensure_samples_root()
        return ensure_dir(self.dir)

    def require_process(self, process_name: str) -> ProcessPathsBase:
        self.require_dir()
        if process_name == "synthesis":
            synthesis = self.synthesis()
            synthesis.require_dir()
            return synthesis

        elif process_name == "triangulation":
            triangulation = self.triangulation()
            triangulation.require_dir()
            return triangulation

        elif process_name == "meshing":
            meshing = self.meshing()
            meshing.require_dir()
            return meshing

        elif process_name == "solving":
            solving = self.solving()
            solving.require_dir()
            return solving

        elif process_name == "scanning":
            scanning = self.scanning()
            scanning.require_dir()
            return scanning

        elif process_name == "diffusion":
            diffusion = self.diffusion()
            diffusion.require_dir()
            return diffusion

        else:
            raise ValueError(f"Unknown process name: {process_name}")


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


@dataclass(frozen=True)
class SingleSolutionPaths(ProcessPathsBase):
    name: str = "solving"

    @property
    def solution(self) -> Path:
        return self.dir / "solution.bp"

    def require_solution(self) -> Path:
        self.require_dir()
        require_file(self.solution)
        return require_extension(self.solution, ".bp")


@dataclass(frozen=True)
class ScanningPaths(ProcessPathsBase):
    name: str = "scanning"

    @property
    def scan(self) -> Path:
        return self.dir / "dataframe.csv"

    @property
    def plots(self) -> Path:
        return ensure_dir(self.dir / "plots")

    def require_scan(self) -> Path:
        self.require_dir()
        require_file(self.scan)
        return require_extension(self.scan, ".csv")
