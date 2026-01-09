from __future__ import annotations

import argparse

from mpi4py import MPI

from ...config.declaration import ProjectConfig, TriangulationConfig
from ...core.io import load_voxels, save_surface_mesh
from ...core.meshing.triangulation import triangulate_voxels
from ...utilities.paths import ProjectPaths
from ..shared import (
    derive_cli_flags_from_config,
    distribute_command_execution,
    document_command_execution,
)

CMD_NAME = "triangulate"


def _execute_single_sample_id(
    paths: ProjectPaths, config: ProjectConfig, sample_id: str, size: int
) -> None:
    """Execute process for a single sample ID"""
    # get resolved config
    cmdconfig: TriangulationConfig = config.triangulate

    voxels_path = paths.sample(sample_id).synthesis().require_voxels()
    voxels = load_voxels(voxels_path)

    # generate surface mesh
    surface_mesh, metadata = triangulate_voxels(
        voxels,
        cmdconfig.smoothing_iterations,
        cmdconfig.decimation_target,
        cmdconfig.shrinkage_tolerance,
    )

    output_paths = paths.sample(sample_id).triangulation()
    output_paths.ensure_dir()
    surface_mesh_path = output_paths.mesh

    save_surface_mesh(surface_mesh, surface_mesh_path)

    document_command_execution(
        output_paths,
        config,
        CMD_NAME,
        size,
        sample_id,
        inputs={"voxel_model": str(voxels_path.expanduser().resolve())},
        outputs={"surface_mesh": str(surface_mesh_path.expanduser().resolve())},
        metadata=metadata,
    )

    return


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command declaration"""
    return distribute_command_execution(args, comm, _execute_single_sample_id)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="generate a surface mesh from a voxel model using marching cubes",
        help="generate a surface mesh from a voxel model using marching cubes",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_input",
        type=str,
        help="Either a valid sample ID or path to a text file containing sample IDs (one per line)",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
