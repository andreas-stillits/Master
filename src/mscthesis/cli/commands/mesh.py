from __future__ import annotations

import argparse
import os
import subprocess

from mpi4py import MPI

from ...config.declaration import ProjectConfig, MeshingConfig
from ...core.meshing.gmeshing import run_gmsh_session
from ...utilities.paths import ProjectPaths
from ..shared import (
    derive_cli_flags_from_config,
    distribute_command_execution,
    document_command_execution,
)

CMD_NAME = "mesh"


def _execute_single_sample_id(
    paths: ProjectPaths, config: ProjectConfig, sample_id: str, size: int
) -> None:
    """Execute process for a single sample ID"""
    # get resolved config
    cmdconfig: MeshingConfig = config.mesh

    input_path = paths.sample(sample_id).triangulation().require_brep()

    process_paths = paths.sample(sample_id).meshing()
    process_paths.ensure_dir()
    mesh_path = process_paths.mesh

    metadata = run_gmsh_session(
        input_path,
        mesh_path,
        cmdconfig.boundary_margin_fraction,
        cmdconfig.substomatal_cavity_margin_fraction,
        cmdconfig.tolerance,
        cmdconfig.minimum_resolution,
        cmdconfig.maximum_resolution,
        cmdconfig.minimum_distance,
        cmdconfig.maximum_distance,
        cmdconfig.inlet_base_resolution_factor,
    )

    document_command_execution(
        process_paths,
        config,
        CMD_NAME,
        size,
        sample_id,
        inputs={"brep_model": str(input_path.expanduser().resolve())},
        outputs={"volumetric_mesh": str(mesh_path.expanduser().resolve())},
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
        description="generate a volumetric mesh model from a BREP representation using gmsh",
        help="generate a volumetric mesh model from a BREP representation using gmsh",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_input",
        type=str,
        help="Either a valid sample ID or path to a text file containing sample IDs (one per line)",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
