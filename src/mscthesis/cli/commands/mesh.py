from __future__ import annotations

import argparse

from ...config.declaration import MeshingConfig
from ...core.meshing.gmeshing import run_gmsh_session
from ..shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    setup_command,
)

CMD_NAME = "mesh"


def _cmd(args: argparse.Namespace) -> None:
    """Command declaration"""
    paths, config, sample_id = setup_command(args)

    cmdconfig: MeshingConfig = config.mesh

    input_path = paths.sample(sample_id).triangulation().require_brep()

    process_paths = paths.sample(sample_id).meshing()
    process_paths.ensure_dir()
    mesh_path = process_paths.mesh

    metadata = run_gmsh_session(
        input_path,
        mesh_path,
        cmdconfig.global_resolution_factor,
        cmdconfig.min_stomatal_feature,
        cmdconfig.max_stomatal_feature,
        cmdconfig.min_cellular_feature,
        cmdconfig.max_stomatal_dist_factor,
        cmdconfig.min_cellular_dist_factor,
        cmdconfig.max_cellular_dist_factor,
        cmdconfig.min_boundary_dist_factor,
        cmdconfig.max_boundary_dist_factor,
        cmdconfig.min_points_boundary,
        cmdconfig.max_points_boundary,
        cmdconfig.boundary_margin_fraction,
        cmdconfig.substomatal_cavity_margin_fraction,
        cmdconfig.tolerance,
    )

    document_command_execution(
        process_paths,
        config,
        CMD_NAME,
        sample_id,
        inputs={"brep_model": str(input_path.expanduser().resolve())},
        outputs={"volumetric_mesh": str(mesh_path.expanduser().resolve())},
        metadata=metadata,
    )

    return


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
        "sample_id",
        type=str,
        help="A valid sample ID",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
