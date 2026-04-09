from __future__ import annotations

import argparse

import pandas as pd

from ...config.declaration import (
    ValidationConfig,
)
from ...core.io import (
    load_dataframe,
    save_dataframe,
)
from ...core.plotting import plot_validation_results
from ...core.solvers import (
    DiffusionSolver,
    DiffusionSolverConfig,
    UniformSolver,
    UniformSolverConfig,
)
from ...core.validation import (
    copy_reference_files,
    meshing,
    prepare_batches,
    solving,
)
from ...utilities.parallel import distribute
from ...utilities.paths import require_file
from ..shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    dump_resolved_command_config,
    setup_command,
)

CMD_NAME = "validation"


def _cmd(args: argparse.Namespace) -> None:
    """Command declaration"""
    paths, config, sample_id = setup_command(args)

    cmdconfig: ValidationConfig = config.validation

    # define paths
    validation_paths = paths.validate(cmdconfig.tag)
    validation_paths.verify_tag()
    sample_paths = paths.sample(sample_id)
    sample_dir = sample_paths.dir
    brep_path = sample_paths.triangulation().require_brep()

    # ---------------------------------------------------------------------------------

    # make a hard copy of the contents of sample_dir in validation_dir for reference
    # only copy files for up to and including meshing
    if not validation_paths.dir.exists():
        copy_reference_files(sample_dir, validation_paths)

    # ---------------------------------------------------------------------------------

    if not cmdconfig.no_meshing or not cmdconfig.no_solving:
        batches = prepare_batches(
            cmdconfig.resolution_factor_max,
            cmdconfig.resolution_factor_num,
            validation_paths,
        )

    # ---------------------------------------------------------------------------------

    if not cmdconfig.no_meshing:

        meshing_args = (
            config.mesh.min_stomatal_feature,
            config.mesh.min_cellular_feature,
            config.mesh.min_stomatal_dist_factor,
            config.mesh.max_stomatal_dist_factor,
            config.mesh.min_cellular_dist_factor,
            config.mesh.max_cellular_dist_factor,
            config.mesh.min_boundary_dist_factor,
            config.mesh.max_boundary_dist_factor,
            config.mesh.min_points_boundary,
            config.mesh.max_points_boundary,
            config.mesh.boundary_margin_fraction,
            config.mesh.substomatal_cavity_margin_fraction,
            config.mesh.tolerance,
        )

        _ = distribute(meshing, batches, cmdconfig.workers, brep_path, *meshing_args)
        dump_resolved_command_config(
            config, "mesh", validation_paths.ensure_meshes_dir() / "config.json"
        )

    # ---------------------------------------------------------------------------------

    if not cmdconfig.no_solving:
        if cmdconfig.problem_type == "uniform":
            SolverClass = UniformSolver
            solver_config = UniformSolverConfig(
                cmdconfig.stomatal_aspect,
                cmdconfig.stomatal_epsilon,
                cmdconfig.ksp_rtol,
                cmdconfig.quad_degree,
                order=1,  # placeholder, will be set in solving()
            )
            parameters = cmdconfig.parameters_uniform
        elif cmdconfig.problem_type == "diffusion":
            SolverClass = DiffusionSolver
            solver_config = DiffusionSolverConfig(
                cmdconfig.stomatal_aspect,
                cmdconfig.stomatal_epsilon,
                cmdconfig.ksp_rtol,
                cmdconfig.quad_degree,
                order=1,  # placeholder, will be set in solving()
            )
            parameters = cmdconfig.parameters_diffusion
        else:
            raise ValueError(f"Unsupported problem type: {cmdconfig.problem_type}")

        qoi_metrics = distribute(
            solving,
            batches,
            cmdconfig.workers,
            SolverClass,
            solver_config,
            validation_paths,
            parameters,
        )

        dataframe = pd.DataFrame(qoi_metrics)
        dataframe.sort_values(["resolution_factor", "order"]).reset_index(
            drop=True, inplace=True
        )
        save_dataframe(dataframe, validation_paths.results)

    # ---------------------------------------------------------------------------------

    if not validation_paths.dir.exists():
        raise FileNotFoundError(
            "Validation not previously computed. Please execute with no_meshing and no_solving flags set to false."
        )

    dataframe = load_dataframe(require_file(validation_paths.results))
    plot_validation_results(
        dataframe, validation_paths.ensure_plots_dir(), show=(not cmdconfig.no_show)
    )

    # ---------------------------------------------------------------------------------

    document_command_execution(
        validation_paths,
        config,
        CMD_NAME,
        sample_id,
        inputs={"surface_mesh": str(brep_path.expanduser().resolve())},
        outputs={
            "meshes": str(validation_paths.ensure_meshes_dir().expanduser().resolve()),
            "solutions": str(
                validation_paths.ensure_solutions_dir().expanduser().resolve()
            ),
            "results": str(validation_paths.results.expanduser().resolve()),
            "plots": str(validation_paths.ensure_plots_dir().expanduser().resolve()),
        },
        metadata={
            "origin_copy": str(
                validation_paths.ensure_reference_dir().expanduser().resolve()
            ),
        },
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="validate the proposed mesh resolution strategy for either diffusion or reaction problems",
        help="validate the proposed mesh resolution strategy for either diffusion or reaction problems",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="A valid sample ID",
    )

    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
