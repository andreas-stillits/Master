from __future__ import annotations

import argparse

from ...config.declaration import UniformSolutionConfig
from ...core.io import load_volumetric_mesh, save_fem_solution
from ...core.solvers import (
    MeshContext,
    UniformSolver,
    UniformSolverConfig,
)
from ..shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    setup_command,
)

CMD_NAME = "solve-uniform"


def _cmd(args: argparse.Namespace) -> None:
    """Command declaration"""
    paths, config, sample_id = setup_command(args)

    cmdconfig: UniformSolutionConfig = config.solve_uniform

    input_path = paths.sample(sample_id).meshing().require_mesh()

    process_paths = paths.sample(sample_id).solving()
    process_paths.ensure_dir()
    solution_path = process_paths.solution

    # insert _<args.extension> before file suffix if extension is provided
    if args.extension:
        solution_path = solution_path.with_name(
            f"{solution_path.stem}_{args.extension}{solution_path.suffix}"
        )

    solver_config = UniformSolverConfig(
        cmdconfig.stomatal_aspect,
        cmdconfig.stomatal_epsilon,
        cmdconfig.ksp_rtol,
        cmdconfig.quad_degree,
        cmdconfig.order,
    )

    mesh_ctx: MeshContext = load_volumetric_mesh(input_path)

    solver = UniformSolver(
        solver_config,
        mesh_ctx,
    )

    solution, analysis = solver.solve_for(
        cmdconfig.absorption,
        cmdconfig.transport,
        cmdconfig.compensation,
    )

    if not cmdconfig.no_save:
        save_fem_solution(solution, mesh_ctx, solution_path)

    metadata = {
        **analysis,
    }

    document_command_execution(
        process_paths,
        config,
        CMD_NAME,
        sample_id,
        inputs={"volumetric_mesh": str(input_path.expanduser().resolve())},
        outputs={"solution": str(solution_path.expanduser().resolve())},
        metadata=metadata,
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="solve the diffusion problem for a given choice of parameters",
        help="solve the diffusion problem for a given choice of parameters",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="A valid sample ID",
    )
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        help="(optional) identifier to add to the standard filename, e.g. solution.bp --> solution_<extension>.bp",
    )

    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
