from __future__ import annotations

import argparse

import pandas as pd

from ...config.declaration import ScanningConfig
from ...core.io import save_dataframe
from ...core.plotting.scanning import plot_scanning_results
from ...core.scanning import (
    generate_workload,
    run_batch,
)
from ...core.solvers import UniformSolver, UniformSolverConfig
from ...utilities.parallel import distribute, generate_batches_round_robin
from ..shared import (
    derive_cli_flags_from_config,
    document_command_execution,
    setup_command,
)

CMD_NAME = "scan"


def _cmd(args: argparse.Namespace) -> None:
    """Command declaration"""
    paths, config, sample_id = setup_command(args)

    # get resolved config
    cmdconfig: ScanningConfig = config.scan

    mesh_path = paths.sample(sample_id).meshing().require_mesh()
    process_paths = paths.sample(sample_id).scanning()
    process_paths.ensure_dir()
    scan_path = process_paths.scan

    absorption_range = (
        cmdconfig.absorption_min,
        cmdconfig.absorption_max,
        cmdconfig.absorption_num,
    )

    transport_range = (
        cmdconfig.transport_min,
        cmdconfig.transport_max,
        cmdconfig.transport_num,
    )

    solver_config = UniformSolverConfig(
        cmdconfig.stomatal_aspect,
        cmdconfig.ksp_type,
        cmdconfig.ksp_rtol,
        cmdconfig.pc_type,
        cmdconfig.quad_degree,
        cmdconfig.order,
    )

    workload = generate_workload(
        transport_range,
        absorption_range,
        cmdconfig.compensation,
        cmdconfig.geometry_factor,
    )
    batches = generate_batches_round_robin(workload, cmdconfig.max_workers)

    results = distribute(
        run_batch,
        batches,
        cmdconfig.max_workers,
        mesh_path,
        solver_config,
        UniformSolver,
    )

    dataframe = pd.DataFrame(results)
    dataframe.sort_values(["absorption", "transport"], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    # annotate sample id to ease later aggregation across scans
    dataframe["sample_id"] = sample_id

    save_dataframe(dataframe, scan_path)

    plot_scanning_results(dataframe, process_paths.plots)

    metadata = {
        "num_simulations": len(workload),
    }

    document_command_execution(
        process_paths,
        config,
        CMD_NAME,
        sample_id,
        inputs={"volumetric_mesh": str(mesh_path.expanduser().resolve())},
        outputs={
            "scan_results": str(scan_path.expanduser().resolve()),
            "plot_results": str(process_paths.plots.expanduser().resolve()),
        },
        metadata=metadata,
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command to a subparser"""
    # declare command name - must match name of its configs attribute in ProjectConfig
    parser = subparsers.add_parser(
        CMD_NAME,
        description="scan over many absorption and transport parameters",
        help="scan over many absorption and transport parameters",
        epilog=f"msc {CMD_NAME} [options] <sample_id>",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="A valid sample ID",
    )
    parser = derive_cli_flags_from_config(parser, CMD_NAME)
    parser.set_defaults(cmd=_cmd)
