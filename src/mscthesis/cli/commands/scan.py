from __future__ import annotations

import argparse

import pandas as pd

from ...config.declaration import ScanningConfig
from ...core.io import save_dataframe
from ...core.scanning import (
    generate_workload,
    run_batch,
)
from ...core.solvers import UniformSolver, UniformSolverConfig
from ...utilities.fetching import fetch_manifest_quantities
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

    fetched_data = fetch_manifest_quantities(
        paths.sample(sample_id).require_process("meshing").require_manifest(),
        "plug_aspect",
        "mesophyll_area_fraction",
    )

    plug_aspect = fetched_data["plug_aspect"]
    stomatal_area_fraction = cmdconfig.stomatal_aspect**2 / plug_aspect**2
    mesophyll_area_fraction = fetched_data["mesophyll_area_fraction"]

    solver_config = UniformSolverConfig(
        cmdconfig.compensation,
        plug_aspect,
        cmdconfig.stomatal_aspect,
        cmdconfig.stomatal_epsilon,
        stomatal_area_fraction,
        mesophyll_area_fraction,
        cmdconfig.order,
    )

    workload = generate_workload(absorption_range, transport_range)
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
    dataframe.sort_values(["absorption", "transport"]).reset_index(drop=True)

    save_dataframe(dataframe, scan_path)

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
