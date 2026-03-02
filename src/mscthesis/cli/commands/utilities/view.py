from __future__ import annotations

import argparse

from mpi4py import MPI

from ....config.declaration import ProjectConfig
from ....utilities.paths import ProjectPaths


def _cmd(args: argparse.Namespace, comm: MPI.Intracomm) -> None:
    """Command to generate a view of samples, but by processes"""
    config: ProjectConfig = args.config
    # get samples path
    paths: ProjectPaths = ProjectPaths(config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_inventories_root()
    paths.ensure_processes_root()

    # get sample IDs from inventory
    sample_ids = list(paths.samples.iterdir())
    sample_ids = [p.name for p in sample_ids if p.is_dir()]

    # get names of subdirectories in sample_ids, which correspond to processes
    process_names = set()
    for sample_id in sample_ids:
        sample_path = paths.samples / sample_id
        for process_path in sample_path.iterdir():
            if process_path.is_dir():
                process_names.add(process_path.name)

    process_names = sorted(process_names)

    # create a dir in processes for each process name
    for process_name in process_names:
        process_dir = paths.processes / process_name
        process_dir.mkdir(exist_ok=True)

        # for each sample, check if it has a subdir with the process name, and if so, create a symlink to it in the process dir
        for sample_id in sample_ids:
            sample_process_path = paths.samples / sample_id / process_name
            if sample_process_path.is_dir():
                symlink_path = process_dir / sample_id
                if not symlink_path.exists():
                    symlink_path.symlink_to(sample_process_path)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the view-by-process command on the given subparsers object."""
    parser = subparsers.add_parser(
        "view-by-process",
        description="Generate a view of samples, but by processes",
        help="Generate a view of samples, but by processes",
        epilog="Example: msc view-by-process \n",
    )

    parser.set_defaults(cmd=_cmd)
