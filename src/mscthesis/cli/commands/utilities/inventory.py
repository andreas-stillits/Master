from __future__ import annotations

import argparse

from ....config.declaration import ProjectConfig
from ....utilities.paths import ProjectPaths


def _cmd(args: argparse.Namespace) -> None:
    """Command to update the inventory of available sample IDs."""
    config: ProjectConfig = args.config
    # get samples path
    paths: ProjectPaths = ProjectPaths(config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_inventories_root()

    # get a list of directory names present in samples_root
    sample_ids = list(paths.samples.iterdir())
    # filter to directories only
    sample_ids = [sid for sid in sample_ids if sid.is_dir()]
    # sort for consistency
    sample_ids.sort()

    # declare inventory
    inventory_path = paths.inventories / "all_samples.txt"
    with open(inventory_path, "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id.name}\n")

    print(f"Inventory updated with {len(sample_ids)} sample IDs at {inventory_path}")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the update-inventory command on the given subparsers object."""
    parser = subparsers.add_parser(
        "update-inventory",
        description="Update inventory of available sample IDs",
        help="Update the inventory of available sample IDs",
        epilog="Example: msc update-inventory \n",
    )
    parser.set_defaults(cmd=_cmd)
