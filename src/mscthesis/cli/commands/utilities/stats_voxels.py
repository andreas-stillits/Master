from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ....config.declaration import ProjectConfig
from ....utilities.paths import ProjectPaths


def _cmd(args: argparse.Namespace) -> None:
    """Command to aggregate statistics for voxel data"""
    config: ProjectConfig = args.config
    # get samples path
    paths: ProjectPaths = ProjectPaths(config.behavior.storage_root)
    paths.require_base()
    paths.ensure_samples_root()
    paths.ensure_stats_root()

    # get sample IDs from inventory
    sample_ids = list(paths.samples.iterdir())

    # get a list of paths to all synthesis manifests
    manifests: list[Path] = []
    sample_ids: list[str] = []

    dataframe = pd.DataFrame(
        columns=["sample_id", "mean_radius", "mean_porosity", "type"]
    )

    for sample_path in paths.samples.iterdir():
        sample_id = sample_path.name
        sample_ids.append(sample_id) if sample_path.is_dir() else None
        manifest: Path = paths.sample(sample_id).synthesis().require_manifest()
        manifests.append(manifest)
        # print(f"Found manifest for sample {sample_id} at {manifest}")
        # read the manifest as a dictionary using json
        with open(manifest, "r") as f:
            manifest_dict = json.load(f)
            meta_dict = manifest_dict["meta"]
            # print(f"Manifest for sample {sample_id}: {manifest_dict}")
            mean_radius = meta_dict["mean_radius"]
            mean_porosity = meta_dict["mean_porosity"]
            type = meta_dict["type"]
            dataframe = pd.concat(
                [
                    dataframe,
                    pd.DataFrame(
                        {
                            "sample_id": [sample_id],
                            "mean_radius": [mean_radius],
                            "mean_porosity": [mean_porosity],
                            "type": [type],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # save the dataframe as a csv file in the stats directory
    stats_path = paths.stats / "voxel_stats.csv"
    dataframe.to_csv(stats_path, index=False, decimal=",", sep="\t")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the stats-voxels command on the given subparsers object."""
    parser = subparsers.add_parser(
        "stats-voxels",
        description="Aggregate statistics for voxel data",
        help="Aggregate statistics for voxel data",
        epilog="Example: msc stats-voxels [options]\n",
    )
    parser.set_defaults(cmd=_cmd)
