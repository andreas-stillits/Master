from __future__ import annotations

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from mscthesis.config.declaration import ProjectConfig
from mscthesis.core.io import load_dataframe
from mscthesis.core.plotting.scanning import plot_heatmap, std_layout
from mscthesis.utilities.ids import validate_sample_id
from mscthesis.utilities.paths import ProjectPaths, require_file
from mscthesis.utilities.plotting import (
    figure,
    use_style,
)

PLOT_KEY = "ias resistance"
SCAN_PATH = "/home/andreasstillits/coding/master/.treasury/processes/scanning/00019/dataframe.csv"


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("sample_id", type=str, help="sample id to inspect")
    parser.add_argument(
        "plot_key", type=str, help="key to plot (e.g. 'ias_resistance')"
    )
    args = parser.parse_args()
    sample_id = validate_sample_id(args.sample_id, required_digits=5)
    plot_key = args.plot_key

    paths = ProjectPaths(ProjectConfig().behavior.storage_root)
    scan_path = paths.sample(sample_id).scanning().scan
    require_file(scan_path)

    use_style()
    df = load_dataframe(scan_path)
    fig, ax = figure(size="single")
    std_layout(ax, plot_key)
    #
    pivot_args = {
        "index": "transport",
        "columns": "absorption",
    }

    if plot_key in df.columns:
        plot_heatmap(fig, ax, df.pivot(**pivot_args, values=plot_key))
    elif plot_key == "surface_diversity":
        # pivot df on absorption and transport, then compute diversity as the square root of surface_variance over surface_mean
        pivot_var = df.pivot(**pivot_args, values="surface_variance")
        pivot_mean = df.pivot(**pivot_args, values="surface_mean")
        plot_heatmap(fig, ax, (np.sqrt(pivot_var) / pivot_mean))
    elif plot_key == "airspace_diversity":
        # pivot df on absorption and transport, then compute diversity as the square root of airspace_variance over airspace_mean
        pivot_var = df.pivot(**pivot_args, values="airspace_variance")
        pivot_mean = df.pivot(**pivot_args, values="airspace_mean")
        plot_heatmap(fig, ax, (np.sqrt(pivot_var) / pivot_mean))
    elif plot_key == "ias_resistance":
        pivot_stomatal = df.pivot(**pivot_args, values="substomatal_mean")
        pivot_surface = df.pivot(**pivot_args, values="surface_mean")
        pivot_flux = df.pivot(**pivot_args, values="stomatal_flux_equiv")
        plot_heatmap(fig, ax, (pivot_stomatal - pivot_surface) / pivot_flux)

    plt.show()
    plt.close("all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
