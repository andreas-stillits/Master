from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mscthesis.core.io import load_dataframe
from mscthesis.utilities.plotting import (
    figure,
    save,
    set_axis_labels,
    use_style,
)

PLOT_KEY = "ias resistance"
SCAN_PATH = "/home/andreasstillits/coding/master/.treasury/processes/scanning/00019/dataframe.csv"
SAVE_PATH = "/home/andreasstillits/coding/master/tmp/scan.pdf"


def _std_layout(ax: plt.Axes) -> None:
    set_axis_labels(ax, r"Absorption $\phi$", r"Transport $\gamma$")
    ax.set_xlim(0.01, 100.0)
    ax.set_ylim(0.01, 100.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(PLOT_KEY)
    ax.plot([0.01, 100.0], [0.01, 100.0], color="darkred", linestyle=":", linewidth=1.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.5)

    return


def _plot_heatmap(fig: plt.Figure, ax: plt.Axes, pivot: pd.DataFrame) -> None:
    c = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.to_numpy(),
        shading="auto",
        cmap="inferno",
    )
    fig.colorbar(c, ax=ax, label=PLOT_KEY)
    return


def main() -> int:
    use_style()
    df = load_dataframe(SCAN_PATH)
    fig, ax = figure(size="double")
    _std_layout(ax)
    #
    pivot_args = {
        "index": "transport",
        "columns": "absorption",
    }

    if PLOT_KEY in df.columns:
        _plot_heatmap(fig, ax, df.pivot(**pivot_args, values=PLOT_KEY))
    elif PLOT_KEY == "surface diversity":
        # pivot df on absorption and transport, then compute diversity as the square root of surface_variance over surface_mean
        pivot_var = df.pivot(**pivot_args, values="surface_variance")
        pivot_mean = df.pivot(**pivot_args, values="surface_mean")
        _plot_heatmap(fig, ax, (np.sqrt(pivot_var) / pivot_mean))
    elif PLOT_KEY == "airspace diversity":
        # pivot df on absorption and transport, then compute diversity as the square root of airspace_variance over airspace_mean
        pivot_var = df.pivot(**pivot_args, values="airspace_variance")
        pivot_mean = df.pivot(**pivot_args, values="airspace_mean")
        _plot_heatmap(fig, ax, (np.sqrt(pivot_var) / pivot_mean))
    elif PLOT_KEY == "ias resistance":
        pivot_stomatal = df.pivot(**pivot_args, values="substomatal_mean")
        pivot_surface = df.pivot(**pivot_args, values="surface_mean")
        pivot_flux = df.pivot(**pivot_args, values="stomatal_flux_equiv")
        _plot_heatmap(fig, ax, (pivot_stomatal - pivot_surface) / pivot_flux)

    #
    try:
        save(fig, SAVE_PATH)
    except Exception as e:
        print("Error occurred while saving the plot")
        raise e

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
