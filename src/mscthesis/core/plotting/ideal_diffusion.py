from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...utilities.plotting import (
    figure,
    save,
    set_axis_labels,
    use_style,
)


def std_layout(ax: plt.Axes, title: str) -> None:
    set_axis_labels(ax, r"Absorption $\phi$", r"Transport $\gamma$", title=title)
    ax.set_xlim(0.01, 100.0)
    ax.set_ylim(0.01, 100.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([0.01, 100.0], [0.01, 100.0], color="darkred", linestyle=":", linewidth=1.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.5)

    return


def plot_diffusion_results(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    use_style()
    #
    pivot_args = {
        "index": "transport",
        "columns": "absorption",
    }
    # create flux figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Assimilation Rate $\alpha_N$")
    flux = df.pivot(**pivot_args, values="stomatal_flux_equiv")
    plot_heatmap(fig, ax, flux)
    save(fig, output_dir / "flux.pdf")

    #
    return
