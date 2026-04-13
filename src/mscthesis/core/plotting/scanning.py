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


def plot_heatmap(fig: plt.Figure, ax: plt.Axes, pivot: pd.DataFrame) -> None:
    c = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.to_numpy(),
        shading="auto",
        cmap="inferno",
    )
    fig.colorbar(c, ax=ax)
    return


def plot_scanning_results(
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
    # create substomatal conc figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Substomatal Concentration $\chi_i$")
    chi_i = df.pivot(**pivot_args, values="substomatal_mean")
    plot_heatmap(fig, ax, chi_i)
    save(fig, output_dir / "chi_i.pdf")
    #
    # create surface conc figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Surface Concentration $\chi_m$")
    chi_m = df.pivot(**pivot_args, values="surface_mean")
    plot_heatmap(fig, ax, chi_m)
    save(fig, output_dir / "chi_m.pdf")
    #
    # create ias drawdown figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"IAS Drawdown $\chi_i - \chi_m$")
    drawdown = chi_i - chi_m
    plot_heatmap(fig, ax, drawdown)
    save(fig, output_dir / "ias_drawdown.pdf")
    #
    # create ias resistance figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"IAS Resistance $r_{ias}$")
    plug_area = df.pivot(**pivot_args, values="plug_area")
    r_ias = (chi_i - chi_m) / (flux / plug_area)
    plot_heatmap(fig, ax, r_ias)
    save(fig, output_dir / "ias_resistance.pdf")
    #
    # create surface diversity figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Surface Diversity $\delta_m$")
    pivot_var = df.pivot(**pivot_args, values="surface_variance")
    pivot_mean = df.pivot(**pivot_args, values="surface_mean")
    diversity_m = np.sqrt(pivot_var) / pivot_mean
    plot_heatmap(fig, ax, diversity_m)
    save(fig, output_dir / "surface_diversity.pdf")
    #
    return
