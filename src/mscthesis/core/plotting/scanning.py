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


# write a heat map version that uses scatter on the raw dataframe instead of pivoting to avoid the need for a regular grid
def plot_scatter_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    df: pd.DataFrame,
    values: pd.Series,
) -> None:
    sc = ax.scatter(
        df["absorption"],
        df["transport"],
        c=values,
        s=100.0,
        cmap="inferno",
    )
    fig.colorbar(sc, ax=ax)
    return


def plot_scanning_results(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    use_style()
    #
    # create flux figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Assimilation Rate $\alpha_N$")
    flux = df["mesophyll_flux_equiv"]
    plot_scatter_heatmap(fig, ax, df, flux)
    save(fig, output_dir / "flux.pdf")
    #
    # create substomatal conc figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Substomatal Concentration $\chi_i$")
    chi_i = df["substomatal_mean"]

    #
    # SHOWCASING THEORETICAL CURVES
    chiis = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    phis = np.logspace(-2, 2, 50)
    chi_ = 0.1
    epsilon = 1 / 2.1
    for chii in chiis:
        gammas = (chii - chi_) / (1 - chii) * (epsilon * phis) / (epsilon + phis)
        ax.plot(phis, gammas, color="lightblue", linestyle="-", linewidth=0.5)
    #
    #
    plot_scatter_heatmap(fig, ax, df, chi_i)
    save(fig, output_dir / "chi_i.pdf")
    #
    # create surface conc figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Surface Concentration $\chi_m$")
    chi_m = df["surface_mean"]
    plot_scatter_heatmap(fig, ax, df, chi_m)
    save(fig, output_dir / "chi_m.pdf")
    #
    # create ias drawdown figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"IAS Drawdown $\chi_i - \chi_m$")
    drawdown = chi_i - chi_m
    plot_scatter_heatmap(fig, ax, df, drawdown)
    save(fig, output_dir / "ias_drawdown.pdf")
    #
    # create ias resistance figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"IAS Resistance $r_{ias}$")
    plug_area = df["plug_area"]
    r_ias = (chi_i - chi_m) / (flux / plug_area)
    plot_scatter_heatmap(fig, ax, df, r_ias)
    save(fig, output_dir / "ias_resistance.pdf")
    #
    # create surface diversity figure
    fig, ax = figure(size="single")
    std_layout(ax, title=r"Surface Diversity $\delta_m$")
    var = df["surface_variance"]
    mean = df["surface_mean"]
    variation = np.sqrt(var) / mean
    plot_scatter_heatmap(fig, ax, df, variation)
    save(fig, output_dir / "surface_diversity.pdf")
    #
    return
