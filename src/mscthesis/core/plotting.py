from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_lineplot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
    label: str | None = None,
    color: str | None = None,
) -> None:
    """
    Make a line plot with optional error bars.
    """
    ax.plot(x, y, label=label, marker="x", color=color)
    if yerr is not None:
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color=color)


def make_flux_plots(
    plot_path: str | Path,
    resolution_factors: np.ndarray,
    an_stomatal_direct: np.ndarray,
    an_stomatal_equiv: np.ndarray,
    an_mesophyll_direct: np.ndarray,
    an_mesophyll_equiv: np.ndarray,
    an_curved: np.ndarray,
    an_top: np.ndarray,
    show: bool = False,
) -> None:
    """
    Plot the degree to which BCs are upheld and total flux conserved.
    Assumes positive entries only.
    """
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    calc = lambda x, y: np.abs((x - y) / x) if np.all(x > 0) else np.zeros_like(x)

    # plot 1: flux at stomatal and mesophyll boundary
    diff_stomatal = calc(an_stomatal_direct, an_stomatal_equiv)
    diff_mesophyll = calc(an_mesophyll_direct, an_mesophyll_equiv)
    diff_total = calc(an_stomatal_equiv, an_mesophyll_equiv + an_curved + an_top)
    diff_active = calc(an_stomatal_equiv, an_mesophyll_equiv)

    make_lineplot(
        axes,
        resolution_factors,
        diff_stomatal,
        label="Stomatal flux adherence",
        color="#D70606",
    )
    make_lineplot(
        axes,
        resolution_factors,
        diff_mesophyll,
        label="Mesophyll flux adherence",
        color="#1f77b4",
    )
    # make_lineplot(
    #     axes,
    #     resolution_factors,
    #     diff_total,
    #     label="Total flux conservation",
    #     color="#2ca02c",
    # )
    make_lineplot(
        axes,
        resolution_factors,
        diff_active,
        label="Active flux conservation",
        color="#ff7f0e",
    )
    axes.plot(0, 0, "w.")
    axes.set_xscale("log")
    axes.set_xlabel("Resolution factor")
    axes.set_ylabel("Relative flux error")
    axes.set_title("Flux conservation and BC adherence")
    axes.legend()
    axes.grid(linestyle="-.", alpha=0.5)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)

    if show:
        plt.show()

    return


def make_concentration_plots(
    plot_path: str | Path,
    resolution_factors: np.ndarray,
    conc_i: np.ndarray,
    conc_t: np.ndarray,
    conc_a: np.ndarray,
    conc_m: np.ndarray,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    make_lineplot(
        axes,
        resolution_factors,
        conc_i,
        label="Substomatal concentration",
        color="#D70606",
    )
    make_lineplot(
        axes,
        resolution_factors,
        conc_t,
        label="Top concentration",
        color="#1f77b4",
    )
    make_lineplot(
        axes,
        resolution_factors,
        conc_a,
        label="Airspace concentration",
        color="#2ca02c",
    )
    make_lineplot(
        axes,
        resolution_factors,
        conc_m,
        label="Mesophyll concentration",
        color="#ff7f0e",
    )
    axes.plot(0, 0, "w.")
    axes.set_xscale("log")
    axes.set_xlabel("Resolution factor")
    axes.set_ylabel("Concentration []")
    axes.set_title("Concentration at key locations")
    axes.legend()
    axes.grid(linestyle="-.", alpha=0.5)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)

    if show:
        plt.show()

    return


def plot_validation_results(
    df: pd.DataFrame, base_path: Path, show: bool = False
) -> None:
    flux_path = base_path / "flux_conservation.png"

    transform = lambda s: np.abs(df[s].to_numpy())
    make_flux_plots(
        flux_path,
        transform("resolution_factor"),
        transform("stomatal_flux_direct"),
        transform("stomatal_flux_equiv"),
        transform("mesophyll_flux_direct"),
        transform("mesophyll_flux_equiv"),
        transform("curved_flux_direct"),
        transform("top_flux_direct"),
        show=show,
    )
    conc_path = base_path / "concentrations.png"
    make_concentration_plots(
        conc_path,
        transform("resolution_factor"),
        transform("substomatal_mean"),
        transform("top_mean"),
        transform("airspace_mean"),
        transform("surface_mean"),
        show=show,
    )

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes.plot(
        transform("resolution_factor"),
        transform("curved_flux_direct"),
        marker="x",
        label="Curved flux",
    )
    axes.plot(
        transform("resolution_factor"),
        transform("top_flux_direct"),
        marker="x",
        label="Top flux",
    )
    axes.plot(
        transform("resolution_factor"),
        transform("stomatal_flux_equiv"),
        marker="x",
        label="Stomatal flux",
    )
    axes.plot(
        transform("resolution_factor"),
        transform("mesophyll_flux_equiv"),
        marker="x",
        label="Mesophyll flux",
    )
    axes.set_xscale("log")
    axes.set_xlabel("Resolution factor")
    axes.set_ylabel("Number of cells")
    axes.set_title("Mesh complexity")
    axes.grid(linestyle="-.", alpha=0.5)
    axes.legend()
    plt.tight_layout()
    plt.show()

    return
