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
    # plot 1: flux at stomatal and mesophyll boundary
    diff_stomatal = (
        (an_stomatal_direct - an_stomatal_equiv) / an_stomatal_direct
        if np.all(an_stomatal_direct > 0)
        else np.zeros_like(an_stomatal_direct)
    )
    diff_mesophyll = (
        (an_mesophyll_direct - an_mesophyll_equiv) / an_mesophyll_direct
        if np.all(an_mesophyll_direct > 0)
        else np.zeros_like(an_mesophyll_direct)
    )
    diff_total = (
        (an_stomatal_direct - an_mesophyll_direct - an_curved - an_top)
        / an_stomatal_direct
        if np.all(an_stomatal_direct > 0)
        else np.zeros_like(an_stomatal_direct)
    )
    diff_active = (
        (an_stomatal_direct - an_mesophyll_direct) / an_stomatal_direct
        if np.all(an_stomatal_direct > 0)
        else np.zeros_like(an_stomatal_direct)
    )
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
    make_lineplot(
        axes,
        resolution_factors,
        diff_total,
        label="Total flux conservation",
        color="#2ca02c",
    )
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


def make_concentration_plots() -> None:

    return


def plot_validation_results(
    df: pd.DataFrame, base_path: Path, show: bool = False
) -> None:
    flux_path = base_path / "flux_conservation.png"
    make_flux_plots(
        flux_path,
        np.abs(df["resolution_factor"].to_numpy()),
        np.abs(df["stomatal_flux_direct"].to_numpy()),
        np.abs(df["stomatal_flux_equiv"].to_numpy()),
        np.abs(df["mesophyll_flux_direct"].to_numpy()),
        np.abs(df["mesophyll_flux_equiv"].to_numpy()),
        np.abs(df["curved_flux_direct"].to_numpy()),
        np.abs(df["top_flux_direct"].to_numpy()),
        show=show,
    )

    return
