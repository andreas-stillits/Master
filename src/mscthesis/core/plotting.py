from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utilities.plotting import (
    figure,
    gridlines,
    label_panel,
    panel_grid,
    save,
    set_axis_labels,
    use_style,
)


def _npy(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.abs(df[key].to_numpy())


def _rel_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs((x - y) / x)


def _split_by_order(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["order"] == 1], df[df["order"] == 2]


def _std_layout(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlog: bool = True,
    ylog: bool = False,
    legend: bool = True,
    loc: str = "center left",
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    set_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)
    if title is not None:
        ax.set_title(title)
    gridlines(ax)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if legend:
        ax.legend(loc=loc, bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1.0, None)
    return


def create_qoi_figure(
    df: pd.DataFrame, show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = panel_grid(nrows=2, ncols=1, size="double", sharex=True)
    df1, df2 = _split_by_order(df)
    x = _npy(df1, "resolution_factor")
    #
    keys_conc = ["substomatal_mean", "surface_mean", "top_mean"]
    labels_conc = [r"$C_{st}$", r"$C_{m}$", r"$C_{top}$"]

    keys_flux = ["stomatal_flux_equiv", "mesophyll_flux_equiv", "top_flux_direct"]
    labels_flux = [r"$A_{st}$", r"$A_{m}$", r"$A_{top}$"]

    for df, linestyle in zip([df1, df2], ["-", "--"], strict=True):
        for key, label in zip(keys_conc, labels_conc, strict=True):
            axs[0].plot(x, _npy(df, key), linestyle=linestyle, label=label)
        for key, label in zip(keys_flux, labels_flux, strict=True):
            axs[1].plot(x, _npy(df, key), linestyle=linestyle, label=label)

    _std_layout(
        axs[0], ylabel="Mean concentrations", title="Conc. QoI", ymin=0.0, ymax=1.05
    )
    _std_layout(
        axs[1], xlabel="Resolution factor", ylabel="Integral fluxes", title="Flux QoI"
    )

    if show:
        plt.show()
    return fig, axs


def create_bc_adherence_figure(
    df: pd.DataFrame, show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = panel_grid(nrows=2, ncols=1, size="double", sharex=True)
    df1, df2 = _split_by_order(df)
    x = _npy(df1, "resolution_factor")
    #
    for df, linestyle in zip([df1, df2], ["-", "--"], strict=True):
        flux_m_direct = _npy(df, "mesophyll_flux_direct")
        flux_m_equiv = _npy(df, "mesophyll_flux_equiv")
        flux_st_direct = _npy(df, "stomatal_flux_direct")
        flux_st_equiv = _npy(df, "stomatal_flux_equiv")
        flux_truth = flux_st_equiv[np.argmin(x)]
        flux_curved = _npy(df, "curved_flux_direct") / flux_truth
        fluc_top = _npy(df, "top_flux_direct") / flux_truth
        axs[0].plot(x, flux_curved, linestyle=linestyle, label="Curved BC")
        axs[0].plot(x, fluc_top, linestyle=linestyle, label="Top BC")
        #
        axs[1].plot(
            x,
            _rel_error(flux_m_equiv, flux_m_direct),
            linestyle=linestyle,
            label="mesophyll",
        )
        axs[1].plot(
            x,
            _rel_error(flux_st_equiv, flux_st_direct),
            linestyle=linestyle,
            label="stomatal",
        )

    _std_layout(axs[0], ylabel="Relative flux magnitude", title="Neumann BC adherence")
    _std_layout(
        axs[1],
        xlabel="Resolution factor",
        ylabel="Relative error",
        title="Robin BC adherence",
        ylog=False,
    )

    if show:
        plt.show()
    return fig, axs


def create_convergence_figure(
    df: pd.DataFrame,
    show: bool = False,
    tolerance: float = 0.01,
    ignore_threshold: float = 1e-4,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = figure(size="double")

    df1, df2 = _split_by_order(df)
    x = _npy(df1, "resolution_factor")
    #
    keys_conc = ["substomatal_mean", "surface_mean", "top_mean"]
    labels_conc = [r"$C_{st}$", r"$C_{m}$", r"$C_{top}$"]
    keys_flux = ["stomatal_flux_equiv", "mesophyll_flux_equiv", "top_flux_direct"]
    labels_flux = [r"$A_{st}$", r"$A_{m}$", r"$A_{top}$"]
    ax.hlines(
        tolerance,
        np.min(x),
        np.max(x),
        colors="darkred",
        linestyles=":",
        linewidth=1.5,
        label=f"{100*tolerance:.0f}% tolerance",
    )
    #
    for key, label in zip(keys_conc, labels_conc, strict=True):
        q1s = _npy(df1, key)
        q2 = _npy(df2, key)[np.argmin(x)]
        ax.plot(x, _rel_error(q2, q1s), label=label, linestyle="-")

    for key, label in zip(keys_flux, labels_flux, strict=True):
        q1s = _npy(df1, key)
        q2 = _npy(df2, key)[np.argmin(x)]
        if key == "top_flux_direct" and q1s[np.argmin(x)] < ignore_threshold:
            continue

        ax.plot(x, _rel_error(q2, q1s), label=label, linestyle="-")

    _std_layout(
        ax,
        xlabel="Resolution factor",
        ylabel="Relative error",
        title="Convergence to target CG2 solution",
        ylog=True,
        ymin=1e-3,
    )

    if show:
        plt.show()
    return fig, ax


def create_solution_figure(
    df: pd.DataFrame, show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = panel_grid(nrows=2, ncols=2, size="double", sharex=True)
    if show:
        plt.show()
    return fig, axs


def plot_validation_results(
    df: pd.DataFrame,
    output_dir: Path,
    plot_fig1: bool = True,
    plot_fig2: bool = True,
    plot_fig3: bool = True,
    plot_fig4: bool = True,
    tolerance: float = 0.01,
    ignore_threshold: float = 1e-4,
    show: bool = False,
) -> None:
    """
    Plot validation results and save as .pdf
    """
    use_style()
    #
    if plot_fig1:
        fig1, axs1 = create_qoi_figure(df, show=show)
        label_panel(axs1[0], "(a)")
        label_panel(axs1[1], "(b)")
        save(fig1, output_dir / "qoi_metrics.pdf")
    #
    if plot_fig2:
        fig2, axs2 = create_bc_adherence_figure(df, show=show)
        label_panel(axs2[0], "(c)")
        label_panel(axs2[1], "(d)")
        save(fig2, output_dir / "bc_adherence.pdf")
    #
    if plot_fig3:
        fig3, ax3 = create_convergence_figure(
            df, tolerance=tolerance, ignore_threshold=ignore_threshold, show=show
        )
        label_panel(ax3, "(e)")
        save(fig3, output_dir / "convergence.pdf")
    #
    if plot_fig4:
        # fig4, axs4 = create_solution_figure(df, show=show)
        # save(fig4, output_dir / "solution.pdf")
        pass
    #
    plt.close("all")
    return
