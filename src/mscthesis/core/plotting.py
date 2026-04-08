from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utilities.plotting import (
    figure,
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


def create_qoi_figure(
    df: pd.DataFrame, show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = panel_grid(nrows=1, ncols=2, size="double", sharex=True)
    if show:
        plt.show()
    return fig, axs


def create_bc_adherence_figure(
    df: pd.DataFrame, show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = panel_grid(nrows=1, ncols=2, size="double", sharex=True)
    if show:
        plt.show()
    return fig, axs


def create_convergence_figure(
    df: pd.DataFrame, show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = panel_grid(nrows=1, ncols=2, size="double", sharex=True)
    if show:
        plt.show()
    return fig, axs


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
    show: bool = False,
) -> None:
    """
    Plot validation results and save as .pdf
    """
    use_style()
    #
    fig1, axs1 = create_qoi_figure(df, show=show)
    save(fig1, output_dir / "qoi_metrics.pdf")
    #
    # fig2, axs2 = create_bc_adherence_figure(df, show=show)
    # save(fig2, output_dir / "bc_adherence.pdf")
    # #
    # fig3, axs3 = create_convergence_figure(df, show=show)
    # save(fig3, output_dir / "convergence.pdf")
    # #
    # fig4, axs4 = create_solution_figure(df, show=show)
    # save(fig4, output_dir / "solution.pdf")
    #
    plt.close("all")
    return
