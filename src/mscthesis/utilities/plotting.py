from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .colors import CYCLE

_DEFAULT_SUFFIX = ".pdf"
_DEFAULT_RCPARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
}


def use_style(overrides: dict[str, Any] | None = None) -> None:
    mpl.rcParams.update(_DEFAULT_RCPARAMS)
    if overrides:
        mpl.rcParams.update(overrides)
    return


def reset_style() -> None:
    mpl.rcdefaults()
    return


# ---------------------------------------------------------------------------
# helpers


def save(
    fig: plt.Figure,
    path: str | Path,
    transparent: bool = False,
    **kwargs: Any,
) -> None:
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(_DEFAULT_SUFFIX)
    fig.savefig(path, transparent=transparent, **kwargs)
    return


def label_panel(
    ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.02, **kwargs: Any
) -> None:
    defaults = dict(transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")
    defaults.update(kwargs)
    ax.text(x, y, label, **defaults)
    return


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return


FIG_SIZES = {
    "single": (3.4, 2.6),  # single-column style
    "double": (7.0, 4.2),  # double-column style
    "square": (3.4, 3.4),
    "talk": (6.0, 4.0),
}


def figure(
    size: str | tuple[float, float] = "single", **kwargs: Any
) -> tuple[plt.Figure, plt.Axes]:
    figsize = FIG_SIZES[size] if isinstance(size, str) else size
    return plt.subplots(figsize=figsize, **kwargs)


def panel_grid(
    nrows: int,
    ncols: int,
    size: str | tuple[float, float] = "double",
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray]:
    figsize = FIG_SIZES[size] if isinstance(size, str) else size
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def set_axis_labels(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return


def gridlines(ax: plt.Axes, **kwargs: Any) -> None:
    ax.grid(linestyle="-.", color="gray", alpha=0.5, **kwargs)
    return


# ---------------------------------------------------------------------------
