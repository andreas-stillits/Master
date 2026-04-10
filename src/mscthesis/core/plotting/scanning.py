from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...utilities.plotting import (
    gridlines,
    label_panel,
    panel_grid,
    save,
    set_axis_labels,
    use_style,
)


def plot_scanning_results(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> None: ...
