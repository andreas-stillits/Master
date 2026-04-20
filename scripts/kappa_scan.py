from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mscthesis.config.declaration import ProjectConfig
from mscthesis.utilities.paths import ProjectPaths
from mscthesis.core.io import load_volumetric_mesh
from mscthesis.core.solvers import UniformSolver, UniformSolverConfig
from mscthesis.utilities.plotting import use_style, figure, save

kappas = np.logspace(3, 10, 8)
sample_id = "00019"


def envelope(
    x: np.ndarray,
    kappa: float,
    epsilon: float = 0.02,
    stomatal_aspect: float = 0.20,
    plug_aspect: float = 0.25,
) -> np.ndarray:
    phi = (x**2 - stomatal_aspect**2) / epsilon / stomatal_aspect**2
    return 0.5 * (1 - np.tanh(phi)) * kappa


def envelope_plot() -> None:
    use_style()
    fig, ax = figure(size="single")
    x = np.linspace(-0.25, 0.25, 500)
    for kappa in kappas:
        y = envelope(x, kappa)
        ax.plot(x, y, label=f"$\\kappa={kappa:.0e}$")
    ax.set_yscale("log")
    ax.legend()
    # set tick marks every 0.02 on the z axis
    # ax.set_xticks(np.arange(-0.25, 0.25 + 0.02, 0.04))
    ax.set_xticks(
        [-0.25, -0.20, -0.15, -0.10, -0.02, 0.0, 0.02, 0.10, 0.15, 0.20, 0.25]
    )
    plt.show()


def run_sim() -> None:
    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root)
    mesh_ctx = load_volumetric_mesh(paths.sample(sample_id).meshing().require_mesh())

    ans_stomatal_flux_equiv = []
    ans_stomatal_flux_direct = []
    ans_mesophyll_flux_equiv = []

    for kappa in kappas:
        solver_config = UniformSolverConfig(
            stomatal_aspect=0.08,
            stomatal_epsilon=0.02,
            kappa=kappa,
            ksp_type="cg",
            ksp_rtol=1e-8,
            pc_type="jacobi",
            quad_degree=4,
            order=2,
        )
        solver = UniformSolver(solver_config, mesh_ctx)
        solution, analysis = solver.solve_for(0.75, 1.0, 0.1)
        ans_stomatal_flux_equiv.append(np.abs(float(analysis["stomatal_flux_equiv"])))
        ans_stomatal_flux_direct.append(np.abs(float(analysis["stomatal_flux_direct"])))
        ans_mesophyll_flux_equiv.append(np.abs(float(analysis["mesophyll_flux_equiv"])))
        print(
            f"Kappa: {kappa:.0e}, stomatal equiv: {ans_stomatal_flux_equiv[-1]:.6f}; stomatal direct: {ans_stomatal_flux_direct[-1]:.6f}; mesophyll equiv: {ans_mesophyll_flux_equiv[-1]:.6f}"
        )

    use_style()
    fig, ax = figure(size="single")
    ax.plot(kappas, ans_stomatal_flux_equiv, marker="o", linestyle="--")
    ax.plot(kappas, ans_stomatal_flux_direct, marker="x", linestyle="--")
    ax.plot(kappas, ans_mesophyll_flux_equiv, marker="^", linestyle="--")
    ax.set_xscale("log")
    plt.show()


def main() -> int:

    # envelope_plot()
    run_sim()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
