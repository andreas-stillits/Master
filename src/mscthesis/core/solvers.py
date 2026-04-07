from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh


@dataclass
class Tags:
    AIRSPACE: int = 1
    TOP: int = 2
    BOTTOM: int = 3
    CURVED: int = 4
    MESOPHYLL: int = 5


@dataclass
class MeshContext:
    mesh: Mesh
    cell_tags: Any
    facet_tags: Any


@dataclass
class SolverConfig:
    stomatal_aspect: float
    stomatal_epsilon: float
    ksp_rtol: float
    order: int


class BaseSolver:
    def __init__(self, solver_config: SolverConfig, mesh_ctx: MeshContext) -> None:
        # solver
        self.stomatal_aspect = solver_config.stomatal_aspect
        self.stomatal_epsilon = solver_config.stomatal_epsilon
        self.ksp_rtol = solver_config.ksp_rtol
        self.order = solver_config.order
        # mesh
        self.mesh = mesh_ctx.mesh
        self.cell_tags = mesh_ctx.cell_tags
        self.facet_tags = mesh_ctx.facet_tags
        # function space and measures
        self.functionspace = fem.functionspace(self.mesh, ("CG", self.order))
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        # coefficients
        self.compensation = fem.Constant(self.mesh, default_scalar_type(0.0))
        self.surface_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))
        self.stomatal_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))
        # dimensions and area fractions
        self.tags = Tags()
        # airspace
        self.airspace_volume = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.dx(self.tags.AIRSPACE))  # type: ignore[reportArgumentType]
            )
        )
        # plug
        self.plug_area = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.ds(self.tags.BOTTOM))  # type: ignore[reportArgumentType]
            )
        )
        self.plug_aspect = np.sqrt(self.plug_area / np.pi)  # radius of cylindrical plug
        # mesophyll
        self.mesophyll_area = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
            )
        )
        self.mesophyll_area_fraction = self.mesophyll_area / self.plug_area
        # stomatal
        x = ufl.SpatialCoordinate(self.mesh)
        phi = x[0] ** 2 + x[1] ** 2 - self.stomatal_aspect**2  # type: ignore[reportIndexIssue]
        self.envelope = 0.5 * (
            1 - ufl.tanh(phi / self.stomatal_epsilon / self.plug_aspect**2)
        )
        self.stomatal_area = float(
            fem.assemble_scalar(
                fem.form(self.envelope * self.ds(self.tags.BOTTOM))  # type: ignore[reportArgumentType]
            )
        )
        self.stomatal_area_fraction = self.stomatal_area / self.plug_area
        return

    def analyze(self, solution: fem.Function) -> dict[str, Any]:
        # FLUXES
        # stomatal
        stomatal_flux_direct = float(
            fem.assemble_scalar(
                fem.form(
                    ufl.dot(ufl.grad(solution), ufl.FacetNormal(self.mesh))
                    * self.ds(self.tags.BOTTOM)
                )  # type: ignore[reportArgumentType]
            )
        )
        stomatal_flux_equiv = float(
            fem.assemble_scalar(
                fem.form(
                    self.stomatal_coeff
                    * self.envelope
                    * (1 - solution)
                    * self.ds(self.tags.BOTTOM)  # type: ignore[reportArgumentType]
                )
            )
        )
        # mesophyll
        mesophyll_flux_direct = float(
            fem.assemble_scalar(
                fem.form(ufl.dot(ufl.grad(solution), ufl.FacetNormal(self.mesh)) * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
            )
        )
        mesophyll_flux_equiv = float(
            fem.assemble_scalar(
                fem.form(self.surface_coeff * (solution - self.compensation) * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
            )
        )
        # curved surface flux
        curved_flux_direct = float(
            fem.assemble_scalar(
                fem.form(ufl.dot(ufl.grad(solution), ufl.FacetNormal(self.mesh)) * self.ds(self.tags.CURVED))  # type: ignore[reportArgumentType]
            )
        )
        # top surface flux
        top_flux_direct = float(
            fem.assemble_scalar(
                fem.form(ufl.dot(ufl.grad(solution), ufl.FacetNormal(self.mesh)) * self.ds(self.tags.TOP))  # type: ignore[reportArgumentType]
            )
        )
        # ---------------------------------------------------------------------------
        # CO2 CONCENTRATIONS

        # substomatal concentration
        substomatal_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution * self.envelope * self.ds(self.tags.BOTTOM))  # type: ignore[reportArgumentType]
                )
            )
            / self.stomatal_area
        )

        # top surface concentration
        top_concentration = float(
            fem.assemble_scalar(
                fem.form(solution * self.ds(self.tags.TOP))  # type: ignore[reportArgumentType]
            )
            / self.plug_area
        )

        # mean airspace concentration
        airspace_mean_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution * self.dx(self.tags.AIRSPACE))  # type: ignore[reportArgumentType]
                )
            )
            / self.airspace_volume
        )

        # mean square airspace concentration
        airspace_mean_square_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution**2 * self.dx(self.tags.AIRSPACE))  # type: ignore[reportArgumentType]
                )
            )
            / self.airspace_volume
        )

        # mean cell surface concentration
        surface_mean_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
                )
            )
            / self.mesophyll_area
            if self.mesophyll_area > 0.0
            else 0.0
        )

        # mean square cell surface concentration
        surface_mean_square_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution**2 * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
                )
            )
            / self.mesophyll_area
            if self.mesophyll_area > 0.0
            else 0.0
        )

        # Calculate square-root of variances
        airspace_variance = airspace_mean_square_conc - airspace_mean_conc**2
        surface_variance = surface_mean_square_conc - surface_mean_conc**2

        return {
            "stomatal_flux_direct": stomatal_flux_direct,
            "stomatal_flux_equiv": stomatal_flux_equiv,
            "mesophyll_flux_direct": mesophyll_flux_direct,
            "mesophyll_flux_equiv": mesophyll_flux_equiv,
            "curved_flux_direct": curved_flux_direct,
            "top_flux_direct": top_flux_direct,
            "substomatal_mean": substomatal_conc,
            "top_mean": top_concentration,
            "airspace_mean": airspace_mean_conc,
            "airspace_variance": airspace_variance,
            "surface_mean": surface_mean_conc,
            "surface_variance": surface_variance,
            "plug_area": self.plug_area,
            "stomatal_area": self.stomatal_area,
            "mesophyll_area": self.mesophyll_area,
            "stomatal_area_fraction": self.stomatal_area_fraction,
            "mesophyll_area_fraction": self.mesophyll_area_fraction,
            "porosity": self.airspace_volume / self.plug_area,
        }

    def solve_for(
        self,
        *args,
        **kwargs,
    ) -> tuple[fem.Function, dict[str, Any]]:
        raise NotImplementedError("Must be implemented by subclass.")


# ------------------------------------------------------------------------
# UNIFORM SOLVER
# ------------------------------------------------------------------------


@dataclass
class UniformSolverConfig(SolverConfig):
    pass


class UniformSolver(BaseSolver):
    def __init__(
        self, solver_config: UniformSolverConfig, mesh_ctx: MeshContext
    ) -> None:
        super().__init__(solver_config, mesh_ctx)
        #
        chi = ufl.TrialFunction(self.functionspace)
        v = ufl.TestFunction(self.functionspace)

        a = (
            ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(self.tags.AIRSPACE)
            + self.surface_coeff * chi * v * self.ds(self.tags.MESOPHYLL)
            + self.stomatal_coeff * self.envelope * chi * v * self.ds(self.tags.BOTTOM)
        )
        L = self.surface_coeff * self.compensation * v * self.ds(
            self.tags.MESOPHYLL
        ) + self.stomatal_coeff * self.envelope * v * self.ds(self.tags.BOTTOM)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[],
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": self.ksp_rtol,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )

    def solve_for(
        self, absorption: float, transport: float, compensation: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.surface_coeff.value = default_scalar_type(
            absorption / self.mesophyll_area_fraction
        )
        self.stomatal_coeff.value = default_scalar_type(
            transport / self.stomatal_area_fraction
        )
        self.compensation.value = default_scalar_type(compensation)
        solution = self.problem.solve()
        return solution, self.analyze(solution)


# ------------------------------------------------------------------------
# AXIAL SOLVER
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# DIFFUSION SOLVER
# ------------------------------------------------------------------------


@dataclass
class DiffusionSolverConfig(SolverConfig):
    pass


class DiffusionSolver(BaseSolver):
    def __init__(
        self, solver_config: DiffusionSolverConfig, mesh_ctx: MeshContext
    ) -> None:
        super().__init__(solver_config, mesh_ctx)
        #
        chi = ufl.TrialFunction(self.functionspace)
        v = ufl.TestFunction(self.functionspace)

        #
        top_facets = self.facet_tags.find(self.tags.TOP)
        top_dofs = fem.locate_dofs_topological(
            self.functionspace, self.mesh.topology.dim - 1, top_facets
        )

        self.boundary_conc = fem.Constant(self.mesh, default_scalar_type(0.0))
        bc = fem.dirichletbc(self.boundary_conc, top_dofs, self.functionspace)

        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(
            self.tags.AIRSPACE
        ) + self.stomatal_coeff * self.envelope * chi * v * self.ds(self.tags.BOTTOM)
        L = self.stomatal_coeff * self.envelope * v * self.ds(self.tags.BOTTOM)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": self.ksp_rtol,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )

    def solve_for(
        self, transport: float, boundary_conc: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.boundary_conc.value = default_scalar_type(boundary_conc)
        self.stomatal_coeff.value = default_scalar_type(
            transport / self.stomatal_area_fraction
        )
        solution = self.problem.solve()
        return solution, self.analyze(solution)
