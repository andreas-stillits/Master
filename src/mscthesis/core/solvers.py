from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh

AIRSPACE_TAG = 1
TOP_TAG = 2
BOTTOM_TAG = 3
CURVED_TAG = 4
MESOPHYLL_TAG = 5


@dataclass
class MeshContext:
    mesh: Mesh
    cell_tags: Any
    facet_tags: Any


@dataclass
class SolverConfig:
    compensation: float
    plug_aspect: float
    stomatal_aspect: float
    stomatal_epsilon: float
    stomatal_area_fraction: float
    mesophyll_area_fraction: float
    order: int


class BaseSolver:
    def __init__(self, solver_config: SolverConfig, mesh_ctx: MeshContext) -> None:
        # solver
        self.compensation = solver_config.compensation
        self.plug_aspect = solver_config.plug_aspect
        self.stomatal_aspect = solver_config.stomatal_aspect
        self.stomatal_epsilon = solver_config.stomatal_epsilon
        self.stomatal_area_fraction = solver_config.stomatal_area_fraction
        self.mesophyll_area_fraction = solver_config.mesophyll_area_fraction
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
        self.compensation = fem.Constant(
            self.mesh, default_scalar_type(self.compensation)
        )
        self.surface_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))
        self.stomatal_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))
        # stomatal envelope
        x = ufl.SpatialCoordinate(self.mesh)
        phi = x[0] ** 2 + x[1] ** 2 - self.stomatal_aspect**2  # type: ignore[reportIndexIssue]
        self.envelope = 0.5 * (
            1 - ufl.tanh(phi / self.stomatal_epsilon / self.plug_aspect**2)
        )
        return

    def analyze(self, solution: fem.Function) -> dict[str, Any]:
        # stomatal flux
        stomatal_flux = fem.assemble_scalar(
            fem.form(
                self.stomatal_coeff
                * self.envelope
                * (1 - solution)
                * self.ds(BOTTOM_TAG)
            )  # type: ignore[reportArgumentType]
        )
        stomatal_flux = float(stomatal_flux)

        # mesophyll flux
        mesophyll_flux = fem.assemble_scalar(
            fem.form(
                self.surface_coeff
                * (solution - self.compensation)
                * self.ds(MESOPHYLL_TAG)
            )  # type: ignore[reportArgumentType]
        )
        mesophyll_flux = float(mesophyll_flux)

        # -------------------------------------------------------------------------

        # stomatal surface area
        stomatal_surface_area = float(
            fem.assemble_scalar(
                fem.form(self.envelope * self.ds(BOTTOM_TAG))  # type: ignore[reportArgumentType]
            )
        )

        mesophyll_surface_area = float(
            fem.assemble_scalar(
                fem.form(1 * self.ds(MESOPHYLL_TAG))  # type: ignore[reportArgumentType]
            )
        )

        airspace_volume = float(
            fem.assemble_scalar(
                fem.form(1 * self.dx(AIRSPACE_TAG))  # type: ignore[reportArgumentType]
            )
        )
        # --------------------------------------------------------------------------

        # substomatal concentration
        substomatal_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution * self.envelope * self.ds(BOTTOM_TAG))  # type: ignore[reportArgumentType]
                )
            )
            / stomatal_surface_area
        )

        # mean airspace concentration
        airspace_mean_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution * self.dx(AIRSPACE_TAG))  # type: ignore[reportArgumentType]
                )
            )
            / airspace_volume
        )

        # mean square airspace concentration
        airspace_mean_square_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution**2 * self.dx(AIRSPACE_TAG))  # type: ignore[reportArgumentType]
                )
            )
            / airspace_volume
        )

        # mean cell surface concentration
        surface_mean_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution * self.ds(MESOPHYLL_TAG))  # type: ignore[reportArgumentType]
                )
            )
            / mesophyll_surface_area
        )

        # mean square cell surface concentration
        surface_mean_square_conc = (
            float(
                fem.assemble_scalar(
                    fem.form(solution**2 * self.ds(MESOPHYLL_TAG))  # type: ignore[reportArgumentType]
                )
            )
            / mesophyll_surface_area
        )

        # Calculate square-root of variances
        airspace_variance = airspace_mean_square_conc - airspace_mean_conc**2
        surface_variance = surface_mean_square_conc - surface_mean_conc**2

        return {
            "stomatal_flux": stomatal_flux,
            "mesophyll_flux": mesophyll_flux,
            "substomatal_mean": substomatal_conc,
            "airspace_mean": airspace_mean_conc,
            "airspace_variance": airspace_variance,
            "surface_mean": surface_mean_conc,
            "surface_variance": surface_variance,
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
            ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(AIRSPACE_TAG)
            + self.surface_coeff * chi * v * self.ds(MESOPHYLL_TAG)
            + self.stomatal_coeff * self.envelope * chi * v * self.ds(BOTTOM_TAG)
        )
        L = self.surface_coeff * self.compensation * v * self.ds(
            MESOPHYLL_TAG
        ) + self.stomatal_coeff * self.envelope * v * self.ds(BOTTOM_TAG)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[],
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": 1e-8,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )

    def solve_for(
        self, absorption: float, transport: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.surface_coeff.value = default_scalar_type(
            absorption / self.mesophyll_area_fraction
        )
        self.stomatal_coeff.value = default_scalar_type(
            transport / self.stomatal_area_fraction
        )
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
        top_facets = self.facet_tags.find(TOP_TAG)
        top_dofs = fem.locate_dofs_topological(
            self.functionspace, self.mesh.topology.dim - 1, top_facets
        )

        self.boundary_conc = fem.Constant(self.mesh, default_scalar_type(0.0))
        bc = fem.dirichletbc(self.boundary_conc, top_dofs, self.functionspace)

        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(
            AIRSPACE_TAG
        ) + self.stomatal_coeff * self.envelope * chi * v * self.ds(BOTTOM_TAG)
        L = self.stomatal_coeff * self.envelope * v * self.ds(BOTTOM_TAG)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": 1e-8,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )

    def solve_for(
        self, transport: float, boundary_conc: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.boundary_conc.value = default_scalar_type(boundary_conc)
        self.stomatal_coeff.value = default_scalar_type(transport)
        solution = self.problem.solve()
        return solution, self.analyze(solution)
