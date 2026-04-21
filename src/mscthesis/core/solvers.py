from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, locate_entities_boundary, meshtags, compute_midpoints


@dataclass
class Tags:
    AIRSPACE: int = 1
    TOP: int = 2
    BOTTOM: int = 3
    CURVED: int = 4
    MESOPHYLL: int = 5
    INLET: int = 6


@dataclass
class MeshContext:
    mesh: Mesh
    cell_tags: Any
    facet_tags: Any


@dataclass
class SolverConfig:
    stomatal_aspect: float
    ksp_type: str
    ksp_rtol: float
    pc_type: str
    quad_degree: int
    order: int


class BaseSolver:
    def __init__(self, solver_config: SolverConfig, mesh_ctx: MeshContext) -> None:
        # solver
        self.stomatal_aspect = solver_config.stomatal_aspect
        self.ksp_type = solver_config.ksp_type
        self.ksp_rtol = solver_config.ksp_rtol
        self.pc_type = solver_config.pc_type
        self.quad_degree = solver_config.quad_degree
        self.order = solver_config.order
        # mesh
        self.mesh = mesh_ctx.mesh
        self.cell_tags = mesh_ctx.cell_tags
        facet_tags = mesh_ctx.facet_tags
        fdim = self.mesh.topology.dim - 1
        self.tags = Tags()

        # dirichlet bc for stomatal inlet
        # identify inlet facets based on geometric criterion
        tol = 1e-12

        def inlet_marker(x):
            r2 = x[0] ** 2 + x[1] ** 2
            return np.isclose(x[2], 0.0, atol=tol) & (
                r2 <= self.stomatal_aspect**2 + tol
            )

        # inlet_facets_geo = locate_entities_boundary(self.mesh, fdim, inlet_marker)

        # restrict candidates to come from the BOTTOM tag group
        bottom_facets = facet_tags.indices[facet_tags.values == self.tags.BOTTOM]
        mid = compute_midpoints(self.mesh, fdim, bottom_facets)
        r2_mid = mid[:, 0] ** 2 + mid[:, 1] ** 2
        mask = np.isclose(mid[:, 2], 0.0, atol=tol) & (
            r2_mid <= self.stomatal_aspect**2 + tol
        )
        inlet_facets = bottom_facets[mask]

        # map tags by facet index
        tag_map = {
            int(f): int(v)
            for f, v in zip(facet_tags.indices, facet_tags.values, strict=True)
        }
        # overwrite with the new INLET tag
        for f in inlet_facets:
            tag_map[int(f)] = self.tags.INLET

        # rebuild arrays
        updated_facets = np.array(sorted(tag_map.keys()), dtype=np.int32)
        updated_values = np.array(
            [tag_map[int(f)] for f in updated_facets], dtype=np.int32
        )

        self.facet_tags = meshtags(self.mesh, fdim, updated_facets, updated_values)

        # function space and measures
        self.functionspace = fem.functionspace(self.mesh, ("CG", self.order))
        self.inlet_dofs = fem.locate_dofs_topological(
            self.functionspace, fdim, inlet_facets
        )

        self.dx = ufl.Measure(
            "dx",
            domain=self.mesh,
            subdomain_data=self.cell_tags,
            metadata={"quadrature_degree": self.quad_degree},
        )
        self.ds = ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facet_tags,
            metadata={"quadrature_degree": self.quad_degree},
        )
        # coefficients
        self.compensation = fem.Constant(self.mesh, default_scalar_type(0.0))
        self.surface_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))
        self.chii = fem.Constant(self.mesh, default_scalar_type(0.0))
        # dimensions and area fractions
        # airspace
        self.airspace_volume = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.dx(self.tags.AIRSPACE))  # type: ignore[reportArgumentType]
            )
        )
        # plug
        self.plug_area = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.ds(self.tags.TOP))  # type: ignore[reportArgumentType]
            )
        )
        self.plug_aspect = np.sqrt(self.plug_area / np.pi)  # radius of cylindrical plug
        # curved
        self.curved_area = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.ds(self.tags.CURVED))  # type: ignore[reportArgumentType]
            )
        )
        # mesophyll
        self.mesophyll_area = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
            )
        )
        self.mesophyll_area_fraction = self.mesophyll_area / self.plug_area
        self.stomatal_area = float(
            fem.assemble_scalar(
                fem.form(1.0 * self.ds(self.tags.INLET))  # type: ignore[reportArgumentType]
            )
        )
        self.stomatal_area_fraction = self.stomatal_area / self.plug_area
        return

    def analyze(self, solution: fem.Function, gradient: fem.Function) -> dict[str, Any]:
        # FLUXES
        normal = ufl.FacetNormal(self.mesh)
        # stomatal
        stomatal_flux_direct = float(
            fem.assemble_scalar(
                fem.form(
                    ufl.dot(gradient, normal) * self.ds(self.tags.INLET)
                )  # type: ignore[reportArgumentType]
            )
        )
        # bottom
        bottom_flux_direct = float(
            fem.assemble_scalar(
                fem.form(ufl.dot(gradient, normal) * self.ds(self.tags.BOTTOM))
            )
        )

        # mesophyll
        mesophyll_flux_direct = float(
            fem.assemble_scalar(
                fem.form(ufl.dot(gradient, normal) * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
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
                fem.form(ufl.dot(gradient, normal) * self.ds(self.tags.CURVED))  # type: ignore[reportArgumentType]
            )
        )
        # top surface flux
        top_flux_direct = float(
            fem.assemble_scalar(
                fem.form(ufl.dot(gradient, normal) * self.ds(self.tags.TOP))  # type: ignore[reportArgumentType]
            )
        )
        # total flux balance
        total_flux_direct = float(
            fem.assemble_scalar(fem.form(ufl.dot(gradient, normal) * self.ds))  # type: ignore[reportArgumentType]
        )
        # ---------------------------------------------------------------------------
        # CO2 CONCENTRATIONS

        # substomatal concentration
        substomatal_conc = float(
            fem.assemble_scalar(
                fem.form(solution * self.ds(self.tags.INLET))  # type: ignore[reportArgumentType]
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
        airspace_variance = (
            float(
                fem.assemble_scalar(
                    fem.form((solution - airspace_mean_conc) ** 2 * self.dx(self.tags.AIRSPACE))  # type: ignore[reportArgumentType]
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
        surface_variance = (
            float(
                fem.assemble_scalar(
                    fem.form((solution - surface_mean_conc) ** 2 * self.ds(self.tags.MESOPHYLL))  # type: ignore[reportArgumentType]
                )
            )
            / self.mesophyll_area
            if self.mesophyll_area > 0.0
            else 0.0
        )

        # stomatal conductance
        tol = 1e-6
        transport = (
            np.abs(mesophyll_flux_equiv / (1.0 - substomatal_conc) / self.plug_area)
            if substomatal_conc < 1.0 - tol
            else None
        )

        return {
            "stomatal_flux_direct": stomatal_flux_direct,
            "bottom_flux_direct": bottom_flux_direct,
            "mesophyll_flux_direct": mesophyll_flux_direct,
            "mesophyll_flux_equiv": mesophyll_flux_equiv,
            "curved_flux_direct": curved_flux_direct,
            "top_flux_direct": top_flux_direct,
            "total_flux_direct": total_flux_direct,
            "substomatal_mean": substomatal_conc,
            "top_mean": top_concentration,
            "airspace_mean": airspace_mean_conc,
            "airspace_variance": airspace_variance,
            "surface_mean": surface_mean_conc,
            "surface_variance": surface_variance,
            "plug_area": self.plug_area,
            "curved_area": self.curved_area,
            "stomatal_area": self.stomatal_area,
            "mesophyll_area": self.mesophyll_area,
            "stomatal_area_fraction": self.stomatal_area_fraction,
            "mesophyll_area_fraction": self.mesophyll_area_fraction,
            "porosity": self.airspace_volume / self.plug_area,
            "transport": transport,
        }

    def solve_for(
        self,
        *args,
        **kwargs,
    ) -> tuple[fem.Function, dict[str, Any]]:
        raise NotImplementedError("Must be implemented by subclass.")

    def compute_gradient(
        self,
        solution: fem.Function,
    ) -> fem.Function:
        grad_order = max(self.order - 1, 0)  # Ensure order is at least 0
        gradientspace = fem.functionspace(
            self.mesh, ("DG", grad_order, (self.mesh.geometry.dim,))
        )
        d_chi = ufl.TrialFunction(gradientspace)
        v = ufl.TestFunction(gradientspace)
        a_proj = ufl.inner(d_chi, v) * self.dx
        L_proj = ufl.inner(ufl.grad(solution), v) * self.dx
        projection_problem = LinearProblem(
            a_proj,
            L_proj,
            bcs=[],
            petsc_options={
                "ksp_type": self.ksp_type,
                "ksp_rtol": self.ksp_rtol,
                "pc_type": self.pc_type,
            },
        )
        return projection_problem.solve()


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

        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(
            self.tags.AIRSPACE
        ) + self.surface_coeff * chi * v * self.ds(self.tags.MESOPHYLL)
        L = self.surface_coeff * self.compensation * v * self.ds(self.tags.MESOPHYLL)

        bc_inlet = fem.dirichletbc(self.chii, self.inlet_dofs, self.functionspace)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[bc_inlet],
            petsc_options={
                "ksp_type": self.ksp_type,
                "ksp_rtol": self.ksp_rtol,
                "pc_type": self.pc_type,
            },
        )

    def solve_for(
        self, chii: float, absorption: float, compensation: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.surface_coeff.value = default_scalar_type(
            absorption / self.mesophyll_area_fraction
        )
        self.chii.value = default_scalar_type(chii)
        self.compensation.value = default_scalar_type(compensation)
        solution = self.problem.solve()
        gradient = self.compute_gradient(solution)
        return solution, self.analyze(solution, gradient)


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

        self.chi_top = fem.Constant(self.mesh, default_scalar_type(0.0))
        bc_top = fem.dirichletbc(self.chi_top, top_dofs, self.functionspace)
        bc_inlet = fem.dirichletbc(self.chii, self.inlet_dofs, self.functionspace)

        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(self.tags.AIRSPACE)
        L = 0.0

        self.problem = LinearProblem(
            a,
            L,
            bcs=[bc_top, bc_inlet],
            petsc_options={
                "ksp_type": self.ksp_type,
                "ksp_rtol": self.ksp_rtol,
                "pc_type": self.pc_type,
            },
        )

    def solve_for(
        self, chii: float, chi_top: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.chii.value = default_scalar_type(chii)
        self.chi_top.value = default_scalar_type(chi_top)
        solution = self.problem.solve()
        gradient = self.compute_gradient(solution)
        return solution, self.analyze(solution, gradient)
