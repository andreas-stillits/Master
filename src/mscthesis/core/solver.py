from __future__ import annotations

from pathlib import Path
from typing import Any

import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh

from mscthesis.config.declaration import ProjectConfig

from ..core.io import load_volumetric_mesh, save_fem_solution
from ..utilities.fetching import fetch_manifest_quantity

AIRSPACE_TAG = 1
TOP_TAG = 2
BOTTOM_TAG = 3
CURVED_TAG = 4
MESOPHYLL_TAG = 5


def _get_stomatal_envelope(
    mesh: Mesh, plug_aspect: float, stomatal_aspect: float, stomatal_epsilon: float
) -> Any:
    x = ufl.SpatialCoordinate(mesh)
    phi = x[0] ** 2 + x[1] ** 2 - stomatal_aspect**2  # type: ignore[reportIndexIssue]
    envelope = 0.5 * (1 - ufl.tanh(phi / stomatal_epsilon / plug_aspect**2))
    return envelope


class UniformSolver:
    def __init__(
        self,
        compensation: float,
        plug_aspect: float,
        stomatal_aspect: float,
        stomatal_epsilon: float,
        stomatal_area_fraction: float,
        mesophyll_area_fraction: float,
        mesh: Mesh,
        cell_tags: Any,
        facet_tags: Any,
        order: int,
    ) -> None:
        # adopt variables
        self.compensation = compensation
        self.plug_aspect = plug_aspect
        self.stomatal_aspect = stomatal_aspect
        self.stomatal_epsilon = stomatal_epsilon
        self.stomatal_area_fraction = stomatal_area_fraction
        self.mesophyll_area_fraction = mesophyll_area_fraction
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.order = order
        # initialize objects
        self.functionspace = fem.functionspace(self.mesh, ("CG", self.order))
        self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

        self.compensation = fem.Constant(
            self.mesh, default_scalar_type(self.compensation)
        )
        self.surface_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))
        self.stomatal_coeff = fem.Constant(self.mesh, default_scalar_type(0.0))

        self.envelope = _get_stomatal_envelope(
            mesh, plug_aspect, stomatal_aspect, stomatal_epsilon
        )

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

    def analyze(self, solution: fem.Function) -> dict[str, Any]:
        # stomatal flux
        stomatal_flux = fem.assemble_scalar(
            fem.form(
                self.stomatal_coeff
                * self.envelope
                * (1 - solution)
                * self.ds(BOTTOM_TAG)
            )
        )
        # mesophyll flux
        mesophyll_flux = fem.assemble_scalar(
            fem.form(
                self.surface_coeff
                * (solution - self.compensation)
                * self.ds(MESOPHYLL_TAG)
            )
        )
        return {
            "stomatal_flux": float(stomatal_flux),
            "mesophyll_flux": float(mesophyll_flux),
        }

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


def single_uniform_solution(
    config: ProjectConfig,
    input_path: str | Path,
    output_path: str | Path,
    sample_id: str,
    compensation: float,
    absorption: float,
    transport: float,
    stomatal_aspect: float,
    stomatal_epsilon: float,
    no_save: bool,
) -> dict[str, Any]:
    """
    Determine the uniform solution for a given set of parameters and sample id
    Args:
        input_path: path to the input mesh file
        output_path: path to the output solution file
        sample_id: sample id for metadata purposes
        compensation: boundary condition value for the mesophyll flux
        absorption: absorption balance parameter
        transport: transport balance parameter
        stomatal_aspect: aspect ratio of the stomatal pore
        stomatal_epsilon: smoothing parameter for the stomatal envelope function
    Returns:
        metadata: dictionary containing relevant metadata about the solution
    """
    mesh, cell_tags, facet_tags = load_volumetric_mesh(input_path)
    data = fetch_manifest_quantity(
        config, sample_id, "meshing", "plug_aspect", "mesophyll_area_fraction"
    )
    plug_aspect = data["plug_aspect"]
    mesophyll_area_fraction = data["mesophyll_area_fraction"]
    stomatal_area_fraction = (stomatal_aspect**2) / (plug_aspect**2)
    solver = UniformSolver(
        compensation,
        plug_aspect,
        stomatal_aspect,
        stomatal_epsilon,
        stomatal_area_fraction,
        mesophyll_area_fraction,
        mesh,
        cell_tags,
        facet_tags,
        order=1,
    )

    solution, analysis = solver.solve_for(absorption, transport)

    if not no_save:
        save_fem_solution(solution, mesh, cell_tags, facet_tags, output_path)

    return {
        "compensation": compensation,
        "absorption": absorption,
        "transport": transport,
        "stomatal_aspect": stomatal_aspect,
        "stomatal_epsilon": stomatal_epsilon,
        "stomatal_flux": analysis["stomatal_flux"],
        "mesophyll_flux": analysis["mesophyll_flux"],
    }
