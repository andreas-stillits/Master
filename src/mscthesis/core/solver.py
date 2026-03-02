from __future__ import annotations

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


def _get_stomatal_envelope(
    mesh: Mesh, plug_aspect: float, stomatal_aspect: float, stomatal_epsilon: float
) -> Any:
    x = ufl.SpatialCoordinate(mesh)
    phi = x[0] ** 2 + x[1] ** 2 - stomatal_aspect**2  # type: ignore[reportIndexIssue]
    envolope = 0.5 * (1 - ufl.tanh(phi / stomatal_epsilon / plug_aspect**2))
    return envolope


def _get_measures(mesh: Mesh, cell_tags: Any, facet_tags: Any) -> tuple[Any, Any]:
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    return dx, ds


def analyze_solution(
    params: tuple[float, float, float],
    solution: fem.Function,
    mesh: Mesh,
    cell_tags: Any,
    facet_tags: Any,
    plug_aspect: float,
    stomatal_aspect: float,
    stomatal_epsilon: float,
) -> tuple[float, float]:
    """
    Analyze the solution by computing derived quantities of interest
    Args:
        params (tuple[float, float, float]): The parameters used in the simulation (absorption, transport, compensation)
        solution (fem.Function): The computed solution to analyze
        mesh (Mesh): The mesh on which the solution is defined
        plug_aspect (float): The aspect ratio of the plug
        stomatal_aspect (float): The aspect ratio of the stomata
        stomatal_epsilon (float): The smoothing parameter for the stomatal envelope
    Returns:
        tuple[float, float]: The computed stomatal and mesophyll fluxes
    """
    # compute the assimilatio rate in two ways: stomatal and mesophyll fluxes

    absorption, transport, compensation = params
    dx, ds = _get_measures(mesh, cell_tags, facet_tags)

    # stomatal flux
    envelope = _get_stomatal_envelope(
        mesh, plug_aspect, stomatal_aspect, stomatal_epsilon
    )
    stomatal_flux = fem.assemble_scalar(
        fem.form(transport * envelope * (1 - solution) * ds(BOTTOM_TAG))
    )

    # mesophyll flux
    mesophyll_flux = fem.assemble_scalar(
        fem.form(absorption * (solution - compensation) * ds(MESOPHYLL_TAG))
    )

    return float(stomatal_flux), float(mesophyll_flux)


class UniformSolver:
    def __init__(
        self,
        params: tuple[float, float, float],
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
        self.absorption = params[0]
        self.transport = params[1]
        self.compensation = params[2]
        self.plug_aspect = plug_aspect
        self.stomatal_aspect = stomatal_aspect
        self.stomatal_epsilon = stomatal_epsilon
        self.stomatal_area_fraction = stomatal_area_fraction
        self.mesophyll_area_fraction = mesophyll_area_fraction
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.order = order

    def solve(self) -> fem.Function:

        functionspace = fem.functionspace(self.mesh, ("Lagrange", self.order))
        dx, ds = _get_measures(self.mesh, self.cell_tags, self.facet_tags)

        compensation = fem.Constant(self.mesh, default_scalar_type(self.compensation))
        surface_coeff = fem.Constant(
            self.mesh,
            default_scalar_type(self.absorption / self.mesophyll_area_fraction),
        )
        stomatal_coeff = fem.Constant(
            self.mesh, default_scalar_type(self.transport / self.stomatal_area_fraction)
        )

        chi = ufl.TrialFunction(functionspace)
        v = ufl.TestFunction(functionspace)

        envelope = _get_stomatal_envelope(
            self.mesh, self.plug_aspect, self.stomatal_aspect, self.stomatal_epsilon
        )

        # Weak form
        a = (
            ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx(AIRSPACE_TAG)
            + surface_coeff * chi * v * ds(MESOPHYLL_TAG)
            + stomatal_coeff * envelope * chi * v * ds(BOTTOM_TAG)
        )

        L = surface_coeff * compensation * v * ds(
            MESOPHYLL_TAG
        ) + stomatal_coeff * envelope * v * ds(BOTTOM_TAG)

        problem = LinearProblem(
            a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        solution = problem.solve()

        return solution
