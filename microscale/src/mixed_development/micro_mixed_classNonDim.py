"""
Shifted periodic BCs with Reynolds
Penalty cavitation
Point constraint

Modified: add immersed / masked Reynolds physics based on film thickness H.

Key idea:
- Build a smooth (or sharp) mask m(H) ~ 1 in fluid region (H > hmin) and ~ eps_solid in solid/contact region (H < hmin)
- Multiply mobility K and advection flux F_adv by m(H) so the PDE is effectively "turned off" in solid regions
- Optionally also weight cavitation penalty by the same mask so cavitation is only enforced where fluid exists

- Need a check that the contact area doesn't extend beyond the boundary of the cell.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from dolfin import *  # legacy FEniCS
parameters["form_compiler"]["quadrature_degree"] = 4
# -------------------------------------------------------------------
# Logging / utilities
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="simulation_log.log",
    filemode="w",
)
logger = logging.getLogger(__name__)


@contextmanager
def capture_output(stream_name: str = "stdout"):
    """Context manager that captures spam from Newton solver."""
    old_stream = getattr(sys, stream_name)
    buf = io.StringIO()
    setattr(sys, stream_name, buf)
    try:
        yield buf
    finally:
        setattr(sys, stream_name, old_stream)


# -------------------------------------------------------------------
# Data containers
# -------------------------------------------------------------------
@dataclass(frozen=True)
class MicroPhysicalParameters:
    Ux: float
    Uy: float
    eta0: float
    rho0: float
    penalty_gamma: float
    xmax: float
    ymax: float
    p0: float
    dpdx: float
    dpdy: float
    H0: float
    h_min: float
    HT: float
    PT: float
    Tend: float
    k_spring: float

    @property
    def U(self):
        # Account for incoming velocity being 1/2 already (your original comment)
        return as_vector((Constant(2.0 * self.Ux), Constant(2.0 * self.Uy)))

    @property
    def dpx_celljump(self) -> float:
        return self.dpdx * self.xmax

    @property
    def dpy_celljump(self) -> float:
        return self.dpdy * self.ymax


@dataclass(frozen=True)
class MicroSolverSettings:
    relaxation_parameter: float
    max_iterations: int
    abs_tolerance: float
    rel_tolerance: float
    delta_h: float       # transition width for smoothing contact mask (0 => sharp)
    eps_solid: float     # small mobility in contact to avoid singularity
    eps_smooth: float    # smoothing for smooth_pos
    print_progress: bool


# -------------------------------------------------------------------
# Periodic BC
# -------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    def inside(self, X, on_boundary):
        return on_boundary and (
            (near(X[0], 0.0) or near(X[1], 0.0))
            and not (
                (near(X[0], 1.0) and near(X[1], 0.0))
                or (near(X[0], 0.0) and near(X[1], 1.0))
                or (near(X[0], 1.0) and near(X[1], 1.0))
            )
        )

    def map(self, X, Y):
        if near(X[0], 1.0) and near(X[1], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1] - 1.0
        elif near(X[0], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1]
        else:
            Y[0] = X[0]
            Y[1] = X[1] - 1.0


# -------------------------------------------------------------------
# Solver
# -------------------------------------------------------------------
class MicroMixedSolver:
    """
    Best-practice refactor notes:
    - __init__ only stores long-lived state and calls setup helpers
    - heavy actions (solve, file export) are explicit methods
    - self.* used only for cross-method state
    """

    def __init__(
        self,
        mesh_m: Mesh,
        mesh_h: Mesh,
        physical_params: MicroPhysicalParameters,
        solver_settings: MicroSolverSettings,
        k: int,
        ht: Callable,  # ht(x, y, xmax, ymax, H0) -> UFL expression
        export_vtk: bool = False,
        output_dir: str = "micro_outputs",
        auto_solve: bool = True,
    ) -> None:
        # ---- persistent inputs (legitimate self.*) ----
        self.mesh_m = mesh_m
        self.mesh_h = mesh_h
        self.params = physical_params
        self.settings = solver_settings
        self.k = k
        self.ht_cb = ht
        self.export_vtk = export_vtk
        self.output_dir = output_dir

        # ---- setup (delegated) ----
        self._setup_export()
        self._setup_coordinates()
        self._nondimensionalise()
        self._setup_film_and_mask()
        self._setup_spaces_and_unknowns()
        self._setup_pressure_expressions()
        self._setup_forms()
        self._setup_point_constraint_bc()

        if auto_solve:
            self.solve()
            self.post_process()

    # -------------------------
    # Setup helpers
    # -------------------------
    def _setup_export(self) -> None:
        if self.export_vtk:
            os.makedirs(self.output_dir, exist_ok=True)
            self._file_times: list[tuple[str, float]] = []

    def _setup_coordinates(self) -> None:
        self.x, self.y = SpatialCoordinate(self.mesh_m)

    def _nondimensionalise(self) -> None:
        """
        - mesh coordinates are nondimensional in [0,1]x[0,1]
        - params/settings inputs are dimensional
        """
        p = self.params
        s = self.settings

        # Physical cell sizes
        self.Lx = float(p.xmax)
        self.Ly = float(p.ymax)

        # Velocity scale
        if p.Ux == 0.0:
            self.Uref = 1.0
        else:
            self.Uref = float(abs(p.Ux))

        # --- Pressure scale (robust max over corners + optional time ramp) ---
        dpx = float(p.dpdx * p.xmax)
        dpy = float(p.dpdy * p.ymax)

        # transient ramp extremes (if used)
        dpt = float(p.Tend * p.PT) if (p.Tend is not None and p.PT is not None) else 0.0

        corner_vals_t0 = [
            p.p0,
            p.p0 + dpx,
            p.p0 + dpy,
            p.p0 + dpx + dpy,
        ]
        corner_vals_t1 = [
            p.p0 + dpt,
            p.p0 + dpt + dpx,
            p.p0 + dpt + dpy,
            p.p0 + dpt + dpx + dpy,
        ]

        #Put the lists together and find the maximum
        self.p_scale = max([1.0] + [abs(v) for v in (corner_vals_t0 + corner_vals_t1)])

        # Nondimensional jumps across the unit cell
        self.dpx_nd = (p.dpdx * p.xmax) / self.p_scale
        self.dpy_nd = (p.dpdy * p.ymax) / self.p_scale

        # Nondimensional velocities (scaled by |Ux|)
        self.Ux_nd = p.Ux / self.Uref
        self.Uy_nd = p.Uy / self.Uref

        # Nondimensional film thickness threshold
        self.hmin_nd = p.h_min / p.H0

        # Nondimensional smoothing widths
        self.delta_h_nd = (s.delta_h / p.H0) if float(s.delta_h) != 0.0 else 0.0
        self.epsP_nd = s.eps_smooth / self.p_scale

        # Nondimensional penalty strength (pressure-like penalty)
        self.gamma_nd = p.penalty_gamma / self.p_scale

        # Reynolds nondimensional group (consistent with your NGSolve omega definition)
        # omega = H0^2 * p_scale / (6 * eta0 * Lx * |Ux|)
        self.omega = (p.H0**2 * self.p_scale) / (6.0 * p.eta0 * self.Lx * self.Uref)

        # Helpers to use in UFL
        self.invLx = Constant(1.0 / self.Lx)
        self.invLy = Constant(1.0 / self.Ly)

    def grad_phys(self, q):
        """
        Physical gradient operator when mesh coords are nondimensional:
            x = Lx*x*, y = Ly*y*
        """
        return as_vector((Dx(q, 0) * self.invLx, Dx(q, 1) * self.invLy))

        

    def _setup_film_and_mask(self) -> None:
        p = self.params
        s = self.settings

        # Film thickness: ht_cb returns DIMENSIONAL H(x,y); convert to nondim h = H/H0
        H_dim = self.ht_cb(self.x, self.y, p.xmax, p.ymax, p.H0)
        self.h_nd = H_dim / Constant(p.H0)

        # Level set for contact mask in nondim thickness units
        phi_mask = self.h_nd - Constant(self.hmin_nd)

        # For post-processing only
        self.chi_fluid = conditional(gt(phi_mask, 0.0), Constant(1.0), Constant(0.0))

        # Smooth/sharp mask (delta in nondim thickness units)
        self.mask = self.smooth_levelset_mask(
            phi=phi_mask,
            delta=Constant(self.delta_h_nd),
            mask_core=Constant(s.eps_solid),
            mask_fluid=Constant(1.0),
        )

        # Constant rho/eta (you can swap in DH/Roelands later as nondim ratios)
        rho_nd = Constant(1.0)
        eta_nd = Constant(1.0)

        # Nondimensional velocity vector (note: your p.U multiplies by 2; keep that convention if desired)
        # If you want to preserve the "incoming is half already" logic, apply it BEFORE nondim:
        Ux_eff = 2.0 * self.Ux_nd
        Uy_eff = 2.0 * self.Uy_nd
        U_nd = as_vector((Constant(Ux_eff), Constant(Uy_eff)))

        # Nondimensional mobility and advection flux (masked)
        # K* = rho* omega * (h^3/eta)
        # F_adv* = rho* (U*h)   (matches NGSolve form; factor 1/2 already handled above if desired)
        self.K_nd = rho_nd * Constant(self.omega) * (self.h_nd**3 / eta_nd) * self.mask
        self.F_adv_nd = rho_nd * (U_nd * self.h_nd) * self.mask


    def _setup_spaces_and_unknowns(self) -> None:
        self.pbc = PeriodicBoundary()

        self.Vper = FunctionSpace(self.mesh_m, "CG", 1, constrained_domain=self.pbc)
        self.Vfull = FunctionSpace(self.mesh_m, "CG", 1)  # for exports
        self.Vdg0 = FunctionSpace(self.mesh_m, "DG", 0)    # for sharp indicator export

        self.w = Function(self.Vper, name="w")
        self.v = TestFunction(self.Vper)
        self.dw = TrialFunction(self.Vper)

    def _setup_pressure_expressions(self) -> None:
        # nondimensional affine ramp encoding the cell jump
        dpx_nd = Constant(self.dpx_nd)
        dpy_nd = Constant(self.dpy_nd)

        self.phi_nd = dpx_nd * self.x + dpy_nd * self.y

        # nondimensional pressure
        self.P_nd = self.w + self.phi_nd

        # dimensional pressure (for outputs / homogenisation in Pa)
        self.P_dim = Constant(self.p_scale) * self.P_nd

        # cavitation penalty uses smooth pos(-P_nd) with nondim epsilon
        self.negP_pos = self.smooth_pos(-self.P_nd, Constant(self.epsP_nd))


    def _setup_forms(self) -> None:
        # Physical gradients on unit mesh
        gradP = self.grad_phys(self.P_nd)
        gradv = self.grad_phys(self.v)

        # Reynolds residual (nondimensional)
        R_reynolds = inner(self.F_adv_nd - self.K_nd * gradP, gradv) * dx

        # Cavitation penalty (nondimensional pressure, nondim gamma)
        R_cav = Constant(self.gamma_nd) * self.mask * self.negP_pos * self.v * dx

        self.R = R_reynolds + R_cav
        self.J = derivative(self.R, self.w, self.dw)


    def _setup_point_constraint_bc(self) -> None:
        p = self.params

        dpx = float(p.dpx_celljump)
        dpy = float(p.dpy_celljump)

        # Your existing gauge logic (DIMENSIONAL)
        p_ref0 = p.p0 - 0.5 * (dpx + dpy)
        min_delta = p_ref0 + min(0.0, dpx, dpy, dpy + dpx)
        p_ref = p_ref0 if min_delta >= 0.0 else (p_ref0 - min_delta)

        # Pin nondimensional w such that P_dim(0,0) = p_ref
        # Since phi_nd(0,0)=0 => P_dim = p_scale * w
        w_pin = p_ref / self.p_scale

        self.bc_pin = DirichletBC(
            self.Vper,
            Constant(w_pin),
            "near(x[0],0.0) && near(x[1],0.0)",
            method="pointwise",
        )


    # -------------------------
    # Core methods
    # -------------------------
    @staticmethod
    def smooth_pos(z, eps):
        # Smooth positive part: pos(z) ~ 0.5*(z + sqrt(z^2 + eps^2))
        return 0.5 * (z + sqrt(z * z + eps * eps))

    @staticmethod
    def smooth_levelset_mask(phi, delta, mask_core, mask_fluid):
        """
        UFL mask:
        phi = H - hmin

        mask = mask_fluid        if phi > +delta
             = mask_core         if phi < -delta
             = linear transition otherwise
        """
        # Branch on *float* to avoid UFL division by zero
        if float(delta) == 0.0:
            return conditional(gt(phi, 0.0), mask_fluid, mask_core)

        t = (phi + delta) / (2.0 * delta)  # [-delta, +delta] -> [0, 1]
        mask_transition = mask_core + (mask_fluid - mask_core) * t

        return conditional(
            gt(phi - delta, 0.0),
            mask_fluid,
            conditional(gt((-delta) - phi, 0.0), mask_core, mask_transition),
        )

    def solve(self) -> None:
        problem = NonlinearVariationalProblem(self.R, self.w, bcs=[self.bc_pin], J=self.J)
        solver = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = self.settings.abs_tolerance
        prm["newton_solver"]["relative_tolerance"] = self.settings.rel_tolerance
        prm["newton_solver"]["maximum_iterations"] = self.settings.max_iterations
        prm["newton_solver"]["report"] = self.settings.print_progress
        prm["newton_solver"]["error_on_nonconvergence"] = False
        prm["newton_solver"]["relaxation_parameter"] = self.settings.relaxation_parameter
        prm["newton_solver"]["linear_solver"] = "mumps"

        self.w.vector().zero()
        solver.solve()

    def post_process(self) -> None:
        # Unmasked physical pressure
        # Dimensional physical pressure in Pa
        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.params.H0) * self.h_nd, self.Vfull)


        # Masked pressure with NaNs outside fluid region # DO WE NEED IF NOT EXPORTING?
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        chi_nodal = interpolate(chi_dg0, self.Vfull)

        p_arr = Pfull.vector().get_local()
        chi_arr = chi_nodal.vector().get_local()

        Pmasked = Function(self.Vfull)
        arr_nan = p_arr.copy()
        arr_nan[chi_arr <= 0.5] = np.nan
        Pmasked.vector().set_local(arr_nan)
        Pmasked.vector().apply("insert")


        # --- domain area and averages ---
        area_total = assemble(Constant(1.0) * dx(domain=self.mesh_m))
        avg_total = assemble(Pfull * dx(domain=self.mesh_m)) / area_total

        # --- fluid-only weighted integral/average (NO NaNs involved) ---
        # Prefer DG0 indicator for a clean "cellwise" fluid region
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = assemble(chi_dg0 * dx(domain=self.mesh_m))

        if fluid_area > 0.0:
            avg_fluid = assemble(Pfull * chi_dg0 * dx(domain=self.mesh_m)) / fluid_area
        else:
            avg_fluid = float("nan")

        print(f"Avg pressure over whole cell = {avg_total:,.6f} Pa")
        print(f"Fluid area fraction          = {fluid_area/area_total:,.6f}")
        print(f"Avg pressure in fluid only   = {avg_fluid:,.6f} Pa")


        
        if self.export_vtk == True:
            Pfull.rename("P", "pressure_unmasked")
            File(os.path.join(self.output_dir, "pressure_unmasked_affine_periodic_penalty_cav_masked.pvd")) << Pfull
            Hfull.rename("H", "film thickness")
            File(os.path.join(self.output_dir, "h_affine_periodic_penalty_cav_masked.pvd")) << Hfull

            Pmasked.rename("P", "pressure")
            File(os.path.join(self.output_dir, "pressure_affine_periodic_penalty_cav_masked.pvd")) << Pmasked

            # Export indicators in DG0
            maskfull = project(self.mask, self.Vdg0)
            maskfull.rename("mask", "immersed mask")
            File(os.path.join(self.output_dir, "mask_affine_periodic_penalty_cav_masked.pvd")) << maskfull
            
            chi_dg0.rename("chi_fluid", "fluid indicator")
            File(os.path.join(self.output_dir, "chi_fluid_affine_periodic_penalty_cav_masked.pvd")) << chi_dg0




def main() -> None:

    n_m = 150  #Size of mixed mesh
    n_h = 60    #Size of hydro mesh

    #Create mesh to pass in to micro solver - we pass both so that the solver can switch if there is a change in whether we are in mixed or hydro
    mesh_m = UnitSquareMesh(n_m, n_m)
    mesh_h = UnitSquareMesh(n_h, n_h)

    Ah = 1e-7
    kx = 3
    ky = 3

    def ht(x, y, xmax, ymax, H0):
        x_d, y_d = x * xmax, y * ymax
        hmin0 = H0 - Ah

        if hmin0 < 0:
            print(
                f"Warning: Negative film thickness hmin0 = {hmin0}."
            )
            sys.stdout.flush()
        out = (
            H0 - x*1e-7 + y*1.0e-7
            + (
                Ah
                * (cos(kx * 2 * pi * x_d / xmax)
                + cos(ky * 2 * pi * y_d / ymax))
            )
        )
        return out
    
    params = MicroPhysicalParameters(
        Ux=20,
        Uy=00.0,
        eta0 = 0.001,
        rho0= 1000,
        penalty_gamma=1e4,
        xmax=1e-6,
        ymax=1e-6,
        p0=1e1,
        dpdx=0.0,
        dpdy=0.0,
        H0=3.8e-7,
        h_min=0.0,
        HT = 0,
        PT = 0,
        Tend = 0,
        k_spring = 1e13,
    )
    settings = MicroSolverSettings(
        relaxation_parameter=0.8,
        max_iterations=200,
        abs_tolerance=1e-8,
        rel_tolerance=1e-6,
        delta_h = 0,
        eps_solid = 1e-8,
        eps_smooth = 1e-6,    
        print_progress= False,
    )
    import time
    start = time.time()
    out_dir = "./data/output/mixed_micro/"
    solver = MicroMixedSolver(mesh_m, mesh_h, params, settings, 1, ht, export_vtk=True, output_dir=out_dir, auto_solve=False)
    solver.solve()
    solver.post_process()
    print(f'Compute time = {time.time() - start}')


if __name__ == "__main__":
    main()