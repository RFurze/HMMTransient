"""
Different version of the original mixed lubrication code that allows us to test diff qudrature degrees
"""

from __future__ import annotations

import logging
import os
import sys
import io
from contextlib import contextmanager
import numpy as np
import xml.etree.ElementTree as ET
from dolfin import *
import numpy as np

# parameters["form_compiler"]["quadrature_degree"] = 4

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

class micro_PhysicalParameters:
    def __init__(
        self,
        Ux,
        Uy,
        eta0,
        rho0,
        penalty_gamma,
        xmax,
        ymax,
        p0,
        dpdx,
        dpdy,
        H0,
        h_min,
        HT,
        PT,
        Tend,
        k_spring,
    ):
        self.Ux = 2 * Ux  # Account for incoming velocity being 1/2 already
        self.Uy = 2 * Uy
        self.eta0 = eta0
        self.rho0 = rho0
        self.penalty_gamma = penalty_gamma
        self.xmax = xmax
        self.ymax = ymax
        self.p0 = p0
        self.dpdx = dpdx
        self.dpdy = dpdy
        self.H0 = H0
        self.h_min = h_min
        self.HT = HT
        self.PT = PT
        self.Tend = Tend
        self.k_spring = k_spring


class micro_SolverSettings:
    def __init__(
        self,
        relaxation_parameter: float,
        max_iterations: int,
        abs_tolerance: float,
        rel_tolerance: float,
        delta_h: float, #Transition width for smoothing contact mask (set to 0 for sharp)
        eps_solid: float, #Porosity or mobility in contact region to avoid singular matrices
        eps_smooth: float, #Smoothing
    ):
        self.relaxation_parameter = relaxation_parameter
        self.max_iterations = max_iterations
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.delta_h = delta_h
        self.eps_solid = eps_solid
        self.eps_smooth = eps_smooth

class PeriodicBoundary(SubDomain):
    def inside(self, X, on_boundary):
        return on_boundary and (
            (near(X[0], 0.0) or near(X[1], 0.0)) and
            (not ((near(X[0], 1.0) and near(X[1], 0.0)) or
                  (near(X[0], 0.0) and near(X[1], 1.0)) or
                  (near(X[0], 1.0) and near(X[1], 1.0))))
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

class MicroMixedSolver:
    def __init__(
        self,
        mesh_m: Mesh,
        mesh_h: Mesh,
        physical_params: micro_PhysicalParameters,
        solver_settings: micro_SolverSettings,
        k: int,
        ht,
        export_vtk: bool = False,
    ) -> None:
        self.mesh_m = mesh_m
        self.mesh_h = mesh_h
        self.params = physical_params
        self.settings = solver_settings
        self.k = k
        self.ht_cb = ht
        self.export_vtk = export_vtk

        if self.export_vtk:
            os.makedirs(self.output_dir, exist_ok=True)
            self._file_times: list[tuple[str, float]] = []
        
        self.x, self.y = SpatialCoordinate(mesh_m)
        self.U  = as_vector((self.params.Ux, self.params.Uy))  # slide in +x
        self.dpx, self.dpy = self.params.dpdx * self.params.xmax, self.params.dpdy * self.params.ymax
        self.C = self.params.p0
        self.rho = self.params.rho0
        self.eta = self.params.eta0
        self.h_min = self.params.h_min
        self.delta_h = self.settings.delta_h
        self.eps_solid = self.settings.eps_solid
        self.eps_smooth = self.settings.eps_smooth

        #evaluate initial film thickness
        self.h = self.ht_cb(
                    self.x, 
                    self.y, 
                    self.params.xmax, 
                    self.params.ymax, 
                    self.params.H0
                    )
        
        self.phi_mask = self.h - self.h_min
        self.chi_fluid = conditional(gt(self.phi_mask, 0.0), Constant(1.0), Constant(0.0))  # for post-processing only TURN OFF IF WE DONT NEED IN ACTUAL PIPELINE
        self.mask = self.smooth_levelset_mask(self.phi_mask, self.delta_h, self.eps_solid, Constant(1.0))

        # -----------------------------
        # Reynolds coefficients (unchanged physics, but masked)
        # -----------------------------
        self.K_base = self.rho*self.h**3/(12.0*self.eta)
        self.F_adv_base = self.rho*(self.U*self.h/2.0)

        # "Immersed" coefficients
        self.K = self.K_base * self.mask
        self.F_adv = self.F_adv_base * self.mask

        self.pbc = PeriodicBoundary()
        self.Vper  = FunctionSpace(self.mesh_m, "CG", 1, constrained_domain=self.pbc)
        self.Vfull = FunctionSpace(self.mesh_m, "CG", 1)  # for pressure/film exportsTURN OFF IF WE DONT NEED IN ACTUAL PIPELINE
        # Discontinuous space for exporting sharp indicator fields without CG ringing/overshoot TURN OFF IF WE DONT NEED IN ACTUAL PIPELINE
        self.Vdg0  = FunctionSpace(self.mesh_m, "DG", 0)


        # -----------------------------
        # Affine ramp phi for jumps
        # -----------------------------
        self.phi = self.dpx*self.x + self.dpy*self.y   # desired jumps when w is periodic

        # Unknown in periodic space
        self.w = Function(self.Vper, name="w")
        self.v = TestFunction(self.Vper)
        self.dw = TrialFunction(self.Vper)

        # Physical pressure (UFL expression)
        self.P = self.w + self.phi

        # Cavitation penalty uses pos(-P)
        self.negP_pos = self.smooth_pos(-self.P, self.eps_smooth)

        # -----------------------------
        # Weak form (periodic => no boundary terms)
        # Reynolds: div(F_adv - K*grad(P)) = 0
        # Weak: ∫ (F_adv - K*grad(P)) · grad(v) dx = 0
        #
        # Masking:
        # - F_adv and K are already multiplied by mask
        # - cavitation penalty is also weighted by mask, so solid/contact regions do not
        #   "care" about P>=0 (pressure there is not physical)
        # -----------------------------
        self.R_reynolds = inner(self.F_adv - self.K*grad(self.P), grad(self.v))*dx
        self.R_cav      = self.params.penalty_gamma * self.mask * self.negP_pos * self.v * dx

        self.R = self.R_reynolds + self.R_cav
        self.J = derivative(self.R, self.w, self.dw)

        #Point constraint
        self.p_ref0 = self.params.p0 - 0.5 * (self.dpx + self.dpy)
        self.min_delta = self.p_ref0 + min(0.0, self.dpx, self.dpy, self.dpy + self.dpx)
        self.p_ref  = self.p_ref0 if (self.min_delta) >= 0.0 else (self.p_ref0 - self.min_delta)

        self.bc_pin = DirichletBC(
            self.Vper,
            Constant(self.p_ref),
            "near(x[0],0.0) && near(x[1],0.0)",
            method="pointwise"
        )

        self.run_solve()
        self.post_process()


    def smooth_pos(self, z, eps):
        # Smooth positive part: pos(z) ~ 0.5*(z + sqrt(z^2 + eps^2))
        return 0.5*(z + sqrt(z*z + eps*eps))

    def smooth_levelset_mask(self, phi, delta, mask_core, mask_fluid):
        """
        UFL version of the NGSolve mask:
        phi = H - hmin

        mask = mask_fluid               if phi > +delta
            = mask_core                if phi < -delta
            = linear transition        otherwise
        """
        # sharp mask if delta == 0 (or extremely small)
        # NOTE: we branch in Python on the *float* value to avoid division by zero in UFL.
        if float(delta) == 0.0:
            return conditional(gt(phi, 0.0), mask_fluid, mask_core)

        t = (phi + delta) / (2.0*delta)  # maps [-delta, +delta] -> [0, 1]
        mask_transition = mask_core + (mask_fluid - mask_core)*t

        return conditional(gt(phi - delta, 0.0),
                        mask_fluid,
                        conditional(gt((-delta) - phi, 0.0),
                                    mask_core,
                                    mask_transition))


    def run_solve(self):
        # -----------------------------
        # Nonlinear solve (Newton)
        # -----------------------------
        problem = NonlinearVariationalProblem(self.R, self.w, bcs=[self.bc_pin], J=self.J)
        solver  = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = self.settings.abs_tolerance
        prm["newton_solver"]["relative_tolerance"] = self.settings.rel_tolerance
        prm["newton_solver"]["maximum_iterations"] = self.settings.max_iterations
        prm["newton_solver"]["report"] = False
        prm["newton_solver"]["error_on_nonconvergence"] = False
        prm["newton_solver"]["relaxation_parameter"] = self.settings.relaxation_parameter
        prm["newton_solver"]["linear_solver"] = "mumps"  # if available; otherwise try "petsc"

        # initial guess: start from 0
        self.w.vector().zero()
        solver.solve()

    def post_process(self):

        # -----------------------------
        # Export physical pressure + mask fields
        # -----------------------------
        # Unmasked (physical) pressure
        Pfull = project(self.P, self.Vfull)
        Pfull.rename("P", "pressure_unmasked")
        File("pressure_unmasked_affine_periodic_penalty_cav_masked.pvd") << Pfull
        print("\nWrote: pressure_unmasked_affine_periodic_penalty_cav_masked.pvd")

        # Masked pressure with NaNs outside the fluid region (ParaView can treat NaN as blank)
        # NOTE: dolfin.interpolate cannot take a raw UFL Conditional, so first project chi into DG0,
        # then interpolate that *Function* to CG for nodal masking.
        chi_dg0   = project(self.chi_fluid, self.Vdg0)       # bounded cellwise in [0,1]
        chi_nodal = interpolate(chi_dg0, self.Vfull)    # nodal values for masking

        p_arr   = Pfull.vector().get_local()
        chi_arr = chi_nodal.vector().get_local()

        Pmasked = Function(self.Vfull)
        arr_nan = p_arr.copy()
        arr_nan[chi_arr <= 0.5] = np.nan
        Pmasked.vector().set_local(arr_nan)
        Pmasked.vector().apply("insert")
        Pmasked.rename("P", "pressure")
        File("pressure_affine_periodic_penalty_cav_masked.pvd") << Pmasked
        print("Wrote: pressure_affine_periodic_penalty_cav_masked.pvd")

        # Film thickness
        hfull = project(self.h, self.Vfull)
        hfull.rename("H", "film thickness")
        File("h_affine_periodic_penalty_cav_masked.pvd") << hfull

        # Export indicators in DG0 to avoid CG projection overshoot near sharp interfaces
        maskfull = project(self.mask, self.Vdg0)
        maskfull.rename("mask", "immersed mask")
        File("mask_affine_periodic_penalty_cav_masked.pvd") << maskfull

        chifull = chi_dg0
        chifull.rename("chi_fluid", "fluid indicator")
        File("chi_fluid_affine_periodic_penalty_cav_masked.pvd") << chifull

        # Homogenise pressure
        pst = assemble(Pfull*dx)



def main() -> None:
    import time
    import math

    # -----------------------------
    # Sweep settings
    # -----------------------------
    quad_degrees = [20, 8, 6, 4, 2]  # adjust as you like

    # -----------------------------
    # Problem setup (same as your existing main)
    # -----------------------------
    n_m = 200   # Size of mixed mesh
    n_h = 60    # Size of hydro mesh

    # Create meshes once (re-used)
    mesh_m = UnitSquareMesh(n_m, n_m)
    mesh_h = UnitSquareMesh(n_h, n_h)

    Ah = 0.2
    kx = 1
    ky = 1

    def ht(x, y, xmax, ymax, H0):
        x_d, y_d = x * xmax, y * ymax
        hmin0 = H0 - Ah

        if hmin0 < 0:
            print(f"Warning: Negative film thickness hmin0 = {hmin0}.")
            sys.stdout.flush()

        return (
            H0
            + Ah * (cos(kx * 2 * pi * x_d / xmax) + cos(ky * 2 * pi * y_d / ymax))
        )

    params = micro_PhysicalParameters(
        Ux=10.0,
        Uy=0.0,
        eta0=1,
        rho0=1,
        penalty_gamma=1e4,
        xmax=1,
        ymax=1,
        p0=1,
        dpdx=10,
        dpdy=0,
        H0=0.5,
        h_min=0.2,
        HT=0,
        PT=0,
        Tend=0,
        k_spring=1e13,
    )

    settings = micro_SolverSettings(
        relaxation_parameter=0.8,
        max_iterations=200,
        abs_tolerance=1e-8,
        rel_tolerance=1e-6,
        delta_h=0,      # sharp mask (as in your current code)
        eps_solid=1e-8,
        eps_smooth=1e-6,
    )

    # -----------------------------
    # Sweep and monitor L2(P) + time
    # -----------------------------
    results = []
    prev_l2 = None

    print("\nQuadrature sweep:")
    print(" qdeg | time (s) |   L2(P)         | rel Δ L2 vs prev")
    print("-----:|---------:|----------------:|-----------------:")

    for q in quad_degrees:
        # Set quadrature degree globally (keeps your code "as is")
        parameters["form_compiler"]["quadrature_degree"] = int(q)

        t0 = time.perf_counter()

        # Run the solver (unchanged)
        solver = MicroMixedSolver(mesh_m, mesh_h, params, settings, 1, ht, False)

        # Compute L2 norm of physical pressure (project P into Vfull)
        # P is UFL expression: solver.P = solver.w + solver.phi
        Pfull = project(solver.P, solver.Vfull)
        # l2P = math.sqrt(assemble(Pfull * Pfull * dx))
        l2P = math.sqrt(assemble((Pfull*Pfull) * solver.mask * dx))

        dt = time.perf_counter() - t0

        if prev_l2 is None:
            rel = float("nan")
        else:
            denom = max(1.0, abs(prev_l2))
            rel = abs(l2P - prev_l2) / denom

        print(f"{q:5d} | {dt:8.3f} | {l2P:16.9e} | {rel:17.9e}")

        results.append((q, dt, l2P, rel))
        prev_l2 = l2P

    # Optional: pick "good enough" q where rel change drops below a threshold
    tol = 1e-4
    stable = [q for (q, _, _, rel) in results if (not math.isnan(rel) and rel < tol)]
    if stable:
        print(f"\nFirst quadrature degree with rel ΔL2 < {tol:g}: q = {stable[0]}")
    else:
        print(f"\nNo quadrature degree in {quad_degrees} achieved rel ΔL2 < {tol:g}")



if __name__ == "__main__":
    main()