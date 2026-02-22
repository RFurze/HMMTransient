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
import time
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
        t = time.time(); print("init: _setup_export"); sys.stdout.flush()
        self._setup_export(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        
        t = time.time(); print("init: _setup_coordinates"); sys.stdout.flush()
        self._setup_coordinates(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _time_controls"); sys.stdout.flush()
        self._setup_time_controls(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _nondimensionalise"); sys.stdout.flush()
        self._nondimensionalise(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _setup__film__and_mask"); sys.stdout.flush()        
        self._setup_film_and_mask(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _setup_spaces_and_unknowns"); sys.stdout.flush()        
        self._setup_spaces_and_unknowns(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _setup_pressure_expressions"); sys.stdout.flush()        
        self._setup_pressure_expressions(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _setup_forms"); sys.stdout.flush()        
        self._setup_forms(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()
        t = time.time(); print("init: _setup_point_constraint_bc"); sys.stdout.flush()        
        self._setup_point_constraint_bc(); print(f"done in {time.time()-t:.3f}s"); sys.stdout.flush()

        if auto_solve:
            self.run_steady()
            self.post_process()

    # -------------------------
    # Setup helpers
    # -------------------------
    def _setup_export(self) -> None:
        if self.export_vtk:
            os.makedirs(self.output_dir, exist_ok=True)
            self._file_times: list[tuple[str, float]] = []

        # Transient XDMF series writers (opened lazily inside run()).
        self._xdmf_P: Optional[XDMFFile] = None
        self._xdmf_H: Optional[XDMFFile] = None
        self._xdmf_Pmasked: Optional[XDMFFile] = None
        self._xdmf_M: Optional[XDMFFile] = None
        self._export_xdmf_series: bool = False

    def _setup_coordinates(self) -> None:
        self.x, self.y = SpatialCoordinate(self.mesh_m)


    def _setup_time_controls(self) -> None:
        """Create Constants used for transient updates (film thickness & pinned pressure)."""
        p = self.params

        # Nondimensional time variable T*; dimensional time is t = T* * t0.
        self.T_nd_const = Constant(0.0)

        # Constants to allow time-varying inputs without rebuilding UFL graphs
        self.H0_const = Constant(float(p.H0))
        self.HT_const = Constant(float(p.HT))
        self.p0_const = Constant(float(p.p0))
        self.PT_const = Constant(float(p.PT))

        # End time (dimensional seconds) used to ramp reference pressure
        self.Tend_dim = float(p.Tend) if p.Tend is not None else 0.0


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

        # Time scale (as in the NGSolve solver): t0 = 2*Lx/|Ux|
        self.t0 = (2.0 * self.Lx) / (self.Uref + 1e-12)

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

            


    def _call_ht(self, x, y):
        """Call film thickness callback; supports steady or transient signatures."""
        p = self.params
        T_dim = self.T_nd_const * Constant(self.t0)  # dimensional time (s)

        try:
            # transient-ready signature: ht(x,y,xmax,ymax,H0,HT,T_dim,Ux,Uy)
            return self.ht_cb(x, y, p.xmax, p.ymax, self.H0_const, self.HT_const, T_dim, 2.0*p.Ux, 2.0*p.Uy)
        except TypeError:
            # steady signature: ht(x,y,xmax,ymax,H0)
            return self.ht_cb(x, y, p.xmax, p.ymax, self.H0_const)

    def _compute_p_ref_dim(self, t_dim: float) -> float:
        """Pinned dimensional reference pressure (Pa), lifted so no corner is negative (gauge=0)."""
        p = self.params
        dpx = float(p.dpx_celljump)
        dpy = float(p.dpy_celljump)

        # Base reference at t=0
        p_ref0 = float(self.p0_const) - 0.5 * (dpx + dpy)

        if self.Tend_dim > 0.0:
            p_ref1 = p_ref0 + float(self.PT_const) * self.Tend_dim
            s = max(0.0, min(1.0, t_dim / self.Tend_dim))
            p_ref = p_ref0 + (p_ref1 - p_ref0) * s
        else:
            p_ref = p_ref0

        min_corner = min(p_ref, p_ref + dpx, p_ref + dpy, p_ref + dpx + dpy)
        if min_corner < 0.0:
            p_ref = p_ref - min_corner
        return p_ref

    def _update_pinned_bc(self, t_dim: float) -> None:
        p_ref = self._compute_p_ref_dim(t_dim)
        w_pin = p_ref / float(self.p_scale)
        self.w_pin_const.assign(w_pin)

    def _setup_film_and_mask(self) -> None:
        p = self.params
        s = self.settings

        # Film thickness: ht_cb returns DIMENSIONAL H(x,y); convert to nondim h = H/H0
        H_dim = self._call_ht(self.x, self.y)
        self.h_nd = H_dim / self.H0_const

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

        # Allocate history holders for transient BDF (functions on periodic space)
        self.h_hist_nm1 = Function(self.Vper, name="h_nm1")
        self.h_hist_nm2 = Function(self.Vper, name="h_nm2")
        self.mask_hist_nm1 = Function(self.Vper, name="mask_nm1")
        self.mask_hist_nm2 = Function(self.Vper, name="mask_nm2")
        self._have_two_hist = False

        # Build baseline (steady) residual/Jacobian
        self._build_forms(include_transient_terms=False, dt_nd=None)

    def _build_forms(self, *, include_transient_terms: bool, dt_nd: float | None) -> None:
        """(Re)build residual/Jacobian for steady or transient solves."""
        gradP = self.grad_phys(self.P_nd)
        gradv = self.grad_phys(self.v)

        R_reynolds = inner(self.F_adv_nd - self.K_nd * gradP, gradv) * dx
        R_cav = Constant(self.gamma_nd) * self.mask * self.negP_pos * self.v * dx

        R = R_reynolds + R_cav

        if include_transient_terms and dt_nd is not None and dt_nd > 0.0:
            if self._have_two_hist:
                alf0, alf1, alf2 = (1.5, -2.0, 0.5)
                R += (alf0 / dt_nd) * self.mask * self.h_nd * self.v * dx
                R += (alf1 / dt_nd) * self.mask_hist_nm1 * self.h_hist_nm1 * self.v * dx
                R += (alf2 / dt_nd) * self.mask_hist_nm2 * self.h_hist_nm2 * self.v * dx
            else:
                alf0, alf1 = (1.0, -1.0)
                R += (alf0 / dt_nd) * self.mask * self.h_nd * self.v * dx
                R += (alf1 / dt_nd) * self.mask_hist_nm1 * self.h_hist_nm1 * self.v * dx

        self.R = R
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

        self.w_pin_const = Constant(w_pin)
        self.bc_pin = DirichletBC(
            self.Vper,
            self.w_pin_const,
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


    # -------------------------
    # Transient XDMF series export helpers
    # -------------------------
    def _open_xdmf_series(self) -> None:
        """Open XDMF writers for transient time-series (pressure + film thickness)."""
        if not self.export_vtk:
            return
        if self._xdmf_P is not None or self._xdmf_H is not None:
            return

        # Use XDMF/HDF5 time series output (ParaView-friendly)
        p_path = os.path.join(self.output_dir, "pressure_series.xdmf")
        pm_path = os.path.join(self.output_dir, "pressure_masked_series.xdmf")
        m_path = os.path.join(self.output_dir, "mask_series.xdmf")
        h_path = os.path.join(self.output_dir, "film_thickness_series.xdmf")

        self._xdmf_P = XDMFFile(MPI.comm_world, p_path)
        self._xdmf_Pmasked = XDMFFile(MPI.comm_world, pm_path)
        self._xdmf_H = XDMFFile(MPI.comm_world, h_path)
        self._xdmf_M = XDMFFile(MPI.comm_world, m_path)

        for f in (self._xdmf_P, self._xdmf_Pmasked, self._xdmf_H, self._xdmf_M):
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            f.parameters["rewrite_function_mesh"] = False

        self._export_xdmf_series = True

    def _write_xdmf_step(self, t_dim: float) -> None:
        """Write one timestep to the XDMF time series (dimensional time in seconds)."""
        if (
            not self._export_xdmf_series
            or self._xdmf_P is None
            or self._xdmf_H is None
            or self._xdmf_Pmasked is None
            or self._xdmf_M is None
        ):
            return

        # Project to a full (non-periodic) space for clean output.
        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.params.H0) * self.h_nd, self.Vfull)

        # Export mask as a separate field (0..1 smooth mask).
        Mfull = project(self.mask, self.Vfull)

        # Export a masked pressure field with NaNs in "solid/contact" regions.
        # Use chi_fluid (sharp indicator) so the NaN masking is crisp.
        Chifull = project(self.chi_fluid, self.Vfull)

        Pmasked = Function(self.Vfull)
        Pmasked.vector()[:] = Pfull.vector()

        p_arr = Pmasked.vector().get_local()
        chi_arr = Chifull.vector().get_local()
        p_arr[chi_arr < 0.5] = np.nan
        Pmasked.vector().set_local(p_arr)
        Pmasked.vector().apply("insert")

        Pfull.rename("P", "pressure")
        Hfull.rename("H", "film_thickness")
        Mfull.rename("mask", "mask")
        Pmasked.rename("P_masked", "pressure_masked")

        t_val = float(t_dim)
        self._xdmf_P.write(Pfull, t_val)
        self._xdmf_Pmasked.write(Pmasked, t_val)
        self._xdmf_H.write(Hfull, t_val)
        self._xdmf_M.write(Mfull, t_val)

    def _close_xdmf_series(self) -> None:
        """Close XDMF writers if opened."""
        if self._xdmf_P is not None:
            self._xdmf_P.close()
            self._xdmf_P = None
        if self._xdmf_Pmasked is not None:
            self._xdmf_Pmasked.close()
            self._xdmf_Pmasked = None
        if self._xdmf_H is not None:
            self._xdmf_H.close()
            self._xdmf_H = None
        if self._xdmf_M is not None:
            self._xdmf_M.close()
            self._xdmf_M = None
        self._export_xdmf_series = False

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

    
    def run_steady(self) -> None:
        """Run a single steady-state solve (alias used by the microscale runner)."""
        self.solve()


    def run(self, Tf: float, n_steps: int = 10) -> None:
        """Run a transient simulation using BE (first step) then BDF2."""
        Tf = float(Tf)
        if Tf <= 0.0 or n_steps <= 1:
            self.run_steady()
            return

        dt_dim = Tf / (n_steps - 1)
        dt_nd = dt_dim / self.t0

        # Initialise at t=0
        self.T_nd_const.assign(0.0)
        self._update_pinned_bc(t_dim=0.0)

        # If exporting and truly transient, write an XDMF/HDF5 time series for ParaView.
        if self.export_vtk and n_steps > 1:
            self._open_xdmf_series()
            # Write initial fields at t=0
            self._write_xdmf_step(t_dim=0.0)

            # Diagnostics at t=0 (computed on DG0 so min() is meaningful).
            try:
                Vdg0 = FunctionSpace(self.mesh_m, "DG", 0)
                phi0_proj = project(self.h_nd - Constant(self.hmin_nd), Vdg0)
                mask0_proj = project(self.mask, Vdg0)

                phi0_local = float(phi0_proj.vector().get_local().min())
                mask0_local = float(mask0_proj.vector().get_local().min())

                phi0 = MPI.min(MPI.comm_world, phi0_local)
                mask0 = MPI.min(MPI.comm_world, mask0_local)

                # print(
                #     f"[Transient step 0/{n_steps-1}] t=0.000000e+00 s | "
                #     f"min(h_nd - hmin_nd)={phi0:.6e} | min(mask)={mask0:.6e}"
                # )
                # sys.stdout.flush()
            except Exception as e:
                print(f"[Transient step 0/{n_steps-1}] diagnostics failed: {e}")
                sys.stdout.flush()
                sys.stdout.flush()
            except Exception as _e:

                sys.stdout.flush()

        h0 = project(self.h_nd, self.Vper)
        m0 = project(self.mask, self.Vper)
        self.h_hist_nm1.assign(h0)
        self.mask_hist_nm1.assign(m0)
        self.h_hist_nm2.assign(h0)
        self.mask_hist_nm2.assign(m0)
        self._have_two_hist = False

        for step in range(1, n_steps):
            t_dim = step * dt_dim
            t_nd = t_dim / self.t0
            self.T_nd_const.assign(t_nd)

            self._update_pinned_bc(t_dim=t_dim)
            self._build_forms(include_transient_terms=True, dt_nd=dt_nd)

            self.solve()

            # Diagnostics: check masking activation (computed on DG0 so min() is meaningful).
            try:
                Vdg0 = FunctionSpace(self.mesh_m, "DG", 0)
                phi_proj = project(self.h_nd - Constant(self.hmin_nd), Vdg0)
                mask_proj = project(self.mask, Vdg0)

                phi_min_local = float(phi_proj.vector().get_local().min())
                mask_min_local = float(mask_proj.vector().get_local().min())

                phi_min = MPI.min(MPI.comm_world, phi_min_local)
                mask_min = MPI.min(MPI.comm_world, mask_min_local)

                print(
                    f"[Transient step {step}/{n_steps-1}] t={t_dim:.6e} s | "
                    f"min(h_nd - hmin_nd)={phi_min:.6e} | min(mask)={mask_min:.6e}"
                )
                sys.stdout.flush()
            except Exception as e:
                print(f"[Transient step {step}/{n_steps-1}] diagnostics failed: {e}")
                sys.stdout.flush()

            # Time-series export
            if self._export_xdmf_series:
                self._write_xdmf_step(t_dim=t_dim)

            # Update histories
            h_curr = project(self.h_nd, self.Vper)
            m_curr = project(self.mask, self.Vper)
            self.h_hist_nm2.assign(self.h_hist_nm1)
            self.mask_hist_nm2.assign(self.mask_hist_nm1)
            self.h_hist_nm1.assign(h_curr)
            self.mask_hist_nm1.assign(m_curr)

            if step >= 2:
                self._have_two_hist = True

        # Close XDMF writers (keeps .xdmf/.h5 consistent on disk)
        if self._export_xdmf_series:
            self._close_xdmf_series()


    def spatial_homogenisation(self):
        """
        Return homogenised quantities with the same tuple structure as the original
        transient microscale solver.

        Returns
        -------
        (Qx, Qy, Pst, Fst, taust_x, taust_y,
         p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin)
        """
        # Domain area
        area_total = assemble(Constant(1.0) * dx(domain=self.mesh_m))

        # --- Pressure / film thickness (dimensional) ---
        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.params.H0) * self.h_nd, self.Vfull)

        p_arr = Pfull.vector().get_local()
        h_arr = Hfull.vector().get_local()

        p_max = float(np.nanmax(p_arr))
        p_min = float(np.nanmin(p_arr))
        h_max = float(np.nanmax(h_arr))
        h_min = float(np.nanmin(h_arr))

        # Average pressure over the whole cell (Pa)
        Pst = float(assemble(Pfull * dx(domain=self.mesh_m)) / area_total)

        # --- Fluid indicator / "fluid fraction" ---
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = assemble(chi_dg0 * dx(domain=self.mesh_m))
        Fst = float(fluid_area / area_total)

        # Effective fluid fraction extrema (kept for downstream compatibility)
        # (new cavitation model does not have f_cav, so we expose a proxy)
        fmax = 1.0 if fluid_area > 0.0 else 0.0
        fmin = 0.0

        # --- Fluxes ---
        # Nondimensional flux vector inside the residual: q_nd = F_adv_nd - K_nd * grad(P_nd)
        gradP_nd = self.grad_phys(self.P_nd)
        q_nd = self.F_adv_nd - self.K_nd * gradP_nd

        # Convert to dimensional volumetric flux per unit width.
        # This is consistent with the nondimensionalisation used in this file:
        # q_dim ~ Uref * H0 * q_nd
        q_dim = Constant(self.Uref * self.params.H0) * q_nd

        Qx = float(assemble(q_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy = float(assemble(q_dim[1] * dx(domain=self.mesh_m)) / area_total)

        # For legacy outputs: provide extrema of the y-flux as qymax/qymin
        qy_full = project(q_dim[1], self.Vfull)
        qy_arr = qy_full.vector().get_local()
        qymax = float(np.nanmax(qy_arr))
        qymin = float(np.nanmin(qy_arr))

        # --- Shear stress proxy (dimensional, Pa) ---
        # Use a simple Couette estimate: tau ~ eta * U / H, averaged over the cell.
        # (Downstream code expects two components.)
        Ux_surf = 2.0 * self.params.Ux
        Uy_surf = 2.0 * self.params.Uy
        eta0 = self.params.eta0

        # Avoid division by zero via a tiny floor in the denominator.
        H_floor = Hfull + Constant(1e-30)
        tau_x_field = Constant(eta0 * Ux_surf) / H_floor
        tau_y_field = Constant(eta0 * Uy_surf) / H_floor

        taust_x = float(assemble(tau_x_field * dx(domain=self.mesh_m)) / area_total)
        taust_y = float(assemble(tau_y_field * dx(domain=self.mesh_m)) / area_total)

        return (
            Qx,
            Qy,
            Pst,
            Fst,
            taust_x,
            taust_y,
            p_max,
            p_min,
            h_max,
            h_min,
            qymax,
            qymin,
            fmax,
            fmin,
        )

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


        
        # Safety: if transient XDMF writers are still open, close them.
        if self._export_xdmf_series:
            self._close_xdmf_series()

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

    Ah = 0
    kx = 1
    ky = 1


    def ht(x, y, xmax, ymax, H0, HT, Tdim, Ux, Uy):
        x_d, y_d = x * xmax, y * ymax

        # Use float(H0) for python-side checks ONLY
        H0_val = float(H0)
        hmin0 = H0_val - Ah
        if hmin0 < 0:
            print(f"Warning: Negative film thickness hmin0 = {hmin0}.")
            sys.stdout.flush()

        # Keep UFL expression for actual thickness
        out = (
            H0 + HT*Tdim
            + Ah*(cos(kx * 2 * pi * (x_d) / xmax) + cos(ky * 2 * pi * y_d / ymax))
        )
        return out
    
    params = MicroPhysicalParameters(
        Ux=0.04994528547621144,
        Uy=1.5470714102138602e-20,
        eta0 = 0.001,
        rho0= 1000,
        penalty_gamma=1e4,
        xmax=1e-6,
        ymax=1e-6,
        p0=3180.0147229299714,
        dpdx=1812560.7270410743,
        dpdy=-17303.083806246395,
        H0=4.6443751562230455e-05,
        h_min=0.0,
        HT = 0, #-0.5e-7/0.005,
        PT = 0, #2/0.005,
        Tend = 0.005,
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
    # solver.solve()
    solver.run(Tf=0, n_steps = 0)
    solver.post_process()
    print(f'Compute time = {time.time() - start}')


if __name__ == "__main__":
    main()