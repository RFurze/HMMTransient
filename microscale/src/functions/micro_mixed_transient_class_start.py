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
from fenics import *  # legacy FEniCS
parameters["form_compiler"]["quadrature_degree"] = 4
set_log_level(LogLevel.ERROR)
set_log_active(False)
# -------------------------------------------------------------------
# Logging / utilities
# -------------------------------------------------------------------
# NOTE: In multiprocessing runs, writing a shared log file from many workers can
# severely slow execution due to file I/O contention. We therefore disable file
# logging by default. Enable by setting MICRO_DEBUG_LOG=1.
_DEBUG_LOG = os.environ.get("MICRO_DEBUG_LOG", "0").strip().lower() in ("1", "true", "yes", "y")

if _DEBUG_LOG:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="simulation_log.log",
        filemode="w",
    )
else:
    # Disable logging to avoid any overhead / file contention in worker processes.
    logging.disable(logging.CRITICAL)

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


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a boolean-ish environment variable."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "y")


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
        self._setup_time_controls()
        self._nondimensionalise()
        self._setup_film_and_mask()
        self._setup_spaces_and_unknowns()
        self._setup_pressure_expressions()
        self._setup_forms()
        self._setup_point_constraint_bc()
        # ---- nonlinear solver reuse ----
        # Build the NonlinearVariationalProblem/Solver once and reuse across solves.
        # If the residual/Jacobian is rebuilt (e.g. transient stepping), we will
        # mark the solver as stale and rebuild it on demand.
        self._nls_problem: Optional[NonlinearVariationalProblem] = None
        self._nls_solver: Optional[NonlinearVariationalSolver] = None
        self._solver_needs_rebuild: bool = True
        self._build_nonlinear_solver()

        # ---- spring-film coupling settings (can be overridden after init) ----
        self.spring_max_iter: int = 20
        self.spring_rtol: float = 1e-6
        self.spring_relax: float = 0.8
        # Cache cell area (used in spring coupling and homogenisation)
        self._area_total: float = 1.0

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

        # Diagnostics can be expensive due to extra projections.
        # Keep off by default; enable with MICRO_DIAGNOSTICS=1.
        self._diagnostics_enabled: bool = _env_flag("MICRO_DIAGNOSTICS", "0")

    def _setup_coordinates(self) -> None:
        self.x, self.y = SpatialCoordinate(self.mesh_m)


    def _setup_time_controls(self) -> None:
        """Create Constants used for transient updates (film thickness & pinned pressure)."""
        p = self.params

        # Nondimensional time variable T*; dimensional time is t = T* * t0.
        self.T_nd_const = Constant(0.0)
        self.T_dim_const = Constant(0.0)

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

    def grad_cell(self, q):
        """Gradient w.r.t. unit-cell coordinates (x*,y*) on UnitSquareMesh."""
        return as_vector((Dx(q, 0), Dx(q, 1)))


    def _call_ht(self, x, y):
        p = self.params

        try:
            return self.ht_cb(
                x,
                y,
                p.xmax,
                p.ymax,
                self.H0_const,
                self.HT_const,
                self.T_dim_const,
                2.0 * p.Ux,
                2.0 * p.Uy,
            )
        except TypeError:
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

        # Film thickness: ht_cb returns DIMENSIONAL H(x,y,t); convert to nondim base h = H/H0
        H_dim = self._call_ht(self.x, self.y)
        self.h_base_nd = H_dim / self.H0_const

        # Uniform spring deflection added to film thickness (nondimensional).
        # If k_spring <= 0, coupling is disabled (hd stays 0).
        self.hd_nd_const = Constant(0.0)

        # Total film thickness used throughout the residual/transient terms
        self.h_nd = self.h_base_nd + self.hd_nd_const

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

        # Nondimensional velocity vector (note: your p.U multiplies by 2; keep that convention if desired)
        # If you want to preserve the "incoming is half already" logic, apply it BEFORE nondim:
        Ux_eff = 2.0 * self.Ux_nd
        Uy_eff = 2.0 * self.Uy_nd
        self.U_nd = as_vector((Constant(Ux_eff), Constant(Uy_eff)))

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

        # --- Roelands viscosity (dimensional) ---
        eta_dim = self.roelands_eta_dim(self.P_dim)     # Pa·s
        eta_nd = eta_dim / Constant(self.params.eta0)   # nondim eta/eta0 (consistent with omega)

        # (optional) keep density constant for now
        rho_nd = Constant(1.0)

        # --- masked transport coefficients (now pressure-dependent via eta_nd) ---
        self.K_nd = rho_nd * Constant(self.omega) * (self.h_nd**3 / eta_nd) * self.mask
        self.F_adv_nd = rho_nd * (self.U_nd * self.h_nd) * self.mask

        # cavitation penalty uses smooth pos(-P_nd) with nondim epsilon
        self.negP_pos = self.smooth_pos(-self.P_nd, Constant(self.epsP_nd))


    def _setup_forms(self) -> None:

        # Allocate history holders for transient BDF (functions on periodic space)
        self.h_hist_nm1 = Function(self.Vper, name="h_nm1")
        self.h_hist_nm2 = Function(self.Vper, name="h_nm2")
        self.mask_hist_nm1 = Function(self.Vper, name="mask_nm1")
        self.mask_hist_nm2 = Function(self.Vper, name="mask_nm2")
        self._have_two_hist = False

        # --- Transient controls as Constants so we do NOT rebuild UFL forms per step ---
        # dt in nondimensional time units
        self.dt_nd_const = Constant(1.0)
        # BDF coefficients a0,a1,a2 for dh/dt ≈ (a0 h^n + a1 h^{n-1} + a2 h^{n-2})/dt
        # (For backward Euler, set a2=0.)
        self.bdf_a0 = Constant(0.0)
        self.bdf_a1 = Constant(0.0)
        self.bdf_a2 = Constant(0.0)
        # Switch to enable/disable transient squeeze term without changing form structure
        self.transient_on = Constant(0.0)

        # Build baseline residual/Jacobian once (transient term present but disabled)
        self._build_forms(include_transient_terms=True, dt_nd=None)

    def _build_forms(self, *, include_transient_terms: bool, dt_nd: float | None) -> None:
        """(Re)build residual/Jacobian for steady or transient solves."""
        # gradP = self.grad_phys(self.P_nd)
        # gradv = self.grad_phys(self.v)
        gradP = self.grad_cell(self.P_nd)
        gradv = self.grad_cell(self.v)


        R_reynolds = inner(self.F_adv_nd - self.K_nd * gradP, gradv) * dx
        R_cav = Constant(self.gamma_nd) * self.mask * self.negP_pos * self.v * dx

        R = R_reynolds + R_cav

        # NOTE: Keep the transient term in the UFL graph permanently (when requested),
        # but multiply by `self.transient_on` so we can toggle it at runtime (no rebuild).
        if include_transient_terms:
            dh_dt = (
                self.bdf_a0 * self.h_nd
                + self.bdf_a1 * self.h_hist_nm1
                + self.bdf_a2 * self.h_hist_nm2
            ) / self.dt_nd_const
            R += self.transient_on * (self.mask * dh_dt * self.v * dx)

        self.R = R
        self.J = derivative(self.R, self.w, self.dw)
        self._solver_needs_rebuild = True

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

    @staticmethod
    def roelands_eta_dim(p_dim):
        """
        Roelands viscosity model (UFL-safe) returning DIMENSIONAL viscosity [Pa·s].

            eta = eta_p * (eta00/eta_p) ** expo_c
            expo_c = min( (1 + max(p,0)/p_r)**z , expo_cap )

        Implemented via log/exp for stability:
            ln(eta) = ln(eta_p) + expo_c * ln(eta00/eta_p)
        """
        # --- parameters (Pa·s, Pa, dimensionless) ---
        eta_p = Constant(6.31e-5)   # limiting viscosity at high pressure
        eta00 = Constant(1.0e-2)    # zero-pressure viscosity
        p_r = Constant(0.5e8)       # reference pressure (Pa)
        z = Constant(0.548)
        expo_cap = Constant(10.0)

        # clamp negative gauge pressures to zero: p_pos = max(p, 0)
        p_pos = conditional(gt(p_dim, Constant(0.0)), p_dim, Constant(0.0))

        # exponent and cap: expo = (1 + p_pos/p_r)^z ; expo_c = min(expo, expo_cap)
        expo = (Constant(1.0) + p_pos / p_r) ** z
        expo_c = conditional(lt(expo, expo_cap), expo, expo_cap)

        # log/exp form (UFL uses ln())
        base_log = ln(eta00 / eta_p)
        ln_eta = ln(eta_p) + expo_c * base_log
        return exp(ln_eta)

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

    # -------------------------
    # Spring-film coupling helpers
    # -------------------------
    def _spring_enabled(self) -> bool:
        try:
            return float(self.params.k_spring) > 0.0
        except Exception:
            return False

    # def _cell_avg_pressure_dim(self) -> float:
    #     """Cell-average dimensional pressure (Pa)."""
    #     # Using cached area avoids repeated assemble(1*dx).
    #     area = getattr(self, "_area_total", None)
    #     if area is None or area <= 0.0:
    #         area = 1.0
    #         self._area_total = area
    #     return float(assemble(self.P_dim * dx(domain=self.mesh_m)) / area)

    def _cell_avg_positive_pressure_dim(self) -> float:
        Ppos = conditional(gt(self.P_dim, Constant(0.0)), self.P_dim, Constant(0.0))
        area = getattr(self, "_area_total", 1.0)
        return float(assemble(Ppos * dx(domain=self.mesh_m)) / area)

    def _target_hd_nd_from_Pst(self, Pst_dim: float) -> float:
        """Return nondimensional deflection hd/H0 from cell-avg pressure, using stiffness k_spring (Pa/m)."""
        kspring = float(self.params.k_spring)
        if kspring <= 0.0:
            return 0.0
        Ppos = max(float(Pst_dim), 0.0)
        hd_dim = Ppos / kspring
        return float(hd_dim / float(self.params.H0))

    def _build_nonlinear_solver(self) -> None:
        """(Re)build the NonlinearVariationalProblem/Solver.

        We reuse the solver between calls to avoid repeated setup costs.
        If self.R/self.J is rebuilt (e.g. transient stepping), call this again.
        """
        self._nls_problem = NonlinearVariationalProblem(self.R, self.w, bcs=[self.bc_pin], J=self.J)
        self._nls_solver = NonlinearVariationalSolver(self._nls_problem)

        prm = self._nls_solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = self.settings.abs_tolerance
        prm["newton_solver"]["relative_tolerance"] = self.settings.rel_tolerance
        prm["newton_solver"]["maximum_iterations"] = self.settings.max_iterations
        prm["newton_solver"]["report"] = self.settings.print_progress
        prm["newton_solver"]["error_on_nonconvergence"] = True
        prm["newton_solver"]["relaxation_parameter"] = self.settings.relaxation_parameter
        prm["newton_solver"]["linear_solver"] = "mumps"

        # prm["newton_solver"]["linear_solver"] = "gmres"
        # prm["newton_solver"]["preconditioner"] = "ilu"

        self._solver_needs_rebuild = False

    def solve(self) -> None:
        # Reuse solver unless forms were rebuilt (e.g. transient time stepping).
        if self._nls_solver is None or getattr(self, "_solver_needs_rebuild", True):
            self._build_nonlinear_solver()

        # Keep a clean initial guess (matches previous behaviour)
        # self.w.vector().zero()
        self._nls_solver.solve()


    def run_steady(self) -> None:
        """Run a single steady-state solve (alias used by the microscale runner)."""
        # Ensure transient squeeze term is disabled for steady solves.
        if hasattr(self, "transient_on"):
            self.transient_on.assign(0.0)
        self.solve()


    def run(self, Tf: float, n_steps: int = 10) -> None:
        """Run a transient simulation from t=0 to t=Tf using BE (first step) then BDF2."""
        self.run_interval(t_start=0.0, t_end=Tf, n_steps=n_steps)


    def run_interval(self, t_start: float, t_end: float, n_steps: int = 10) -> None:
        """Run a transient simulation from t=t_start to t=t_end using BE (first step) then BDF2."""
        t_start = float(t_start)
        t_end = float(t_end)
        if t_end <= t_start or n_steps <= 1:
            self.run_steady()
            return

        dt_dim = (t_end - t_start) / (n_steps - 1)
        dt_nd = dt_dim / self.t0

        # Configure XDMF output cadence: at most ~20 writes across the run.
        # If n_steps <= 20, write every step.
        if n_steps <= 20:
            export_stride = 1
        else:
            export_stride = int(np.ceil((n_steps - 1) / 20.0))

        # Initialise at t=t_start
        self.T_nd_const.assign(t_start / self.t0)
        self.T_dim_const.assign(t_start)
        self._update_pinned_bc(t_dim=t_start)

        # If exporting and truly transient, write an XDMF/HDF5 time series for ParaView.
        if self.export_vtk and n_steps > 1:
            self._open_xdmf_series()
            # Write initial fields at t=0
            self._write_xdmf_step(t_dim=t_start)

            # Optional diagnostics at t=0 (extra projections can be expensive)
            if self._diagnostics_enabled:
                try:
                    phi0_proj = project(self.h_nd - Constant(self.hmin_nd), self.Vdg0)
                    mask0_proj = project(self.mask, self.Vdg0)

                    phi0_local = float(phi0_proj.vector().get_local().min())
                    mask0_local = float(mask0_proj.vector().get_local().min())

                    phi0 = MPI.min(MPI.comm_world, phi0_local)
                    mask0 = MPI.min(MPI.comm_world, mask0_local)

                    print(
                        f"[Transient step 0/{n_steps-1}] t={t_start:.6e} s | "
                        f"min(h_nd - hmin_nd)={phi0:.6e} | min(mask)={mask0:.6e}"
                    )
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[Transient step 0/{n_steps-1}] diagnostics failed: {e}")
                    sys.stdout.flush()

        h0 = project(self.h_nd, self.Vper)
        m0 = project(self.mask, self.Vper)
        self.h_hist_nm1.assign(h0)
        self.mask_hist_nm1.assign(m0)
        self.h_hist_nm2.assign(h0)
        self.mask_hist_nm2.assign(m0)
        self._have_two_hist = False

        # Enable transient term now that histories exist (no form rebuild needed)
        self.dt_nd_const.assign(float(dt_nd))
        self.transient_on.assign(1.0)
        # Backward Euler coefficients for the first transient step
        self.bdf_a0.assign(1.0)
        self.bdf_a1.assign(-1.0)
        self.bdf_a2.assign(0.0)
        for step in range(1, n_steps):
            t_dim = t_start + step * dt_dim
            t_nd = t_dim / self.t0
            self.T_nd_const.assign(t_nd)
            self.T_dim_const.assign(t_dim)

            self._update_pinned_bc(t_dim=t_dim)

            # Switch coefficients to BDF2 once we have two history states.
            # (Step 1 uses backward Euler; step>=2 uses BDF2.)
            use_bdf2 = bool(step >= 2)
            self._have_two_hist = use_bdf2

            if use_bdf2:
                self.bdf_a0.assign(1.5)
                self.bdf_a1.assign(-2.0)
                self.bdf_a2.assign(0.5)
            else:
                self.bdf_a0.assign(1.0)
                self.bdf_a1.assign(-1.0)
                self.bdf_a2.assign(0.0)
            # --- Solve with optional spring-film coupling ---
            # --- Solve with optional spring-film coupling ---
            if self._spring_enabled():
                hd_old = float(self.hd_nd_const)
                
                # Start with a very conservative initial relaxation factor for the first iteration
                # (You can expose this as a setting if you want, e.g., self.settings.initial_omega)
                omega = 0.05 
                r_old = 0.0
                
                for _it in range(int(self.spring_max_iter)):
                    self.solve()
                    # Pst_dim = self._cell_avg_pressure_dim()
                    Pst_dim = self._cell_avg_positive_pressure_dim()
                    hd_tgt = self._target_hd_nd_from_Pst(Pst_dim)

                    # Calculate the current residual (difference between predicted target and current state)
                    r_curr = hd_tgt - hd_old

                    # Update the relaxation parameter using Aitken's delta-squared process
                    # We skip the very first iteration (_it == 0) because we need history
                    if _it > 0:
                        diff = r_curr - r_old
                        if abs(diff) > 1e-14:  # Safety check to prevent division by zero
                            # Aitken update formula for scalar sequences
                            omega = -omega * (r_old / diff)
                        
                        # Clamp omega to maintain stability
                        # 0.01 prevents the solver from stalling completely
                        # 1.0 prevents it from overshooting like it did previously
                        omega = max(0.01, min(0.2, omega))

                    # Apply the dynamically relaxed update
                    hd_new = hd_old + omega * r_curr
                    
                    # --- Real-Time Diagnostics ---
                    # Only print if we are actually taking multiple iterations to converge
                    # if int(self.spring_max_iter) > 1:
                    #     print(f"  -> Spring Iter {_it:02d}: omega = {omega:.4f} | residual = {r_curr:.2e} | hd = {hd_new:.4e}")
                    #     sys.stdout.flush()
                    # -----------------------------

                    self.hd_nd_const.assign(hd_new)

                    # Convergence check based on the actual applied displacement

                    # Convergence check based on the actual applied displacement
                    denom = 1.0 + abs(hd_old)
                    if abs(hd_new - hd_old) <= float(self.spring_rtol) * denom:
                        break

                    # Store state for the next iteration's Aitken calculation
                    hd_old = hd_new
                    r_old = r_curr

                # Final consistent solve with the converged hd
                self.solve()
            else:
                self.solve()

            # Optional diagnostics: check masking activation (extra projections)
            if self._diagnostics_enabled:
                try:
                    phi_proj = project(self.h_nd - Constant(self.hmin_nd), self.Vdg0)
                    mask_proj = project(self.mask, self.Vdg0)

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

            # Time-series export (throttled to at most ~20 writes over the whole run)
            if self._export_xdmf_series:
                if (step % export_stride) == 0 or step == (n_steps - 1):
                    self._write_xdmf_step(t_dim=t_dim)

            # Update histories
            h_curr = project(self.h_nd, self.Vper)
            m_curr = project(self.mask, self.Vper)
            self.h_hist_nm2.assign(self.h_hist_nm1)
            self.mask_hist_nm2.assign(self.mask_hist_nm1)
            self.h_hist_nm1.assign(h_curr)
            self.mask_hist_nm1.assign(m_curr)

            # _have_two_hist is now set at the start of the step when choosing
            # BDF coefficients.

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
        area_total = 1.0

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
        gradP_nd = self.grad_cell(self.P_nd)
        q_nd = self.F_adv_nd - self.K_nd * gradP_nd
        # q_nd1 = self.F_adv_nd
        # q_nd2 = - self.K_nd * gradP_nd

        # Convert to dimensional volumetric flux per unit width.
        # This is consistent with the nondimensionalisation used in this file:
        # q_dim ~ Uref * H0 * q_nd
        q_dim = Constant(self.Uref * self.params.H0 /2) * q_nd
        # q_dim1 = Constant(self.Uref * self.params.H0 /2) * q_nd1
        # q_dim2 = Constant(self.Uref * self.params.H0 /2) * q_nd2
        
        # Qx1 = float(assemble(q_dim1[0] * dx(domain=self.mesh_m)) / area_total)
        # Qx2 = float(assemble(q_dim2[0] * dx(domain=self.mesh_m)) / area_total)
        # print(f'Homogenised Qx1 {Qx1}, Qx2 {Qx2}, total area {area_total}')
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

        # Pfull = project(self.P_dim, self.Vfull)
        # P00 = float(Pfull(Point(0.0, 0.0)))
        # P10 = float(Pfull(Point(1.0, 0.0)))
        # P01 = float(Pfull(Point(0.0, 1.0)))

        # print("P10-P00 =", P10 - P00, " expected ~", self.params.dpdx * self.params.xmax)
        # print("P01-P00 =", P01 - P00, " expected ~", self.params.dpdy * self.params.ymax)
        # print("Approx dp/dx =", (P10 - P00)/self.params.xmax, " expected ~", self.params.dpdx)
        # print("Approx dp/dy =", (P01 - P00)/self.params.ymax, " expected ~", self.params.dpdy)
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
        area_total = 1.0
        avg_total = assemble(Pfull * dx(domain=self.mesh_m)) / area_total

        # --- fluid-only weighted integral/average (NO NaNs involved) ---
        # Prefer DG0 indicator for a clean "cellwise" fluid region
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = assemble(chi_dg0 * dx(domain=self.mesh_m))

        if fluid_area > 0.0:
            avg_fluid = assemble(Pfull * chi_dg0 * dx(domain=self.mesh_m)) / fluid_area
        else:
            avg_fluid = float("nan")

        # print(f"Avg pressure over whole cell = {avg_total:,.6f} Pa")
        # print(f"Fluid area fraction          = {fluid_area/area_total:,.6f}")
        # print(f"Avg pressure in fluid only   = {avg_fluid:,.6f} Pa")


        
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


def q_re(transient, H, P, Ux, Uy, gradpx, gradpy, Pst, Hdot=None, Pdot=None, dt=0.0):
    if transient:
        Hdt = H + Hdot * dt
        Pdt = P + Pdot * dt

    else:
        Hdt = H
        Pdt = P

    eta = roelands(Pst)
    # print(f'Qx1@ {Ux * Hdt}, Qx2@ {- (Hdt**3) * gradpx / (12 * eta)}')
    qx = Ux * Hdt - (Hdt**3) * gradpx / (12 * eta)
    qy = Uy *Hdt -(Hdt**3) * gradpy / (12 * eta)
    Pst = P
    q = np.column_stack((qx, qy))
    return q

def roelands(p):
    """
    Roelands viscosity model in multiplicative/log form:

        eta = eta_p * (eta00/eta_p) ** ((1 + max(p,0)/p_r)**z_capped)

    where z_capped := min((1 + max(p,0)/p_r)**z, expo_cap).
    """

    # # --- parameters (Pa·s, Pa, dimensionless) ---
    eta_p = 6.31e-5  # Pa·s  (limiting viscosity at high pressure)
    eta00 = 1.0e-2  # Pa·s  (zero-pressure viscosity)
    p_r = 0.5e8  # Pa    (Roelands reference pressure; POSITIVE)
    z = 0.548  # Roelands exponent
    expo_cap = 10.0  # cap on exponent to avoid runaway growth

    # clamp negative gauge pressures to zero
    p_pos = np.maximum(p, 0.0)

    # compute exponent and cap it
    expo = (1.0 + p_pos / p_r) ** z
    expo_c = np.where(
        expo < expo_cap, expo, expo_cap
    )  # if expo < cap use expo else cap

    # log/exp form for numerical stability
    base_log = np.log(eta00 / eta_p)
    ln_eta = np.log(eta_p) + expo_c * base_log
    return np.exp(ln_eta)
    
def _fmt(x, w=10):
    try:
        if x != x:  # NaN
            return f"{'nan':>{w}}"
        ax = abs(float(x))
        if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
            return f"{float(x):>{w}.3e}"
        return f"{float(x):>{w}.6f}"
    except Exception:
        return f"{str(x):>{w}}"
    
# def main() -> None:
#     from CONFIGPenalty import micro_physical, micro_solver, ideal_film_thickness

#     n_m = 40  # Size of mixed mesh
#     n_h = 40  # Size of hydro mesh

#     # Create mesh to pass in to micro solver
#     mesh_m = UnitSquareMesh(n_m, n_m)
#     mesh_h = UnitSquareMesh(n_h, n_h)

#     inputtask = 9300  # 8665
#     tasks = np.load("data/input/OneToOne/tasks.npy", allow_pickle=True)
#     print(f"Shape tasks: {np.shape(tasks)}")
#     (task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _, _, _, _, Hdot, Pdot) = tasks[inputtask, :]
#     print(f"Task {inputtask}: {tasks[inputtask, :]}")
#     print(f"H end = {H}, Pend = {P}")

#     def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
#         Ah = ideal_film_thickness.Ah
#         kx = ideal_film_thickness.kx
#         ky = ideal_film_thickness.ky

#         x_d, y_d = x * xmax, y * ymax

#         # python-side safety checks must use python floats (NOT UFL)
#         H0_val = float(H0)
#         HT_val = float(HT)
#         T_val = float(T)

#         hmin0 = H0_val - Ah
#         hmin1 = H0_val - Ah + HT_val * T_val

#         if (hmin0 < 0.0) or (hmin1 < 0.0):
#             print(
#                 f"Warning: Negative film thickness at T={T_val:.6e}s, "
#                 f"hmin0={hmin0:.6e}, hmin1={hmin1:.6e}."
#             )
#             sys.stdout.flush()

#         return H0 + HT * T + 0.5 * Ah * (
#             cos(kx * 2 * pi * (x_d + T * Ux) / xmax)
#             + cos(ky * 2 * pi * (y_d + Uy * T) / ymax)
#         )
#         # return H0 + HT * T

#     params = MicroPhysicalParameters(
#         Ux=Ux,
#         Uy=Uy,
#         eta0=micro_physical.eta0,
#         rho0=micro_physical.rho0,
#         penalty_gamma=getattr(micro_physical, "penalty_gamma", 1e8),
#         xmax=micro_physical.xmax,
#         ymax=micro_physical.ymax,
#         p0=P,
#         dpdx=gradp1,
#         dpdy=gradp2,
#         H0=H,
#         h_min=0.0,  # Mixed masking disabled
#         HT=Hdot,
#         PT=Pdot,
#         Tend=0.05,
#         k_spring=getattr(micro_physical, "k_spring", 0.0),
#     )

#     settings = MicroSolverSettings(
#         relaxation_parameter=getattr(micro_solver, "relaxation_parameter", 1.0),
#         max_iterations=getattr(micro_solver, "max_iterations", 100),
#         abs_tolerance=getattr(micro_solver, "abs_tolerance", 1e-8),
#         rel_tolerance=getattr(micro_solver, "rel_tolerance", 1e-6),
#         delta_h=getattr(micro_solver, "delta_h", 0.0),
#         eps_solid=getattr(micro_solver, "eps_solid", 1e-8),
#         eps_smooth=getattr(micro_solver, "eps_smooth", 1e-6),
#         print_progress=False,
#     )

#     import time

#     # -----------------------------
#     # Sweep controls
#     # -----------------------------
#     t_end = float(params.Tend)
#     # Starting from 0.045 with 100 time steps seems close to values from starting at 0 with 1000 time steps
#     tstart_values = [0.0, 0.04975, 0.0495, 0.0499]
#     nsteps_values = [2, 5, 10, 1000]

#     # -----------------------------
#     # Helpers: percent diff + table printing (NO pandas)
#     # -----------------------------
#     def _pct_diff(numer: float, denom: float, eps: float = 1e-30) -> float:
#         """
#         Return 100 * numer/denom with safe handling for small denom.
#         If denom ~ 0, returns nan.
#         """
#         d = float(denom)
#         if abs(d) <= eps:
#             return float("nan")
#         return 100.0 * float(numer) / d

#     def _format_cell(x: float, width: int = 12) -> str:
#         if x != x:  # NaN
#             return f"{'nan':>{width}}"
#         ax = abs(x)
#         if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
#             return f"{x:>{width}.3e}"
#         return f"{x:>{width}.6f}"

#     def _print_table(title: str, row_keys, col_keys, values_dict):
#         """
#         values_dict[(row_key, col_key)] -> float
#         """
#         print("\n" + "=" * 80)
#         print(title)
#         print("=" * 80)

#         col_w = 12
#         row_w = 10

#         header = f"{'n_steps':>{row_w}}"
#         for c in col_keys:
#             header += f"{('t=' + f'{c:.4f}'):>{col_w}}"
#         print(header)

#         for r in row_keys:
#             line = f"{str(r):>{row_w}}"
#             for c in col_keys:
#                 val = values_dict.get((r, c), float("nan"))
#                 line += _format_cell(val, width=col_w)
#             print(line)

#     # -----------------------------
#     # Storage for tables:
#     # rows = n_steps, cols = tstart
#     # -----------------------------
#     pct_dP = {}      # (n_steps, tstart) -> %diff dP
#     pct_dQ = {}      # (n_steps, tstart) -> %diff dQ (vector magnitude)
#     pct_dQx = {}     # (n_steps, tstart) -> %diff dQx
#     pct_dQy = {}     # (n_steps, tstart) -> %diff dQy
#     times = {}       # (n_steps, tstart) -> runtime seconds (optional)

#     # NEW: store actual dP and ||dQ|| for later "difference vs reference" tables
#     dP_vals = {}        # (n_steps, tstart) -> dP (signed)
#     dQnorm_vals = {}    # (n_steps, tstart) -> ||dQ||

#     print(f"Running transient micro solver for t in [tstart, {t_end}] with varying tstart/n_steps")

#     for tstart in tstart_values:
#         tstart = float(tstart)
#         if tstart >= t_end:
#             print(f"Skipping tstart={tstart:.6f}: must satisfy tstart < tend={t_end:.6f}")
#             continue

#         for n_steps in nsteps_values:
#             n_steps = int(n_steps)

#             start = time.time()
#             out_dir = f"./data/output/mixed_micro6/tstart_{tstart:.4f}_nsteps_{n_steps}/"

#             solver = MicroMixedSolver(
#                 mesh_m,
#                 mesh_h,
#                 params,
#                 settings,
#                 1,
#                 ht,
#                 export_vtk=False,
#                 output_dir=out_dir,
#                 auto_solve=False,
#             )
#             solver.T_dim_const.assign(tstart)
#             solver.T_nd_const.assign(tstart / solver.t0)
#             solver._update_pinned_bc(t_dim=tstart)
#             solver.transient_on.assign(0.0)
#             solver.solve()

#             solver.run_interval(t_start=tstart, t_end=t_end, n_steps=n_steps)

#             (
#                 Qx,
#                 Qy,
#                 Pst,
#                 Fst,
#                 taust_x,
#                 taust_y,
#                 p_max,
#                 p_min,
#                 max_h,
#                 min_h,
#                 qymax,
#                 qymin,
#                 fmax,
#                 fmin,
#             ) = solver.spatial_homogenisation()

#             # --- Reference over [tstart, t_end]
#             dt_interval = t_end - tstart
#             H_start = params.H0 + params.HT * tstart
#             P_start = params.p0 + params.PT * tstart

#             Q_re = q_re(
#                 True,
#                 H_start,
#                 P_start,
#                 params.Ux,
#                 params.Uy,
#                 params.dpdx,
#                 params.dpdy,
#                 Pst,
#                 Hdot=params.HT,
#                 Pdot=params.PT,
#                 dt=dt_interval,
#             )  # shape (1,2)
#             P_re = P_start + Pdot * dt_interval

#             Q = np.array([float(Qx), float(Qy)], dtype=float)
#             Qre = np.array([float(Q_re[0, 0]), float(Q_re[0, 1])], dtype=float)

#             dQ = Q - Qre
#             dP = float(Pst) - float(P_re)

#             # NEW: store signed dP and ||dQ|| for "difference vs reference" tables
#             dP_vals[(n_steps, tstart)] = float(dP)
#             dQnorm_vals[(n_steps, tstart)] = float(np.linalg.norm(dQ))

#             # --- Percent differences
#             pctP = _pct_diff(abs(dP), abs(P_re))

#             norm_dQ = float(np.linalg.norm(dQ))
#             norm_Qre = float(np.linalg.norm(Qre))
#             pctQ = _pct_diff(norm_dQ, norm_Qre)

#             pctQx = _pct_diff(abs(dQ[0]), abs(Qre[0]))
#             pctQy = _pct_diff(abs(dQ[1]), abs(Qre[1]))

#             pct_dP[(n_steps, tstart)] = pctP
#             pct_dQ[(n_steps, tstart)] = pctQ
#             pct_dQx[(n_steps, tstart)] = pctQx
#             pct_dQy[(n_steps, tstart)] = pctQy

#             elapsed = time.time() - start
#             times[(n_steps, tstart)] = elapsed

#             print(f"tstart={tstart:.6f}, n_steps={n_steps}, compute_time={elapsed:.3f}s")
#             print(f"Qst = [{Qx}, {Qy}], Q_re = {Qre}, dQ = {dQ}")
#             print(f"Pst = {Pst}, P_re = {P_re}, dP = {dP}")
#             print(f"%diff |dQ| = {pctQ:.6g}%, %diff dP = {pctP:.6g}%")
#             sys.stdout.flush()

#     # -----------------------------
#     # Print the requested tables
#     # -----------------------------
#     t_cols = [float(x) for x in tstart_values if float(x) < t_end]
#     n_rows = [int(x) for x in nsteps_values]

#     _print_table(
#         title="Table 1: % diff in dP  (100*|Pst-P_re|/|P_re|)  rows=n_steps, cols=tstart",
#         row_keys=n_rows,
#         col_keys=t_cols,
#         values_dict=pct_dP,
#     )

#     _print_table(
#         title="Table 2: % diff in dQ magnitude  (100*||Q-Q_re||/||Q_re||)  rows=n_steps, cols=tstart",
#         row_keys=n_rows,
#         col_keys=t_cols,
#         values_dict=pct_dQ,
#     )

#     # -----------------------------
#     # NEW: Difference tables vs reference (n_steps=1000, tstart=0.0)
#     # -----------------------------
#     ref_key = (1000, 0.0)
#     ref_dP = dP_vals.get(ref_key, float("nan"))
#     ref_dQnorm = dQnorm_vals.get(ref_key, float("nan"))

#     def _diff_vs_ref(values_dict, ref_value):
#         out = {}
#         for r in n_rows:
#             for c in t_cols:
#                 v = values_dict.get((r, c), float("nan"))
#                 if (v != v) or (ref_value != ref_value):  # NaN checks
#                     out[(r, c)] = float("nan")
#                 else:
#                     out[(r, c)] = (float(v) - float(ref_value))/float(ref_value)
#         return out

#     dP_diff_vs_ref = _diff_vs_ref(dP_vals, ref_dP)
#     dQnorm_diff_vs_ref = _diff_vs_ref(dQnorm_vals, ref_dQnorm)

#     _print_table(
#         title=(
#             "Table 3: ΔdP vs reference  (dP - dP_ref), "
#             "reference is n_steps=1000, tstart=0.0"
#         ),
#         row_keys=n_rows,
#         col_keys=t_cols,
#         values_dict=dP_diff_vs_ref,
#     )

#     _print_table(
#         title=(
#             "Table 4: Δ||dQ|| vs reference  (||dQ|| - ||dQ||_ref), "
#             "reference is n_steps=1000, tstart=0.0"
#         ),
#         row_keys=n_rows,
#         col_keys=t_cols,
#         values_dict=dQnorm_diff_vs_ref,
#     )


def main() -> None:
    from CONFIGPenalty import micro_physical, micro_solver, ideal_film_thickness

    import os
    import sys
    import time
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from fenics import UnitSquareMesh, cos, pi  # assuming you already import fenics elsewhere

    n_m = 40  # Size of mixed mesh
    n_h = 40  # Size of hydro mesh

    # Create mesh to pass in to micro solver
    mesh_m = UnitSquareMesh(n_m, n_m)
    mesh_h = UnitSquareMesh(n_h, n_h)

    inputtask = 9300  # 8665
    tasks = np.load("data/input/OneToOne/tasks.npy", allow_pickle=True)
    print(f"Shape tasks: {np.shape(tasks)}")
    (task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _, _, _, _, Hdot, Pdot) = tasks[inputtask, :]
    print(f"Task {inputtask}: {tasks[inputtask, :]}")
    print(f"H end = {H}, Pend = {P}")

    def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        Ah = ideal_film_thickness.Ah
        kx = ideal_film_thickness.kx
        ky = ideal_film_thickness.ky

        x_d, y_d = x * xmax, y * ymax

        # python-side safety checks must use python floats (NOT UFL)
        H0_val = float(H0)
        HT_val = float(HT)
        T_val = float(T)

        hmin0 = H0_val - Ah
        hmin1 = H0_val - Ah + HT_val * T_val

        if (hmin0 < 0.0) or (hmin1 < 0.0):
            print(
                f"Warning: Negative film thickness at T={T_val:.6e}s, "
                f"hmin0={hmin0:.6e}, hmin1={hmin1:.6e}."
            )
            sys.stdout.flush()

        return H0 + HT * T + 0.5 * Ah * (
            cos(kx * 2 * pi * (x_d + T * Ux) / xmax)
            + cos(ky * 2 * pi * (y_d + Uy * T) / ymax)
        )

    params = MicroPhysicalParameters(
        Ux=Ux,
        Uy=Uy,
        eta0=micro_physical.eta0,
        rho0=micro_physical.rho0,
        penalty_gamma=getattr(micro_physical, "penalty_gamma", 1e8),
        xmax=micro_physical.xmax,
        ymax=micro_physical.ymax,
        p0=P,
        dpdx=gradp1,
        dpdy=gradp2,
        H0=H,
        h_min=0.0,  # Mixed masking disabled
        HT=Hdot,
        PT=Pdot,
        Tend=0.05,
        k_spring=getattr(micro_physical, "k_spring", 0.0),
    )

    settings = MicroSolverSettings(
        relaxation_parameter=getattr(micro_solver, "relaxation_parameter", 1.0),
        max_iterations=getattr(micro_solver, "max_iterations", 100),
        abs_tolerance=getattr(micro_solver, "abs_tolerance", 1e-8),
        rel_tolerance=getattr(micro_solver, "rel_tolerance", 1e-6),
        delta_h=getattr(micro_solver, "delta_h", 0.0),
        eps_solid=getattr(micro_solver, "eps_solid", 1e-8),
        eps_smooth=getattr(micro_solver, "eps_smooth", 1e-6),
        print_progress=False,
    )

    # -----------------------------
    # Sweep controls (FIXED dt)
    # -----------------------------
    t_end = float(params.Tend)

    # Pick a dt you want to test (fixed step size)
    dt = 5.0e-5  # <-- YOU SET THIS

    # Starting times to test
    tstart_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.045, 0.049, 0.0495, 0.0499]

    # Where to save outputs
    base_out = "./data/output/mixed_micro6_fixed_dt/"
    os.makedirs(base_out, exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def run_with_fixed_dt(solver, t_start: float, t_end: float, dt: float) -> int:
        """
        Advance solver from t_start to t_end using (mostly) fixed dt by repeatedly calling:
            solver.run_interval(t, t+dt, n_steps=1)
        If the last interval is shorter than dt (remainder), we do a final shorter step to land at t_end.
        Returns number of steps executed.
        """
        t = float(t_start)
        n = 0
        while t < t_end - 1e-15:
            t_next = t + dt
            if t_next > t_end:
                # remainder step (cannot keep dt constant AND land exactly on t_end)
                rem = t_end - t
                print(f"  [fixed-dt] remainder step: {rem:.6e}s (dt={dt:.6e}) to hit t_end exactly")
                solver.run_interval(t_start=t, t_end=t_end, n_steps=1)  # dt = rem
                n += 1
                t = t_end
                break
            solver.run_interval(t_start=t, t_end=t_next, n_steps=1)  # dt exactly
            n += 1
            t = t_next
        return n

    def compute_reference_terms(tstart: float, t_end: float):
        """
        Same logic you already use: reference is based on start state at tstart and dt_interval.
        """
        dt_interval = t_end - tstart
        H_start = params.H0 + params.HT * tstart
        P_start = params.p0 + params.PT * tstart

        Q_re_arr = q_re(
            True,
            H_start,
            P_start,
            params.Ux,
            params.Uy,
            params.dpdx,
            params.dpdy,
            Pst=None,  # q_re in your code ignores/overrides? keep signature consistent if needed
            Hdot=params.HT,
            Pdot=params.PT,
            dt=dt_interval,
        )
        # NOTE: your original code used Pdot variable; here we use params.PT (same thing).
        P_re = P_start + params.PT * dt_interval
        Qre = np.array([float(Q_re_arr[0, 0]), float(Q_re_arr[0, 1])], dtype=float)
        return Qre, float(P_re)

    # Storage
    results = {}  # tstart -> dict of outputs

    print(f"Running fixed-dt sweep: dt={dt:.6e}, t_end={t_end:.6f}")
    sys.stdout.flush()

    # -----------------------------
    # Run all tstarts
    # -----------------------------
    for tstart in tstart_values:
        tstart = float(tstart)
        if tstart >= t_end:
            print(f"Skipping tstart={tstart:.6f}: must satisfy tstart < tend={t_end:.6f}")
            continue

        start_wall = time.time()
        out_dir = os.path.join(base_out, f"tstart_{tstart:.6f}_dt_{dt:.3e}")
        os.makedirs(out_dir, exist_ok=True)

        solver = MicroMixedSolver(
            mesh_m,
            mesh_h,
            params,
            settings,
            1,
            ht,
            export_vtk=False,
            output_dir=out_dir,
            auto_solve=False,
        )

        # initialise at tstart like you did
        solver.T_dim_const.assign(tstart)
        solver.T_nd_const.assign(tstart / solver.t0)
        solver._update_pinned_bc(t_dim=tstart)
        solver.transient_on.assign(0.0)
        solver.solve()

        # advance with fixed dt (as much as possible)
        n_steps_done = run_with_fixed_dt(solver, t_start=tstart, t_end=t_end, dt=dt)

        (
            Qx,
            Qy,
            Pst,
            Fst,
            taust_x,
            taust_y,
            p_max,
            p_min,
            max_h,
            min_h,
            qymax,
            qymin,
            fmax,
            fmin,
        ) = solver.spatial_homogenisation()

        # your "reference over [tstart, t_end]" for dP/dQ definition
        dt_interval = t_end - tstart
        H_start = params.H0 + params.HT * tstart
        P_start = params.p0 + params.PT * tstart

        Q_re = q_re(
            True,
            H_start,
            P_start,
            params.Ux,
            params.Uy,
            params.dpdx,
            params.dpdy,
            Pst,
            Hdot=params.HT,
            Pdot=params.PT,
            dt=dt_interval,
        )
        P_re = P_start + params.PT * dt_interval

        Q = np.array([float(Qx), float(Qy)], dtype=float)
        Qre = np.array([float(Q_re[0, 0]), float(Q_re[0, 1])], dtype=float)

        dQ = Q - Qre          # vector
        dP = float(Pst) - float(P_re)  # scalar

        elapsed = time.time() - start_wall
        results[tstart] = dict(
            n_steps=n_steps_done,
            elapsed=elapsed,
            Q=Q, Qre=Qre, dQ=dQ,
            Pst=float(Pst), P_re=float(P_re), dP=float(dP),
            out_dir=out_dir,
        )

        print(f"tstart={tstart:.6f}, steps={n_steps_done}, wall={elapsed:.3f}s")
        print(f"  dQ = {dQ} (||dQ||={np.linalg.norm(dQ):.6e}), dP = {dP:.6e}")
        sys.stdout.flush()

    # -----------------------------
    # Reference (tstart=0) for error-vs-start-time plots
    # -----------------------------
    # -----------------------------
    # Reference (tstart=0) for error-vs-start-time plots
    # -----------------------------
    if 0.0 not in results:
        raise RuntimeError("You must include tstart=0.0 in tstart_values to define the reference run.")

    # These are the END-OF-RUN values you want to compare against
    dP_ref_end = float(results[0.0]["dP"])                # scalar
    dQ_ref_end = np.array(results[0.0]["dQ"], dtype=float) # vector (2,)

    # errors vs tstart: compare each end-of-run dP/dQ directly to the tstart=0 end-of-run dP/dQ
    t_list = sorted(results.keys())

    err_dP = np.array([float(results[t]["dP"]) - dP_ref_end for t in t_list], dtype=float)

    # vector error in dQ (component-wise)
    err_dQx = np.array([float(results[t]["dQ"][0]) - float(dQ_ref_end[0]) for t in t_list], dtype=float)
    err_dQy = np.array([float(results[t]["dQ"][1]) - float(dQ_ref_end[1]) for t in t_list], dtype=float)

    # OPTIONAL: one-number summary error for dQ (norm of vector difference)
    err_dQ_vecnorm = np.array([float(np.linalg.norm(results[t]["dQ"] - dQ_ref_end)) for t in t_list], dtype=float)

    # OPTIONAL: difference in norms (what you originally asked for, but now relative to tstart=0 run)
    err_dQ_normdiff = np.array(
        [float(np.linalg.norm(results[t]["dQ"])) - float(np.linalg.norm(dQ_ref_end)) for t in t_list],
        dtype=float,
    )

    # -----------------------------
    # Plot + save figures
    # -----------------------------
    fig_dir = os.path.join(base_out, f"figs_dt_{dt:.3e}")
    os.makedirs(fig_dir, exist_ok=True)

    # dP error vs start time
    plt.figure()
    plt.plot(t_list, err_dP, marker="o")
    plt.xlabel("Start time tstart (s)")
    plt.ylabel("dP_end(tstart) - dP_end(tstart=0)")
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "err_dP_end_vs_tstart.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # dQ error vs start time: vector-difference norm (recommended single scalar)
    plt.figure()
    plt.plot(t_list, err_dQ_vecnorm, marker="o")
    plt.xlabel("Start time tstart (s)")
    plt.ylabel("||dQ_end(tstart) - dQ_end(0)||")
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "err_dQ_vecnorm_end_vs_tstart.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # OPTIONAL: component-wise
    plt.figure()
    plt.plot(t_list, err_dQx, marker="o", label="dQx_end - dQx_end(0)")
    plt.plot(t_list, err_dQy, marker="o", label="dQy_end - dQy_end(0)")
    plt.xlabel("Start time tstart (s)")
    plt.ylabel("Component error in dQ at end")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "err_dQ_components_end_vs_tstart.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # quick summary table to stdout
    print("\n" + "=" * 90)
    print(f"Fixed-dt sweep summary (dt={dt:.3e}). Reference is tstart=0.0")
    print("=" * 90)
    print(f"{'tstart':>10} {'steps':>8} {'dP':>14} {'err_dP':>14} {'||dQ||':>14} {'err_||dQ||':>14}")
    for i, t in enumerate(t_list):
        dP_i = float(results[t]["dP"])
        dQn_i = float(np.linalg.norm(results[t]["dQ"]))
        print(
            f"{t:10.6f} {int(results[t]['n_steps']):8d} "
            f"{dP_i:14.6e} {err_dP[i]:14.6e} {dQn_i:14.6e} {err_dQ_vecnorm[i]:14.6e}"
        )
    print("=" * 90)

if __name__ == "__main__":
    main()