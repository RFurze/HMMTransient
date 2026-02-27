#!/usr/bin/env python3
"""Initialise the transient macroscale solver at each time step.

State from the previous step (``p_init.npy``, ``xi_rot_prev.npy`` etc.) is
loaded and either a load-balance or coupling solve is performed depending on
``--c_iter``. The updated pressure and rotated state are written back to
``output_dir`` for the next time step.

Key command line options from :func:`parse_common_args` (``with_time=True``):
    ``--Time`` and ``--DT`` - current simulation time and step size.
    ``--lb_iter``/``--c_iter`` - iteration counters.
    ``--output_dir`` - directory for all data files.
"""

from __future__ import annotations
from utils.cli import parse_common_args
from utils.output_layout import infer_case_id
from pathlib import Path
from typing import Any
import os
import sys
import numpy as np
from fenics import *
from dataclasses import asdict
from CONFIGPenalty import (
    material,
    mesh,
    solver as solver_params,
    transient,
)
from macroscale.src.functions.macro_HMM_penalty_transient_EHL import (
    material_parameters,
    mesh_parameters,
    solver_parameters,
    meshfn,
    EHLSolver,
)

set_log_level(LogLevel.ERROR)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def append_line(path: Path, value: Any) -> None:
    """Append *value* (with trailing newline) to *path* with controlled formatting."""
    with path.open("a", encoding="utf-8") as f:
        if isinstance(value, float):
            f.write(f"{value:.16f}\n")  # fixed-point, 16 decimals
        elif isinstance(value, (list, tuple, np.ndarray)):
            f.write(
                " ".join(f"{v:.16f}" if isinstance(v, float) else str(v) for v in value)
                + "\n"
            )
        else:
            f.write(f"{value}\n")


def read_floats(path: Path) -> np.ndarray:
    """Return all whitespace‑separated floats contained in *path*."""
    return np.loadtxt(path, ndmin=1)


def read_last_force(path: Path) -> np.ndarray:
    """Read the last force vector stored in *path*."""
    last = path.read_text().strip().splitlines()[-1]
    return np.fromstring(last.replace("[", "").replace("]", ""), sep=" ").reshape(1, 3)



def _load_if_exists(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except OSError:
        return None

def _load_scalar_if_exists(path: Path) -> float | None:
    try:
        arr = np.load(path)
        return float(arr)
    except OSError:
        return None

def _save_scalar(path: Path, value: float) -> None:
    np.save(path, np.array(value, dtype=float))

def _mix(prev_used: np.ndarray | None, new: np.ndarray, omega: float) -> np.ndarray:
    """Under-relax: used = prev_used + omega*(new - prev_used). If no prev_used, use new."""
    if prev_used is None or prev_used.shape != new.shape:
        return new
    return prev_used + omega * (new - prev_used)

def _aitken_update_omega(
    omega_k: float,
    delta_k: np.ndarray,
    delta_km1: np.ndarray | None,
    omega_min: float,
    omega_max: float,
    eps: float = 1e-30,
) -> float:
    """
    Aitken Δ² for vector fixed-point iterations.
    Using formula:
      omega_{k+1} = - omega_k * <delta_k, delta_k - delta_{k-1}> / ||delta_k - delta_{k-1}||^2
    where delta_k = (new - used_prev) at iteration k.
    """
    if delta_km1 is None:
        return float(np.clip(omega_k, omega_min, omega_max))

    d = (delta_k - delta_km1).ravel()
    num = float(np.dot(delta_k.ravel(), d))
    den = float(np.dot(d, d))

    if den < eps:
        # No meaningful change -> keep omega
        return float(np.clip(omega_k, omega_min, omega_max))

    omega_kp1 = -omega_k * (num / den)
    return float(np.clip(omega_kp1, omega_min, omega_max))

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_common_args("Initialise Transient Macroscale Problem", with_time=True)
    lb_iter, c_iter, T, DT, output_dir = (
        args.lb_iter,
        args.c_iter,
        args.Time,
        args.DT,
        args.output_dir,
    )
    output_dir = Path(args.output_dir)  # make it a Path object
    output_dir.mkdir(parents=True, exist_ok=True)

    new_cycle = lb_iter == 0 and c_iter == 1

    step_idx = int(round(args.Time / args.DT))
    step_idx_last = int(round((args.Time - args.DT) / args.DT))

    prefix = f"T_{step_idx}_"

    if step_idx_last < 0:
        prefix_last = prefix
    else:
        prefix_last = f"T_{step_idx_last}_"

    # ------------------------------------------------------------------
    # Load previous state
    # ------------------------------------------------------------------
    # Fields written by the previous iteration/step
    p_last_T = np.load(os.path.join(output_dir, f"{prefix_last}p.npy"))
    deform_last_T = np.load(os.path.join(output_dir, f"{prefix_last}def.npy"))
    h_last_T = np.load(os.path.join(output_dir, f"{prefix_last}h.npy"))
    xi_prev = np.load(output_dir / "xi_rot_prev.npy")
    xi_last_T = np.load(os.path.join(output_dir, f"{prefix_last}xi.npy"))

    print(f"new cycle: {new_cycle}")
    if new_cycle:
        try:
            ecc_val = read_floats(output_dir / "d_eccentricity_out.txt")
        except OSError:
            print("d_eccentricity_out.txt not found; using eccentricities.txt")
            ecc = read_floats(output_dir / "eccentricities.txt")
            ecc_val = np.array([ecc[-1]])
    else:
        ecc = read_floats(output_dir / "eccentricities.txt")
        ecc_val = ecc[-1]
    ecc_val = np.asarray(ecc_val)

    if ecc_val.ndim == 0:
        last_ecc = np.array([0.0, 0.0, float(ecc_val)])
    else:
        last_ecc = np.array([0.0, 0.0, float(ecc_val[-1])])

    if new_cycle:
        (output_dir / "d_load_balance_err.txt").write_text("1\n")
        (output_dir / "lb_eccentricities.txt").write_text(f"{last_ecc[2]:.12f}\n")

    print(f"Last_ecc = {last_ecc}")

    # ------------------------------------------------------------------
    # Build solver with prior state data
    # ------------------------------------------------------------------
    solver = EHLSolver(
        meshfn(mesh_parameters(**asdict(mesh))),
        material_parameters(**asdict(material)),
        solver_parameters(**asdict(solver_params)),
        infer_case_id(args.output_dir),
    )

    solver.reinitialise_solver(eccentricity=last_ecc)

    # Always load the last converged time-step fields for transient terms.
    solver.load_state(p_last_T, deform_last_T, h=h_last_T, time=T, dt=DT)

    # Choose initial guesses for the nonlinear solve.
    if c_iter == 1:
        p_guess = p_last_T
        deform_guess = deform_last_T
        h_guess = h_last_T
        print("Initial guess source: last converged transient step")
    else:
        p_guess = np.load(output_dir / "p_init.npy")
        deform_guess = np.load(output_dir / "def_init.npy")
        h_guess = np.load(output_dir / "h_init.npy")
        print("Initial guess source: previous coupling iteration in current load-balance")
    # p_guess = p_last_T
    # deform_guess = deform_last_T
    # h_guess = h_last_T
    solver.p.vector()[:] = p_guess
    solver.delta.vector()[:] = deform_guess
    solver.h.vector()[:] = h_guess

    # ------------------------------------------------------------------
    # Either load‑balance or coupling solve
    # ------------------------------------------------------------------
    if c_iter == 1 and lb_iter == 0:
        print("Starting initial load-balance solve…")
        solver.update_contact_separation(
            solver.material_properties.eccentricity0,
            HMMState=False,
            transientState=True,
            EHLState=True,
        )
        _ = solver.solve_loadbalance_EHL(HMMState=False, transientState=True)

    elif c_iter == 1 and lb_iter >= 1:
        print("Starting smooth solve with new eccentricity before HMM solve")
        solver.initialise_velocity()
        solver.update_contact_separation(
            solver.material_properties.eccentricity0,
            HMMState=False,
            transientState=True,
            EHLState=True,
        )
        print(
            f"Solving HMM with eccentricity: {solver.material_properties.eccentricity0[2]:.12f}"
        )
        xi, load_balance_err = solver.EHL_balance_equation(
            solver.material_properties.eccentricity0[2],
            HMMState=False,
            transientState=True,
        )

    else:
        print("Starting HMM coupling solve…")
        macro_only = os.getenv("MACRO_ONLY") == "1"
        if macro_only:
            n = len(p_guess)
            dQx = np.zeros(n)
            dQy = np.zeros(n)
            dP = np.zeros(n)
            taustx = np.zeros(n)
            tausty = np.zeros(n)
        else:
            print("Loading correction terms...")

            # --- new micro corrections from this coupling iteration ---
            dQx_new = np.load(output_dir / "dQx.npy")
            dQy_new = np.load(output_dir / "dQy.npy")
            dP_new  = np.load(output_dir / "dP.npy")

            taustx_new = np.load(output_dir / "taustx.npy")
            tausty_new = np.load(output_dir / "tausty.npy")

            pmax_new = np.load(output_dir / "pmax.npy")
            pmin_new = np.load(output_dir / "pmin.npy")
            hmax_new = np.load(output_dir / "hmax.npy")
            hmin_new = np.load(output_dir / "hmin.npy")

            # --- load previous "used" corrections (what we actually applied last time) ---
            dQx_used_prev = _load_if_exists(output_dir / "dQx_used.npy")
            dQy_used_prev = _load_if_exists(output_dir / "dQy_used.npy")
            dP_used_prev  = _load_if_exists(output_dir / "dP_used.npy")

            taustx_used_prev = _load_if_exists(output_dir / "taustx_used.npy")
            tausty_used_prev = _load_if_exists(output_dir / "tausty_used.npy")

            pmax_used_prev = _load_if_exists(output_dir / "pmax_used.npy")
            pmin_used_prev = _load_if_exists(output_dir / "pmin_used.npy")
            hmax_used_prev = _load_if_exists(output_dir / "hmax_used.npy")
            hmin_used_prev = _load_if_exists(output_dir / "hmin_used.npy")

            # --- Aitken state from disk (scalar omega and previous delta) ---
            # We compute omega using dP only, and apply that omega to everything.
            omega_path = output_dir / "coupling_omega.npy"
            delta_path = output_dir / "coupling_delta_dP.npy"

            # defaults / bounds (tune via env vars)
            omega_init = float(os.getenv("COUPLING_OMEGA_INIT", "0.3"))
            omega_min  = float(os.getenv("COUPLING_OMEGA_MIN",  "0.05"))
            omega_max  = float(os.getenv("COUPLING_OMEGA_MAX",  "0.8"))

            omega_k = _load_scalar_if_exists(omega_path)
            if omega_k is None:
                omega_k = omega_init

            delta_km1 = _load_if_exists(delta_path)

            # Optional hard reset (e.g. at start of LB iteration)
            if os.getenv("COUPLING_AITKEN_RESET", "0") == "1":
                delta_km1 = None
                omega_k = omega_init

            # --- form delta_k using dP (delta_k = new - used_prev) ---
            if dP_used_prev is None or dP_used_prev.shape != dP_new.shape:
                # no previous "used" state -> cannot Aitken update; just accept new and persist
                omega_used = float(np.clip(omega_k, omega_min, omega_max))
                dQx = dQx_new
                dQy = dQy_new
                dP  = dP_new
                taustx = taustx_new
                tausty = tausty_new
                pmax = pmax_new
                pmin = pmin_new
                hmax = hmax_new
                hmin = hmin_new

                # delta_k for next time (still store something sensible)
                delta_k = (dP_new - dP_new)  # zeros
            else:
                delta_k = (dP_new - dP_used_prev)

                # Aitken update omega_{k+1} based on delta_k and delta_{k-1}
                omega_next = _aitken_update_omega(
                    omega_k=omega_k,
                    delta_k=delta_k,
                    delta_km1=delta_km1,
                    omega_min=omega_min,
                    omega_max=omega_max,
                )

                # Use the UPDATED omega (common practice: compute omega_{k+1}, apply it immediately)
                omega_used = omega_next

                # --- apply the same omega to all correction arrays ---
                dQx = _mix(dQx_used_prev, dQx_new, omega_used)
                dQy = _mix(dQy_used_prev, dQy_new, omega_used)
                dP  = _mix(dP_used_prev,  dP_new,  omega_used)

                taustx = _mix(taustx_used_prev, taustx_new, omega_used)
                tausty = _mix(tausty_used_prev, tausty_new, omega_used)

                pmax = _mix(pmax_used_prev, pmax_new, omega_used)
                pmin = _mix(pmin_used_prev, pmin_new, omega_used)
                hmax = _mix(hmax_used_prev, hmax_new, omega_used)
                hmin = _mix(hmin_used_prev, hmin_new, omega_used)

            # --- persist "used" corrections for next coupling iteration ---
            np.save(output_dir / "dQx_used.npy", dQx)
            np.save(output_dir / "dQy_used.npy", dQy)
            np.save(output_dir / "dP_used.npy",  dP)

            np.save(output_dir / "taustx_used.npy", taustx)
            np.save(output_dir / "tausty_used.npy", tausty)

            np.save(output_dir / "pmax_used.npy", pmax)
            np.save(output_dir / "pmin_used.npy", pmin)
            np.save(output_dir / "hmax_used.npy", hmax)
            np.save(output_dir / "hmin_used.npy", hmin)

            # --- persist Aitken state for next time ---
            # store omega used and the current delta_k for dP
            _save_scalar(omega_path, omega_used)
            np.save(delta_path, delta_k)

            if os.getenv("COUPLING_AITKEN_DIAG", "0") == "1":
                n_delta = float(np.linalg.norm(delta_k.ravel()))
                print(f"[AITKEN] omega_used={omega_used:.4f} (min={omega_min:.3f}, max={omega_max:.3f}) ||delta_dP||={n_delta:.3e}")

        solver.apply_corrections(
            (dQx, dQy, np.zeros_like(dQx)),
            (taustx, tausty, np.zeros_like(taustx)),
            dP,
            p_bounds=(pmax, pmin),
            h_bounds=(hmax, hmin),
        )
        solver.export("dQ", tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)
        solver.export("dP", tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)
        solver.export("taust_rot", tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)
        solver.export('hmin', tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)
        solver.export('hmax', tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)
        solver.export('pmax', tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)
        solver.export('pmin', tag="COUPLING", iter=c_iter, lbiter=lb_iter, T=T)

        # solver.solver_params.Rnewton_relaxation_parameter = 0.2
        solver.initialise_velocity()
        solver.update_contact_separation(
            solver.material_properties.eccentricity0,
            HMMState=True,
            transientState=True,
            EHLState=True,
        )
        print(
            f"Solving HMM with eccentricity: {solver.material_properties.eccentricity0[2]:.12f}"
        )
        # ADDED 14/01/26
        xi, load_balance_err = solver.EHL_balance_equation(
            solver.material_properties.eccentricity0[2],
            HMMState=True,
            transientState=True,
        )
        for field in ("p", "h", "delta"):
            solver.export(field, tag="coupling_cont", iter=c_iter, lbiter=lb_iter, T=T)

    # ------------------------------------------------------------------
    # Post‑processing
    # ------------------------------------------------------------------
    xi_rot_array = np.asarray(solver.rotate_xi())
    solver.calcQ()
    solver.calc_gradP()

    xi_out = solver.construct_transient_xi(xi_rot_array, xi_last_T)
    print(f"Shape of xi_out: {np.shape(xi_out)}")

    try:
        load_balance_errs = read_floats(output_dir / "d_load_balance_err.txt")
    except OSError:
        load_balance_errs = np.array([1.0])

    last_force = read_last_force(output_dir / "forces.txt")

    # Verbose diagnostics (kept from original script)
    print(f"last_load_balance_err = {load_balance_errs[-1]}")
    print(f"last_force = {last_force}")

    p_max = np.max(solver.p.vector()[:])
    diff_z = solver.load[2] + solver.force[2]
    denom = abs(solver.load[2]) if abs(solver.load[2]) > 0 else 1.0
    load_balance_err = float(diff_z) / denom

    print(f"P_max for T={T}, lb_iter={lb_iter}, c_iter={c_iter} = {p_max}")

    # Export for visualisation
    for field in ("p", "Q", "h"):
        solver.export(field, tag="init", iter=lb_iter)

    # Save state for coupling iteration continuity
    np.save(output_dir / "p_init.npy", solver.p.vector()[:])
    np.save(output_dir / "def_init.npy", solver.delta.vector()[:])
    np.save(output_dir / "h_init.npy", solver.h.vector()[:])

    np.save(output_dir / "xi_rot.npy", xi_out)  # Saving the xi for the next micro run

    # ------------------------------------------------------------------
    # Coupling error
    # ------------------------------------------------------------------
    if c_iter != 1:
        p0_path = output_dir / "p0.npy"
        if p0_path.exists():
            p0 = np.load(p0_path)
        else:
            print("p0.npy not found; reconstructing from xi_rot_prev.npy")
            p0 = xi_prev[1, :] + DT * xi_prev[12, :]
        p1 = xi_out[1, :] + DT * xi_out[12, :]
        d_coupling_err = np.linalg.norm(p1 - p0) / np.linalg.norm(p0)
        print(
            f"DT*xi_out[12, :] norm = {np.linalg.norm(DT * xi_out[12, :]):.12e}, p norm = {np.linalg.norm(xi_out[1, :]):.12e}"
        )
        print(
            f"p0 norm = {np.linalg.norm(p0):.12e}, p1 norm = {np.linalg.norm(p1):.12e}, p1-p0 norm = {np.linalg.norm(p1 - p0):.12e}"
        )
        np.save(p0_path, p1)
    else:
        p1 = xi_out[1, :] + DT * xi_out[12, :]
        d_coupling_err = 1
        print(
            f"coupling iter 1 DT*xi_out[12, :] norm = {np.linalg.norm(DT * xi_out[12, :]):.12e}, p norm = {np.linalg.norm(xi_out[1, :]):.12e}, p1 norm = {np.linalg.norm(p1):.12e}"
        )
        np.save(output_dir / "p0.npy", p1)

    append_line(output_dir / "d_coupling_errs.txt", d_coupling_err)

    print("Coupling error = %.3e", d_coupling_err)
    print("Load balance error = %.3e", load_balance_err)
    print("force          : %s", solver.force)
    print("last_force     : %s", last_force)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    append_line(output_dir / "forces.txt", solver.force)

    # -- coupling converged -------------------------------------------
    if abs(d_coupling_err) < transient.coupling_tol:
        for field in ("p", "Q", "h", "dQ", "dP"):
            solver.export(field, tag="COUPLING", iter=lb_iter)
        print(
            "Coupling convergence achieved (%.3e) [lb_iter=%d, c_iter=%d, T=%d]",
            d_coupling_err,
            lb_iter,
            c_iter,
            T,
        )

        # -- load‑balance convergence ----------------------------------
        if abs(load_balance_err) < transient.load_balance_tol:
            for func in ("p", "Q", "h", "dQ", "dP", "pmax", "pmin", "hmax","hmin"):
                solver.export_series(func, T)
                print(f"Exporting {func} for T={T}, lb_iter={lb_iter}, c_iter={c_iter}")
                # solver.export(func, tag="Transient", iter=c_iter, lbiter=lb_iter, T=T)
            solver.calc_shear_stress()
            solver.calc_friction()
            print(f"Macro only friction Coefficient : {solver.dim_friction}")
            solver.calc_hom_friction()
            print(f"Multiscale friction Coefficient : {solver.friction_coeff}")
            append_line(output_dir / "d_friction.txt", solver.friction_coeff)
            append_line(output_dir / "d_friction_macro.txt", solver.dim_friction)
            append_line(
                output_dir / "d_eccentricity.txt",
                solver.material_properties.eccentricity0,
            )
            append_line(
                output_dir / "d_eccentricity_out.txt",
                solver.material_properties.eccentricity0[2],
            )
            append_line(
                output_dir / "eccentricities.txt",
                solver.material_properties.eccentricity0[2],
            )
            append_line(output_dir / "d_load_balance_err.txt", load_balance_err)
            print(
                "Load balance convergence achieved (%.3e) [lb_iter=%d, c_iter=%d, T=%d]",
                load_balance_err,
                lb_iter,
                c_iter,
                T,
            )
            # np.save(output_dir / "xi_last_T.npy", xi_out)

            np.save(
                os.path.join(output_dir, f"{prefix}xi.npy"), xi_rot_array
            )  # update to store actual current time step final xi values
            np.save(os.path.join(output_dir, f"{prefix}h.npy"), solver.h.vector()[:])
            np.save(os.path.join(output_dir, f"{prefix}p.npy"), solver.p.vector()[:])
            np.save(
                os.path.join(output_dir, f"{prefix}def.npy"), solver.delta.vector()[:]
            )

        # -- load‑balance NOT converged -------------------------------
        else:

            print(
                "Load balance convergence NOT achieved (%.3e) [lb_iter=%d, c_iter=%d, T=%d]",
                load_balance_err,
                lb_iter,
                c_iter,
                T,
            )
            if lb_iter < 2:  # lb_iter reset to 1 at the start of each time step
                print(
                    f"Updating eccentricity for lb_iter={lb_iter:2d} using scaling of load balance"
                )
                new_ecc = (
                    solver.material_properties.eccentricity0[2]
                    * (1 + load_balance_err * transient.scaling_factor)
                    * solver.material_properties.Rc
                    / solver.material_properties.c
                )
            else:
                print(
                    f"Updating eccentricity for lb_iter={lb_iter:2d} using secant method"
                )
                load_balance_ecc_history = read_floats(
                    output_dir / "lb_eccentricities.txt"
                )
                print(
                    f"lb_iter={lb_iter:2d}  ecc_in={solver.material_properties.eccentricity0[2]:.6e}  "
                    f"err={load_balance_err:+.3e}  "
                    f"ecc_last={load_balance_ecc_history[-2]:.15e}  "
                    f"eccentricity0={solver.material_properties.eccentricity0[2]:.6e}  "
                    f"Δecc={solver.material_properties.eccentricity0[2]-load_balance_ecc_history[-2]}  "
                    f"Δerr={load_balance_err-load_balance_errs[-1]:+.3e}"
                )

                new_ecc = (
                    (
                        solver.material_properties.eccentricity0[2]
                        - load_balance_err
                        * (
                            solver.material_properties.eccentricity0[2]
                            - load_balance_ecc_history[-2]
                        )
                        / (load_balance_err - load_balance_errs[-1])
                    )
                    * solver.material_properties.Rc
                    / solver.material_properties.c
                )

            solver.material_properties.eccentricity0[2] = (
                new_ecc * solver.material_properties.c / solver.material_properties.Rc
            )

            print("Updated eccentricity       : %.6f", new_ecc)
            print(
                "Updated eccentricity (normalised): %.6f",
                solver.material_properties.eccentricity0[2]
                * solver.material_properties.Rc
                / solver.material_properties.c,
            )
            append_line(output_dir / "d_load_balance_err.txt", load_balance_err)
            append_line(
                output_dir / "lb_eccentricities.txt",
                new_ecc * solver.material_properties.c / solver.material_properties.Rc,
            )
            append_line(
                output_dir / "eccentricities.txt",
                new_ecc * solver.material_properties.c / solver.material_properties.Rc,
            )

    # -- coupling NOT converged ---------------------------------------
    else:
        append_line(
            output_dir / "eccentricities.txt",
            solver.material_properties.eccentricity0[2],
        )
        print(
            "Coupling convergence NOT achieved (%.3e) [lb_iter=%d, c_iter=%d, T=%d]",
            d_coupling_err,
            lb_iter,
            c_iter,
            T,
        )

    print("transient_macro_init.py completed successfully.")


if __name__ == "__main__":
    main()
