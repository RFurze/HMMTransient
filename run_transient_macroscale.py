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
    # if c_iter == 1:
    #     p_guess = p_last_T
    #     deform_guess = deform_last_T
    #     h_guess = h_last_T
    #     print("Initial guess source: last converged transient step")
    # else:
    #     p_guess = np.load(output_dir / "p_init.npy")
    #     deform_guess = np.load(output_dir / "def_init.npy")
    #     h_guess = np.load(output_dir / "h_init.npy")
    #     print("Initial guess source: previous coupling iteration in current load-balance")
    p_guess = p_last_T
    deform_guess = deform_last_T
    h_guess = h_last_T
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
            print(f"Loading correction terms...")
            # dQ = np.load(os.path.join(args.output_dir, "dq_results.npy"))
            # dQx = dQ[:, 0]
            # dQy = dQ[:, 1]
            # dP = np.load(os.path.join(args.output_dir, "dp_results.npy"))
            dQx = np.load(output_dir / "dQx.npy")
            dQy = np.load(output_dir / "dQy.npy")
            dP = np.load(output_dir / "dP.npy")
            # taustall = np.load(os.path.join(args.output_dir, "tau_results.npy"))
            # taustx = taustall[:, 0]
            # tausty = taustall[:, 1]
            taustx = np.load(output_dir / "taustx.npy")
            tausty = np.load(output_dir / "tausty.npy")
            pmax = np.load(os.path.join(args.output_dir, "pmax.npy"))
            pmin = np.load(os.path.join(args.output_dir, "pmin.npy"))
            hmax = np.load(os.path.join(args.output_dir, "hmax.npy"))
            hmin = np.load(os.path.join(args.output_dir, "hmin.npy"))

        solver.apply_corrections(
            (dQx, dQy, np.zeros_like(dQx)),
            (taustx, tausty, np.zeros_like(taustx)),
            dP,
            p_bounds=(pmax, pmin),
            h_bounds=(hmax, hmin),
        )
        solver.export("dQ", tag="COUPLING0", iter=lb_iter)
        solver.export("dP", tag="COUPLING0", iter=lb_iter)
        solver.export("taust_rot", tag="mid", iter=lb_iter)

        # solver.solver_params.Rnewton_relaxation_parameter = 0.2
        solver.initialise_velocity()
        solver.update_contact_separation(
            solver.material_properties.eccentricity0,
            HMMState=True,
            transientState=True,
            EHLState=True,  # ADDED 29/01/26
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
