#!/usr/bin/env python3
"""Apply MLS corrections to the macroscale model and test convergence.

The script consumes predictions from :mod:`4_run_MLS` (``dQx.npy``,
``dP.npy`` …) located in ``--output_dir``.  It reloads the previous macro
state from ``p_init.npy`` and updates the solver with the MLS corrections.
After solving, it records diagnostic errors in ``coupling_error.txt`` and
``load_balance_err.txt`` and writes the updated fields back to
``p_init.npy``, ``Q_init.npy`` etc. for the next iteration.
"""

import os
import sys
from utils.cli import parse_common_args
from utils.output_layout import infer_case_id
import numpy as np
from fenics import *

set_log_level(LogLevel.ERROR)

from dataclasses import asdict
from CONFIGPenalty import (
    material,
    mesh,
    solver as solver_params,
    STEADY_COUPLING_TOL,
    STEADY_LOAD_BALANCE_TOL,
    STEADY_SCALING_FACTOR,
)

from macroscale.src.functions.macro_HMM_penalty_transient_EHL import (
    material_parameters,
    mesh_parameters,
    solver_parameters,
    meshfn,
    EHLSolver,
)

# ----------------------------------------------------------------


def main():
    """Load corrections, solve the macroscale problem and log errors."""
    args = parse_common_args("Perform coupling iteration")

    lb_iter = args.lb_iter
    c_iter = args.c_iter

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    coupling_tol = STEADY_COUPLING_TOL
    load_balance_tol = STEADY_LOAD_BALANCE_TOL
    scaling_factor = STEADY_SCALING_FACTOR

    # Reload the previous pressure field and correction terms from MLS.
    p_last = np.load(os.path.join(args.output_dir, "p_init.npy"))
    def_last = np.load(os.path.join(args.output_dir, "def_init.npy"))

    # dQx = np.load(os.path.join(args.output_dir, "dQx.npy"))
    # dQy = np.load(os.path.join(args.output_dir, "dQy.npy"))
    # taust_x = np.load(os.path.join(args.output_dir, "taustx.npy"))
    # taust_y = np.load(os.path.join(args.output_dir, "tausty.npy"))
    # pmax = np.load(os.path.join(args.output_dir, "pmax.npy"))
    # pmin = np.load(os.path.join(args.output_dir, "pmin.npy"))
    # hmax = np.load(os.path.join(args.output_dir, "hmax.npy"))
    # hmin = np.load(os.path.join(args.output_dir, "hmin.npy"))
    # dP = np.load(os.path.join(args.output_dir, "dP.npy"))

    # Loading code if doing 1:1 - pick up results straight from the micro sims
    dQ = np.load(os.path.join(args.output_dir, "dq_results.npy"))
    dQx = dQ[:, 0]
    dQy = dQ[:, 1]
    dQz = np.zeros_like(dQx)  # Assuming dQz is zero in this case
    dQ_r = np.column_stack((dQx, dQy, dQz))

    tau = np.load(os.path.join(args.output_dir, "tau_results.npy"))
    taust_x = tau[:, 0]
    taust_y = tau[:, 1]
    taust_z = np.zeros_like(taust_x)
    taust_r = np.column_stack((taust_x, taust_y, taust_z))

    dP = np.load(os.path.join(args.output_dir, "dp_results.npy"))
    pmax = np.load(os.path.join(args.output_dir, "pmax_results.npy"))
    pmin = np.load(os.path.join(args.output_dir, "pmin_results.npy"))
    hmax = np.load(os.path.join(args.output_dir, "hmax_results.npy"))
    hmin = np.load(os.path.join(args.output_dir, "hmin_results.npy"))

    # Retrieve history for eccentricity, load-balance errors and forces.
    with open(os.path.join(args.output_dir, "eccentricities.txt"), "r") as f:
        lines = f.readlines()
        all_eccentricities = np.array([float(line.strip()) for line in lines]).reshape(
            -1,
        )

    print(f"all_eccentricities =\n{all_eccentricities}")

    last_eccentricity = np.array([0, 0, all_eccentricities[-1]])
    print(f"last_eccentricity = {last_eccentricity}")

    with open(os.path.join(args.output_dir, "load_balance_err.txt"), "r") as f:
        lines = f.readlines()
        load_balance_errs = np.array([float(line.strip()) for line in lines]).reshape(
            -1,
        )

    print(f"load_balance_errs =\n{load_balance_errs}")
    print(f"last_load_balance_err = {load_balance_errs[-1]}")

    with open(os.path.join(args.output_dir, "forces.txt"), "r") as file:
        _forces = file.readlines()
    _last_force = _forces[-1].strip().replace("[", "").replace("]", "")
    last_force = np.fromstring(_last_force, sep=" ")
    last_force = last_force.reshape(1, 3)

    print(f"last_force = {last_force}")

    solver = EHLSolver(
        meshfn(mesh_parameters(**asdict(mesh))),
        material_parameters(**asdict(material)),
        solver_parameters(**asdict(solver_params)),
        infer_case_id(args.output_dir),
    )

    # solver.h_regularisation_param = (
    #     0  # Turn off regularisation for the coupling iteration
    # )
    # solver._reg_decay_S0 = 0
    # solver._reg_decay_S_current = 0

    print(
        f"percent_eccentricity = {last_eccentricity[2]*solver.material_properties.Rc/solver.material_properties.c}"
    )

    solver.reinitialise_solver(eccentricity=last_eccentricity)

    solver.apply_corrections(
        (dQx, dQy, np.zeros_like(dQx)),
        (taust_x, taust_y, np.zeros_like(taust_x)),
        dP,
        p_bounds=(pmax, pmin),
        h_bounds=(hmax, hmin),
    )

    solver.calc_hom_friction()
    solver.load_state(p_last, def_last)

    solver.update_contact_separation(
        solver.material_properties.eccentricity0,
        HMMState=True,
        transientState=False,
        EHLState=True,
    )

    # SOLVE
    xi, load_balance_err = solver.EHL_balance_equation(
        solver.material_properties.eccentricity0[2], HMMState=True, transientState=False
    )
    solver.calcQ()
    solver.calc_shear_stress()
    solver.calc_friction()
    solver.calc_hom_friction()

    for func in [
        "p",
        "Q",
        "dQ",
        "dP",
        "pmax",
        "pmin",
        "hmax",
        "hmin",
        "h",
    ]:
        solver.export(func, tag="Steady", iter=c_iter, lbiter=lb_iter)
        # solver.export_series(func, 0)

    # Check coupling convergence.
    # - If coupling has converged, check load balance
    # - If load balance has converged, store and end load balance iteration
    # - If load balance has not converged, update eccentricity and proceed
    force_err = np.linalg.norm(solver.force - last_force) / np.linalg.norm(last_force)
    print(f"Coupling error = {force_err}")
    print(f"Load balance error = {load_balance_err}")
    print(f"force = {solver.force}")
    print(f"last_force = {last_force}")
    with open(os.path.join(args.output_dir, "forces.txt"), "a") as f:
        f.write(f"{solver.force}\n")

    with open(os.path.join(args.output_dir, "coupling_error.txt"), "a") as f:
        f.write(f"{force_err}\n")

    if abs(force_err) < coupling_tol:
        print(
            f"Coupling convergence achieved with error: {force_err:.3e}, for lb_iter={lb_iter}, c_iter={c_iter}"
        )
        if abs(load_balance_err) < load_balance_tol:
            for func in [
                "p",
                "Q",
                "h",
                "dQ",
                "dP",
                "pmax",
                "pmin",
                "hmax",
                "hmin",
            ]:
                solver.export(func, tag="Steady", iter=c_iter, lbiter=lb_iter)
                solver.export_series(func, 0)
            with open(os.path.join(args.output_dir, "load_balance_err.txt"), "a") as f:
                f.write(f"{load_balance_err}\n")
            print(
                f"Load balance convergence achieved with error: {load_balance_err:.3e}, for lb_iter={lb_iter}, c_iter={c_iter}"
            )
        else:
            with open(os.path.join(args.output_dir, "load_balance_err.txt"), "a") as f:
                f.write(f"{load_balance_err}\n")
            print(
                f"Load balance convergence NOT achieved with error: {load_balance_err:.3e}, for lb_iter={lb_iter}, c_iter={c_iter}"
            )
            history = {
                "eccentricities": all_eccentricities,
                "load_balance_errs": load_balance_errs,
            }
            params = {"scaling_factor": scaling_factor}
            new_ecc, strategy = solver.update_eccentricity(
                load_balance_err, history, params
            )
            phys_ecc = (
                new_ecc * solver.material_properties.Rc / solver.material_properties.c
            )
            print(f"Updated eccentricity ({strategy}): {phys_ecc}")
            with open(os.path.join(args.output_dir, "eccentricities.txt"), "a") as f:
                f.write(f"{new_ecc}\n")
            solver.update_contact_separation(
                solver.material_properties.eccentricity0,
                HMMState=False,
                transientState=False,
                EHLState=True,
            )
            xi = solver.solve_loadbalance_secant(HMMState=False, transientState=False)
            xi, load_balance_err = solver.EHL_balance_equation(
                solver.material_properties.eccentricity0[2],
                HMMState=False,
                transientState=False,
            )
            solver.calcQ()

    else:
        print(
            f"Coupling convergence NOT achieved with error: {force_err:.3e}, for lb_iter={lb_iter}, c_iter={c_iter}"
        )

    # Prepare data for the next iteration.
    p_max_init = np.max(solver.p.vector()[:])
    force_init = solver.force
    step_idx = 0
    prefix = f"T_{step_idx}_"
    np.save(os.path.join(args.output_dir, "p_init.npy"), solver.p.vector()[:])
    np.save(os.path.join(args.output_dir, "def_init.npy"), solver.delta.vector()[:])
    np.save(os.path.join(args.output_dir, "Q_init.npy"), solver.Q.vector()[:])
    np.save(os.path.join(args.output_dir, "h_init.npy"), solver.h.vector()[:])

    np.save(os.path.join(args.output_dir, f"{prefix}p.npy"), solver.p.vector()[:])
    np.save(os.path.join(args.output_dir, f"{prefix}h.npy"), solver.h.vector()[:])
    np.save(os.path.join(args.output_dir, f"{prefix}def.npy"), solver.delta.vector()[:])

    xi_rot = solver.rotate_xi()

    np.save(os.path.join(args.output_dir, "xi_rot.npy"), xi_rot)
    np.save(os.path.join(args.output_dir, "xi_rot_prev.npy"), xi_rot)
    np.save(os.path.join(args.output_dir, f"{prefix}xi.npy"), xi_rot)
    print("Initialisation xi values saved to xi_rot_prev.npy")

    print("macro_solve.py completed successfully.")


if __name__ == "__main__":
    main()
