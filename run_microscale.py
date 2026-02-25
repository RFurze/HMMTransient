"""Execute microscale tasks in parallel and save the results to disk.

This version uses Python's multiprocessing (spawn) to distribute the input
task list over N worker processes (default: 12), execute tasks concurrently,
and then write results back out **in the same order as the original tasks**.

It is intended as a drop-in replacement for the MPI futures runner, but can be
launched simply as:

    python3 run_microscale.py

You can override the process count via either:
- environment variable: MICRO_NPROCS
- or, if your parse_common_args() provides it, --num-procs / --nprocs / --n_procs
  (auto-detected by attribute name).

Notes:
- We avoid importing FEniCS (dolfin) at module import time. Each worker imports
  dolfin lazily to keep multiprocessing spawn safe.
- Each worker keeps a per-process mesh cache to avoid re-creating meshes for
  every task.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from functools import partial
import multiprocessing as mp
from typing import Any, Dict, Tuple

# -----------------------------------------------------------------------------
# Threading guards (important when running many processes)
# -----------------------------------------------------------------------------
# Set these early so they apply inside worker imports too.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Optional periodic full-GC frequency (0 means disabled).
os.environ.setdefault("MICRO_GC_EVERY", "100")


import numpy as np


def _get_env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _compute_chunksize(n_tasks: int, n_procs: int) -> int:
    """Pick a reasonable Pool chunksize (override with MICRO_CHUNKSIZE)."""
    env_chunk = _get_env_int("MICRO_CHUNKSIZE", 0)
    if env_chunk > 0:
        return env_chunk
    # Heuristic: keep ~50 chunks per worker overall (i.e. small-ish chunks)
    # to balance task-time variability while keeping IPC overhead reasonable.
    return max(1, n_tasks // max(1, n_procs * 50))

def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

from utils.cli import parse_common_args
from CONFIGPenalty import micro_physical, micro_solver, ideal_film_thickness

from microscale.src.functions.micro_mixed_transient_class import (
    MicroMixedSolver,
    MicroPhysicalParameters,
    MicroSolverSettings,
)
# -----------------------------------------------------------------------------
# Per-worker mesh cache (avoid re-building FEniCS meshes per task)
# -----------------------------------------------------------------------------
# _MESH_CACHE = None


# def _get_worker_meshes():
#     """Create / return meshes local to this worker process."""
#     global _MESH_CACHE
#     if _MESH_CACHE is None:
#         # Import inside worker to be spawn-safe.
#         from dolfin import UnitSquareMesh

#         # Match mesh creation in micro_mixed_classNonDim.py main()
#         n_m = _get_env_int('MICRO_MESH_N_M', _get_env_int('MICRO_MESH_N', 150))
#         n_h = _get_env_int('MICRO_MESH_N_H', _get_env_int('MICRO_MESH_N', 60))
#         _MESH_CACHE = (UnitSquareMesh(n_m, n_m), UnitSquareMesh(n_h, n_h))
#     return _MESH_CACHE

def _get_worker_meshes():
    """Create / return meshes local to this worker process."""
    # Import inside worker to be spawn-safe.
    from dolfin import UnitSquareMesh

    # Match mesh creation in micro_mixed_classNonDim.py main()
    n_m = _get_env_int('MICRO_MESH_N_M', _get_env_int('MICRO_MESH_N', 40))
    n_h = _get_env_int('MICRO_MESH_N_H', _get_env_int('MICRO_MESH_N', 40))
    return (UnitSquareMesh(n_m, n_m), UnitSquareMesh(n_h, n_h))

# -----------------------------------------------------------------------------
# Task parsing (new structure)
# -----------------------------------------------------------------------------
def _parse_task(task, transient: bool):
    if transient:
        (
            task_id,
            row_idx,
            H,
            P,
            Ux,
            Uy,
            _,
            gradp1,
            gradp2,
            _,
            _,
            _,
            _,
            Hdot,
            Pdot,
        ) = task
        return (
            int(task_id),
            int(row_idx),
            float(H),
            float(P),
            float(Ux),
            float(Uy),
            float(gradp1),
            float(gradp2),
            float(Hdot),
            float(Pdot),
        )
    else:
        (task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _, _, _, _) = task
        return (
            int(task_id),
            int(row_idx),
            float(H),
            float(P),
            float(Ux),
            float(Uy),
            float(gradp1),
            float(gradp2),
            0.0,
            0.0,
        )

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


def q_re(transient, H, P, Ux, Uy, gradpx, gradpy, Pst, Hdot=None, Pdot=None, dt=0.0):
    if transient:
        Hdt = H + Hdot * dt
        Pdt = P + Pdot * dt

    else:
        Hdt = H
        Pdt = P

    eta = roelands(Pst)
    qx = Ux * Hdt - (Hdt**3) * gradpx / (12 * eta)
    qy = Uy * Hdt -(Hdt**3) * gradpy / (12 * eta)
    q = np.column_stack((qx, qy))
    return q

def run_micro_task(task, transient: bool = False, macro_dt: float | None = None, export_vtk: bool = False):
    """Run one microscale task and return results in the legacy tuple format.

    IMPORTANT: This function must be top-level picklable for multiprocessing spawn.
    """
    start = time.time()

    # Import dolfin bits inside worker (spawn-safe).
    from dolfin import cos, pi

    mesh_m, mesh_h = _get_worker_meshes()

    task_id, row_idx, H, P, Ux, Uy, gradp1, gradp2, Hdot, Pdot = _parse_task(task, transient=transient)

    tend = ((macro_dt if (macro_dt is not None) else _get_env_float('MICRO_TEND', 0.05)) if transient else 0.0)

    def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        Ah = ideal_film_thickness.Ah
        kx = ideal_film_thickness.kx
        ky = ideal_film_thickness.ky

        x_d, y_d = x * xmax, y * ymax

        # python-side safety checks must use python floats (NOT UFL)
        H0_val = float(H0)
        HT_val = float(HT)
        T_val  = float(T)

        hmin0 = H0_val - Ah
        hmin1 = H0_val - Ah + HT_val * T_val

        if (hmin0 < 0.0) or (hmin1 < 0.0):
            print(f"Warning: Negative film thickness at T={T_val:.6e}s, hmin0={hmin0:.6e}, hmin1={hmin1:.6e}.")
            sys.stdout.flush()

        # return UFL expression (this is what the solver needs)
        return H0 + HT * T + 0.5 * Ah * (cos(kx * 2 * pi * x_d / xmax) + cos(ky * 2 * pi * y_d / ymax))



    # Physical + solver parameters
    phys = MicroPhysicalParameters(
        Ux=Ux,
        Uy=Uy,
        eta0=micro_physical.eta0,
        rho0=micro_physical.rho0,
        penalty_gamma=getattr(micro_physical, "penalty_gamma", 1e4),
        xmax=micro_physical.xmax,
        ymax=micro_physical.ymax,
        p0=P,
        dpdx=gradp1,
        dpdy=gradp2,
        H0=H,
        h_min=0.0,  # Mixed masking disabled
        HT=Hdot,
        PT=Pdot,
        Tend=tend,
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

    solver = MicroMixedSolver(
        mesh_m=mesh_h,  # Not doing mixed yet so currently locked in to coarse hydro mesh
        mesh_h=mesh_h,
        physical_params=phys,
        solver_settings=settings,
        k=1,
        ht=ht,
        export_vtk=export_vtk,
        output_dir=os.path.join(os.getcwd(), "output"),
        auto_solve=False,
    )


    if transient:
        # Not enabled yet (BDF stepping will be added next in the solver).
        solver.run(Tf=tend, n_steps=_get_env_int('MICRO_NSTEPS', 8))
        # solver.post_process()
    else:
        solver.run_steady()

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

    Q = np.column_stack((Qx, Qy))
    tau_res = np.array([taust_x, taust_y])

    # Reference values for the metamodel increment:
    Q_re = q_re(transient, H, P, Ux, Uy, gradp1, gradp2, Pst, Hdot=Hdot, Pdot=Pdot, dt=tend)
    if transient:
        P_re = (P + Pdot * tend)
    else:
        P_re = P

    delta_Q = Q - Q_re
    delta_P = Pst - P_re
    
    # Clean up as aggressively as possible (helps long runs)
    del solver, phys, settings
    gc_every = _get_env_int("MICRO_GC_EVERY", 0)
    if gc_every > 0 and (task_id % gc_every == 0):
        gc.collect()

    return (
        task_id,
        delta_Q,
        delta_P,
        Pst,
        Q,
        Fst,
        tau_res,
        p_max,
        p_min,
        max_h,
        min_h,
        fmax,
        fmin,
        task,
        time.time() - start,
        os.getpid(),
    )


def run_micro_task_indexed(indexed_task, transient: bool = False, macro_dt: float | None = None, export_vtk: bool = False):
    """Wrapper around run_micro_task that preserves the original task index.

    Using imap_unordered improves worker utilisation when task runtimes vary.
    We return the index so the parent can store outputs in the original order.
    """
    i, task = indexed_task
    res = run_micro_task(task, transient=transient, macro_dt=macro_dt, export_vtk=export_vtk)
    return (int(i),) + tuple(res)

# def run_micro_task(task, transient: bool = False, macro_dt: float | None = None, export_vtk: bool = False):

#     task_id, row_idx, H, P, Ux, Uy, gradp1, gradp2, Hdot, Pdot = _parse_task(task, transient=transient)
   
#     return (
#         task_id,
#         np.column_stack((0, 0)),
#         0,
#         P,
#         np.column_stack((0, 0)),
#         0,
#         np.column_stack((0, 0)),
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         task,
#         0,
#         os.getpid(),
#     )

def _infer_num_procs(args) -> int:
    """Try to infer nprocs from args; fall back to env MICRO_NPROCS; then 12."""
    for attr in ("num_procs", "nprocs", "n_procs", "processes", "num_processes"):
        if hasattr(args, attr):
            try:
                v = int(getattr(args, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    try:
        v = int(os.environ.get("MICRO_NPROCS", "24"))
        return v if v > 0 else 24
    except Exception:
        return 24


def main():
    args = parse_common_args("Run Microscale Simulations", with_time=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tasks = np.load(os.path.join(output_dir, "tasks.npy"), allow_pickle=True)

    transient = bool(args.transient)
    macro_dt = args.DT if transient else None
    print(f'dt = {macro_dt}')

    N_tasks = len(tasks)

    # Time-based progress reporting (seconds). Override with MICRO_PROGRESS_SEC env var.
    progress_every = float(os.environ.get("MICRO_PROGRESS_SEC", "300"))
    next_progress_t = None  # set after start_time is known

    num_procs = _infer_num_procs(args)
    chunk_size = _compute_chunksize(N_tasks, num_procs)

    print(
        f"Loaded {N_tasks} tasks. transient={transient}. "
        f"Using {num_procs} process(es), chunksize={chunk_size}."
    )
    sys.stdout.flush()

    start_time = time.time()
    next_progress_t = progress_every
    last_completed = 0
    last_elapsed = 0.0

    task_func = partial(run_micro_task_indexed, transient=transient, macro_dt=macro_dt, export_vtk=False)

    # Preallocate outputs in ORIGINAL TASK ORDER (index order)
    dq_results = np.zeros((N_tasks, 2))
    dp_results = np.zeros(N_tasks)
    PST_out_arr = np.zeros(N_tasks)
    Q_outs = np.zeros((N_tasks, 2))
    Fst_results = np.zeros(N_tasks)
    tau_results = np.zeros((N_tasks, 2))
    pmax_results = np.zeros(N_tasks)
    pmin_results = np.zeros(N_tasks)
    max_h_results = np.zeros(N_tasks)
    min_h_results = np.zeros(N_tasks)
    fmax_results = np.zeros(N_tasks)
    fmin_results = np.zeros(N_tasks)

    # Optional: track which PID handled which task index (useful debug)
    pid_of_task = np.zeros(N_tasks, dtype=int)

    # Use spawn context (matches your parallel_run_test.py)
    ctx = mp.get_context("spawn")

    completed = 0
    with ctx.Pool(processes=num_procs, maxtasksperchild=50) as pool:
        # Wrap tasks with their original index so we can use imap_unordered for better utilisation.
        indexed_tasks = list(enumerate(tasks))
        for res in pool.imap_unordered(task_func, indexed_tasks, chunksize=chunk_size):
            i, res = res[0], res[1:]
            (
                task_id,
                dq_result,
                dp_result,
                PST_out,
                Q_out,
                Fst,
                tau_res,
                pmax,
                pmin,
                max_h,
                min_h,
                fmax,
                fmin,
                task,
                wall_time,
                pid,
            ) = res

            if np.isnan(dq_result).any() or np.isnan(dp_result).any():
                print(f"NaN detected in task index {i} (task_id={task_id}): Task details: {task}")
                sys.stdout.flush()

            dq_results[i] = dq_result
            dp_results[i] = dp_result
            PST_out_arr[i] = PST_out
            Q_outs[i] = Q_out
            Fst_results[i] = Fst
            tau_results[i] = tau_res
            pmax_results[i] = pmax
            pmin_results[i] = pmin
            max_h_results[i] = max_h
            min_h_results[i] = min_h
            fmax_results[i] = fmax
            fmin_results[i] = fmin
            pid_of_task[i] = pid

            completed += 1
            elapsed = time.time() - start_time
            # ---- TIME-GATED PROGRESS PRINT (every progress_every seconds) ----
            if (elapsed >= next_progress_t) or (completed == N_tasks):
                pct = 100.0 * completed / N_tasks

                dt_window = max(1e-9, elapsed - last_elapsed)
                rate_window = (completed - last_completed) / dt_window
                rate_overall = completed / max(1e-9, elapsed)

                remaining = N_tasks - completed
                eta = remaining / max(1e-9, rate_overall)

                print(
                    f"[PROGRESS] {completed}/{N_tasks} ({pct:5.1f}%) "
                    f"elapsed={_fmt_hms(elapsed)} "
                    f"rate={rate_window:.2f}/s (avg {rate_overall:.2f}/s) "
                    f"ETA={_fmt_hms(eta)}"
                )
            sys.stdout.flush()
            # Advance next report time (bucketed, avoids print spam after long stalls)
            next_progress_t = (int(elapsed // progress_every) + 1) * progress_every
            last_completed = completed
            last_elapsed = elapsed

    # Save (matching legacy filenames)
    np.save(os.path.join(output_dir, "dq_results.npy"), dq_results)
    np.save(os.path.join(output_dir, "dp_results.npy"), dp_results)
    if transient:
        np.save(os.path.join(output_dir, "Q_out.npy"), Q_outs)
    np.save(os.path.join(output_dir, "Fst.npy"), Fst_results)
    np.save(os.path.join(output_dir, "tau_results.npy"), tau_results)
    np.save(os.path.join(output_dir, "pmax_results.npy"), pmax_results)
    np.save(os.path.join(output_dir, "pmin_results.npy"), pmin_results)
    np.save(os.path.join(output_dir, "hmax_results.npy"), max_h_results)
    np.save(os.path.join(output_dir, "hmin_results.npy"), min_h_results)

    # Optional debug artifact (similar to your dummy code)
    try:
        np.savetxt(
            os.path.join(output_dir, "task_pid_map.csv"),
            np.column_stack([np.arange(N_tasks), pid_of_task]),
            delimiter=",",
            header="task_index,pid",
            comments="",
        )
    except Exception:
        pass

    elapsed_total = time.time() - start_time
    print(f"run_microscale.py completed successfully in {elapsed_total:.1f}s.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()