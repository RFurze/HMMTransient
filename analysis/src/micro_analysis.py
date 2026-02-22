#!/usr/bin/env python3
"""
Microscale sensitivity studies (CSV outputs) for the *new* FEniCS-based microscale
runner/class:

- Uses run_microscale.py (multiprocessing spawn runner) AND/OR
  MicroMixedSolver in micro_mixed_transient_class.py directly.

Why two execution paths?
- The new run_microscale.py currently hard-codes its worker mesh sizes (n_m=150, n_h=60)
  and uses a fixed transient n_steps=20 internally. That makes classic "mesh study"
  and "tsteps study" impossible *without modifying run_microscale.py*.
- To keep the same studies + the same CSV outputs, this script runs tasks via:
    * "direct" path (default): calls MicroMixedSolver directly, so we can vary:
        - transient n_steps (proxy for old Tsteps)
        - mesh resolution (UnitSquareMesh(n,n))
    * "runner" path (optional): calls run_microscale.run_micro_task, matching your
      production execution more closely, but with fixed n_steps/mesh as per the runner.

Outputs match the legacy micro_timestep_analysis.py layout:
- meta.json
- selected_tasks.csv
- Per study:
    - <study>_raw__...csv
    - <study>_errors__..._vs_...csv
    - <study>_summary_vs_ref.csv (or percentiles for ahratio)

This script expects tasks.npy rows that match the *new* task structure used by
run_microscale._parse_task():

Transient task row:
    (task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _, Hdot, Pdot)

Steady task row:
    (task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _)

"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import multiprocessing as mp
ctx = mp.get_context("spawn")

# New runner + new solver class
import run_microscale_env_ahratio as run_microscale  # patched runner with env overrides
from CONFIGPenalty import micro_solver, ideal_film_thickness, micro_physical

from microscale.src.functions.micro_mixed_transient_class import (
    MicroMixedSolver,
    MicroPhysicalParameters,
    MicroSolverSettings,
)

# -------------------------------------------------------------------
# Task schema for new runner
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TaskIndex:
    task_id: int = 0
    row_idx: int = 1
    H: int = 2
    P: int = 3
    Ux: int = 4
    Uy: int = 5
    # 6: unused
    gradp1: int = 7
    gradp2: int = 8
    # 9: unused
    Hdot: int = 10
    Pdot: int = 11

    @property
    def min_cols_transient(self) -> int:
        return 12

    @property
    def min_cols_steady(self) -> int:
        return 10


def _ensure_2d_tasks(tasks: np.ndarray) -> np.ndarray:
    t = np.asarray(tasks)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    return t


def _validate_task_dim(tasks: np.ndarray, idx: TaskIndex) -> None:
    if tasks.shape[1] < idx.min_cols_steady:
        raise ValueError(f"tasks.npy has {tasks.shape[1]} columns, but expected >= {idx.min_cols_steady}.")


# -------------------------------------------------------------------
# Selection helpers
# -------------------------------------------------------------------

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def select_top_aggressive(
    tasks: np.ndarray,
    idx: TaskIndex,
    n_select: int,
    *,
    weights: Dict[str, float],
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the "most aggressive" tasks (likely to stress the microsolver).

    Score components (available in new task schema):
    - invH: 1/H
    - gmag: sqrt(gradp1^2 + gradp2^2)
    - Hdot: |Hdot|
    - Pdot: |Pdot|

    If robust=True, each component is normalized by a robust scale (median + IQR).
    """
    tasks = _ensure_2d_tasks(tasks)
    n = tasks.shape[0]
    H = tasks[:, idx.H].astype(float)
    g1 = tasks[:, idx.gradp1].astype(float)
    g2 = tasks[:, idx.gradp2].astype(float)

    invH = 1.0 / np.maximum(H, 1e-30)
    gmag = np.sqrt(g1 * g1 + g2 * g2)

    # Hdot/Pdot might not exist if tasks are steady-only; guard with NaNs
    Hdot = np.zeros(n, dtype=float)
    Pdot = np.zeros(n, dtype=float)
    if tasks.shape[1] > idx.Hdot:
        Hdot = np.abs(tasks[:, idx.Hdot].astype(float))
    if tasks.shape[1] > idx.Pdot:
        Pdot = np.abs(tasks[:, idx.Pdot].astype(float))

    comps = {
        "invH": invH,
        "gmag": gmag,
        "Hdot": Hdot,
        "Pdot": Pdot,
    }

    def robust_norm(x: np.ndarray) -> np.ndarray:
        x1 = x[np.isfinite(x)]
        if x1.size == 0:
            return np.zeros_like(x)
        med = np.median(x1)
        q1 = np.percentile(x1, 25)
        q3 = np.percentile(x1, 75)
        scale = max(1e-30, float(q3 - q1))
        return (x - med) / scale

    score = np.zeros(n, dtype=float)
    for k, x in comps.items():
        w = float(weights.get(k, 0.0))
        if w == 0.0:
            continue
        xn = robust_norm(x) if robust else x
        # shift so "larger is more aggressive"
        xn = np.where(np.isfinite(xn), xn, -np.inf)
        score += w * xn

    # Pick top n_select by score
    order = np.argsort(score)[::-1]
    sel_idx = order[: int(max(1, n_select))]
    return tasks[sel_idx, :], sel_idx


# -------------------------------------------------------------------
# CSV writers
# -------------------------------------------------------------------

def write_selected_tasks_csv(path: str, tasks: np.ndarray, idx: TaskIndex) -> None:
    tasks = _ensure_2d_tasks(tasks)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task_local_idx",
                "task_id",
                "row_idx",
                "H",
                "P",
                "Ux",
                "Uy",
                "gradp1",
                "gradp2",
                "Hdot",
                "Pdot",
            ]
        )
        for j in range(tasks.shape[0]):
            row = tasks[j]
            w.writerow(
                [
                    j,
                    int(row[idx.task_id]),
                    int(row[idx.row_idx]),
                    _safe_float(row[idx.H]),
                    _safe_float(row[idx.P]),
                    _safe_float(row[idx.Ux]),
                    _safe_float(row[idx.Uy]),
                    _safe_float(row[idx.gradp1]),
                    _safe_float(row[idx.gradp2]),
                    _safe_float(row[idx.Hdot]) if row.size > idx.Hdot else "",
                    _safe_float(row[idx.Pdot]) if row.size > idx.Pdot else "",
                ]
            )


def write_raw_outputs_csv(path: str, selected_indices: np.ndarray, out: Dict[str, np.ndarray]) -> None:
    dP = np.asarray(out["dP"], float)
    dQx = np.asarray(out["dQx"], float)
    dQy = np.asarray(out["dQy"], float)
    Pst = np.asarray(out["Pst"], float)
    Fst = np.asarray(out["Fst"], float)
    time_s = np.asarray(out["time_s"], float)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_local_idx", "task_global_idx", "dP", "dQx", "dQy", "Pst", "Fst", "time_s"])
        for j in range(len(dP)):
            w.writerow([j, int(selected_indices[j]), float(dP[j]), float(dQx[j]), float(dQy[j]), float(Pst[j]), float(Fst[j]), float(time_s[j])])


def _qmag(dQx: np.ndarray, dQy: np.ndarray) -> np.ndarray:
    return np.sqrt(dQx * dQx + dQy * dQy)


def _rel_err(abs_err: np.ndarray, ref: np.ndarray, eps: float) -> np.ndarray:
    denom = np.maximum(np.abs(ref), eps)
    return abs_err / denom


def _sym_rel_err(x: np.ndarray, ref: np.ndarray, eps: float) -> np.ndarray:
    denom = np.maximum(np.abs(x) + np.abs(ref), eps)
    return 2.0 * np.abs(x - ref) / denom


def summarize_errors_vs_ref(cur: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    cur = np.asarray(cur, float)
    ref = np.asarray(ref, float)
    err = np.abs(cur - ref)
    e = err[np.isfinite(err)]
    if e.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.percentile(e, 50)),
        "p90": float(np.percentile(e, 90)),
        "p99": float(np.percentile(e, 99)),
        "max": float(np.max(e)),
    }


def write_errors_vs_ref_csv(path: str, selected_indices: np.ndarray, cur: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> None:
    cur_dP = np.asarray(cur["dP"], float)
    ref_dP = np.asarray(ref["dP"], float)
    cur_qmag = _qmag(np.asarray(cur["dQx"], float), np.asarray(cur["dQy"], float))
    ref_qmag = _qmag(np.asarray(ref["dQx"], float), np.asarray(ref["dQy"], float))

    abs_eP = np.abs(cur_dP - ref_dP)
    abs_eQ = np.abs(cur_qmag - ref_qmag)

    ref_dP_fin = np.abs(ref_dP[np.isfinite(ref_dP)])
    ref_q_fin = np.abs(ref_qmag[np.isfinite(ref_qmag)])
    eps_rel_dP = max(1e-30, float(np.percentile(ref_dP_fin, 5)) if ref_dP_fin.size else 1e-30)
    eps_rel_dQ = max(1e-30, float(np.percentile(ref_q_fin, 5)) if ref_q_fin.size else 1e-30)

    rel_eP = _rel_err(abs_eP, ref_dP, eps_rel_dP)
    rel_eQ = _rel_err(abs_eQ, ref_qmag, eps_rel_dQ)

    sym_rel_eP = _sym_rel_err(cur_dP, ref_dP, eps_rel_dP)
    sym_rel_eQ = _sym_rel_err(cur_qmag, ref_qmag, eps_rel_dQ)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task_local_idx",
                "task_global_idx",
                "abs_err_dP",
                "abs_err_dQmag",
                "rel_err_dP",
                "rel_err_dQmag",
                "sym_rel_err_dP",
                "sym_rel_err_dQmag",
                "cur_dP",
                "ref_dP",
                "cur_dQmag",
                "ref_dQmag",
            ]
        )
        for j in range(len(cur_dP)):
            w.writerow(
                [
                    j,
                    int(selected_indices[j]),
                    float(abs_eP[j]),
                    float(abs_eQ[j]),
                    float(rel_eP[j]),
                    float(rel_eQ[j]),
                    float(sym_rel_eP[j]),
                    float(sym_rel_eQ[j]),
                    float(cur_dP[j]),
                    float(ref_dP[j]),
                    float(cur_qmag[j]),
                    float(ref_qmag[j]),
                ]
            )


def write_summary_csv(path: str, rows: List[Dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# -------------------------------------------------------------------
# Direct execution path (configurable n_steps + mesh)
# -------------------------------------------------------------------

def _make_ht_steady(Ah: float, kx: float, ky: float):
    """Steady signature ht(x,y,xmax,ymax,H0) -> UFL expr."""
    # Import inside so top-level import remains light-ish
    from dolfin import cos, pi
    def ht(x, y, xmax, ymax, H0):
        x_d, y_d = x * xmax, y * ymax
        return H0 + 0.5 * Ah * (cos(kx * 2 * pi * x_d / xmax) + cos(ky * 2 * pi * y_d / ymax))
    return ht


def run_micro_task_direct(
    task_row: Tuple[float, ...],
    *,
    ah_ratio: Optional[float] = None,
    transient: bool,
    n_steps: int,
    mesh_n: int,
    export_vtk: bool = False,
) -> Tuple[float, np.ndarray, float, float, np.ndarray, float, np.ndarray, float, float, float, float, float, float, Tuple[float, ...], float]:
    """
    Returns a tuple compatible with (subset of) run_microscale.run_micro_task output:
      (task_id, delta_Q, delta_P, Pst, Q, Fst, tau_res, pmax, pmin, max_h, min_h, fmax, fmin, task, wall_time)
    """
    t0 = time.perf_counter()

    from dolfin import UnitSquareMesh

    # Parse task row in the new format
    row = tuple(task_row)
    if transient:
        if len(row) < 12:
            raise ValueError(f"Transient task expects >=12 columns, got {len(row)}")
        task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _, Hdot, Pdot = row[:12]
    else:
        if len(row) < 10:
            raise ValueError(f"Steady task expects >=10 columns, got {len(row)}")
        task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _ = row[:10]
        Hdot = 0.0
        Pdot = 0.0

    task_id = int(task_id)
    H = float(H); P = float(P); Ux = float(Ux); Uy = float(Uy)
    gradp1 = float(gradp1); gradp2 = float(gradp2)
    Hdot = float(Hdot); Pdot = float(Pdot)

    # Build meshes (same mesh used for both args for now)
    mesh_m = UnitSquareMesh(int(mesh_n), int(mesh_n))
    mesh_h = UnitSquareMesh(int(mesh_n), int(mesh_n))

    tend = 0.05 if transient else 0.0

    Ah_base = float(getattr(ideal_film_thickness, "Ah", 0.0))
    # Per-task Ah override for Ah-ratio sweeps: Ah_task = ah_ratio * H_task
    Ah = float(ah_ratio) * float(H) if (ah_ratio is not None) else Ah_base
    kx = float(getattr(ideal_film_thickness, "kx", 1.0))
    ky = float(getattr(ideal_film_thickness, "ky", 1.0))
    ht = _make_ht_steady(Ah, kx, ky)

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
        h_min=0.0,
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
        mesh_m=mesh_h,  # consistent with run_microscale.py: "not doing mixed yet"
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
        solver.run(Tf=tend, n_steps=int(max(1, n_steps)))
        solver.post_process()
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
    tau_res = np.array([taust_x, taust_y], dtype=float)

    # Use the same reference definitions as run_microscale.py
    Q_re = run_microscale.q_re(transient, H, P, Ux, Uy, gradp1, gradp2, Pst, Hdot=Hdot, Pdot=Pdot, dt=tend)
    P_re = (P + Pdot * tend) if transient else P

    delta_Q = Q - Q_re
    delta_P = float(Pst - P_re)

    wall = time.perf_counter() - t0
    return (
        task_id,
        delta_Q,
        delta_P,
        float(Pst),
        Q,
        float(Fst),
        tau_res,
        float(p_max),
        float(p_min),
        float(max_h),
        float(min_h),
        float(fmax),
        float(fmin),
        row,
        wall,
    )


# -------------------------------------------------------------------
# Batch runner
# -------------------------------------------------------------------

# ---- multiprocessing-safe worker plumbing (top-level, picklable) ----
# We use a Pool initializer to stash config in module globals so that the
# worker callable itself is picklable under multiprocessing (spawn).
_RUNNER_CFG = None
_DIRECT_CFG = None

def _init_runner_worker(cfg: dict):
    global _RUNNER_CFG
    _RUNNER_CFG = cfg

def _runner_worker(row):
    global _RUNNER_CFG
    cfg = _RUNNER_CFG or {}
    return run_microscale.run_micro_task(task=tuple(row), **cfg)

def _init_direct_worker(cfg: dict):
    global _DIRECT_CFG
    _DIRECT_CFG = cfg

def _direct_worker(row):
    global _DIRECT_CFG
    cfg = _DIRECT_CFG or {}
    return run_micro_task_direct(tuple(row), **cfg)


def run_batch(
    selected_tasks: np.ndarray,
    *,
    transient: bool,
    ah_ratio: Optional[float] = None,
    n_procs: int,
    exec_mode: str,              # "direct" or "runner"
    n_steps: int,
    mesh_n: int,
    export_vtk: bool,
    selected_indices_global: np.ndarray,
    timing_label: str,
) -> Dict[str, np.ndarray]:
    """
    Run a batch and return dict with keys: dP, dQx, dQy, Pst, Fst, time_s

    exec_mode:
      - "direct": uses MicroMixedSolver directly (supports varying mesh_n and n_steps)
      - "runner": uses run_microscale.run_micro_task (fixed mesh + fixed n_steps per runner)
    """
    tasks = _ensure_2d_tasks(selected_tasks)
    nN = tasks.shape[0]

    dP = np.full(nN, np.nan, dtype=float)
    dQx = np.full(nN, np.nan, dtype=float)
    dQy = np.full(nN, np.nan, dtype=float)
    Pst = np.full(nN, np.nan, dtype=float)
    Fst = np.full(nN, np.nan, dtype=float)
    time_s = np.full(nN, np.nan, dtype=float)


    # If using the production runner, pass sweep controls via environment variables.
    # Child processes created via multiprocessing spawn inherit these values at creation time.
    # Build per-sweep config for workers (must be picklable).
    if exec_mode == "runner":
        worker = _runner_worker
        init = _init_runner_worker
        cfg = dict(
            transient=bool(transient),
            macro_dt=None,
            export_vtk=bool(export_vtk),
            ah_ratio=ah_ratio,
        )
    else:
        worker = _direct_worker
        init = _init_direct_worker
        cfg = dict(
            transient=bool(transient),
            ah_ratio=ah_ratio,
            n_steps=int(n_steps),
            mesh_n=int(mesh_n),
            export_vtk=bool(export_vtk),
        )

    # Convert to plain Python lists for pickling speed
    task_rows = [tasks[j, :].tolist() for j in range(nN)]

    t_batch0 = time.perf_counter()
    with ctx.Pool(processes=int(min(n_procs, nN)), initializer=init, initargs=(cfg,)) as pool:
        results = pool.map(worker, task_rows, chunksize=int(max(1, nN // (10 * n_procs))))

    t_batch1 = time.perf_counter()

    for j, out in enumerate(results):
        if exec_mode == "runner":
            delta_Q = np.asarray(out[1], dtype=float).reshape(-1)
            dP[j] = float(out[2])
            dQx[j] = float(delta_Q[0]) if delta_Q.size > 0 else np.nan
            dQy[j] = float(delta_Q[1]) if delta_Q.size > 1 else np.nan
            Pst[j] = float(out[3])
            Fst[j] = float(out[5])
            # no per-task wall time returned in our runner call here
            time_s[j] = np.nan
        else:
            delta_Q = np.asarray(out[1], dtype=float).reshape(-1)
            dP[j] = float(out[2])
            dQx[j] = float(delta_Q[0]) if delta_Q.size > 0 else np.nan
            dQy[j] = float(delta_Q[1]) if delta_Q.size > 1 else np.nan
            Pst[j] = float(out[3])
            Fst[j] = float(out[5])
            time_s[j] = float(out[-1])

    # Put a coarse batch timing into NaNs if we didn't get per-task timing (runner mode)
    if exec_mode == "runner":
        avg = (t_batch1 - t_batch0) / max(1, nN)
        time_s[:] = avg

    return {"dP": dP, "dQx": dQx, "dQy": dQy, "Pst": Pst, "Fst": Fst, "time_s": time_s}


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------------------------
# Main experiment (mirror old script knobs)
# -------------------------------------------------------------------

def main() -> None:
    # ============================================================
    # USER SETTINGS (edit these)
    # ============================================================
    tasks_npy = "PenaltyDebug4/tasks.npy"
    out_dir = "data/analysis/micro_sensitivity_csv_mixed"

    n_select = 50
    export_vtk = False

    # Execution mode:
    #   "direct" (recommended): lets you vary mesh and n_steps
    #   "runner": uses run_microscale.run_micro_task (fixed mesh + fixed n_steps=20 inside runner)
    exec_mode = "direct"

    # Parallelism for tasks inside each batch
    n_procs_batch = 8

    robust_score = True
    weights = {"invH": 2.0, "gmag": 1.0, "Hdot": 1.0, "Pdot": 1.0}

    # 1) "Tsteps" study -> here interpreted as transient n_steps in solver.run(Tf=tend,n_steps=...)
    do_tsteps_study = False
    tsteps_list = [5, 10, 20, 40]  # include reference; we'll use max() as ref

    # 2) Mesh study -> here interpreted as UnitSquareMesh(n,n)
    do_mesh_study = True
    mesh_n_list = [10, 20, 40, 60, 80, 120, 160, 320]
    mesh_ref_mode = "finest"  # "finest" only (richardson disabled here)
    meshstudy_tsteps = 20     # n_steps for mesh study
    meshstudy_Ah = 2.75e-7

    # 3) Ah-ratio sweep (same idea as before)
    do_ah_ratio_sweep = True
    fixed_tsteps_for_ratio = 20
    ah_ratio_values_explicit: List[float] = []
    ah_ratio_min, ah_ratio_max, ah_ratio_n = 0.0, 0.98, 20

    # Default mesh for non-mesh studies:
    default_mesh_n = 60
    # ============================================================

    _safe_mkdir(out_dir)

    idx = TaskIndex()

    tasks_all = _ensure_2d_tasks(np.load(tasks_npy, allow_pickle=True))
    _validate_task_dim(tasks_all, idx)

    sel_tasks, sel_idx_global = select_top_aggressive(
        tasks_all, idx, n_select, weights=weights, robust=robust_score
    )

    meta = {
        "tasks_npy": os.path.abspath(tasks_npy),
        "out_dir": os.path.abspath(out_dir),
        "n_total": int(tasks_all.shape[0]),
        "n_selected": int(sel_tasks.shape[0]),
        "selected_indices_global": sel_idx_global.tolist(),
        "task_index": asdict(idx),
        "micro_dt": float(getattr(micro_solver, "dt", float("nan"))),
        "weights": weights,
        "robust_score": bool(robust_score),
        "export_vtk": bool(export_vtk),
        "exec_mode": exec_mode,
        "timestamp_unix": time.time(),
        "notes": "Mixed-micro sensitivity script for MicroMixedSolver + new run_microscale runner.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_selected_tasks_csv(os.path.join(out_dir, "selected_tasks.csv"), sel_tasks, idx)

    # Restore globals at end
    old_Ah = float(getattr(ideal_film_thickness, "Ah", 0.0))

    try:
        # --------------------------------------------------
        # 1) Tsteps (n_steps) study
        # --------------------------------------------------
        if do_tsteps_study:
            study_dir = os.path.join(out_dir, "tsteps")
            _safe_mkdir(study_dir)

            ideal_film_thickness.Ah = old_Ah

            tsteps_results: Dict[int, Dict[str, np.ndarray]] = {}
            for T in tsteps_list:
                T = int(T)
                print(f"\n[TSTEPS] Running n_steps={T} (Ah={ideal_film_thickness.Ah:.6e}, mesh_n={default_mesh_n}, exec_mode={exec_mode})", flush=True)
                out = run_batch(
                    sel_tasks,
                    transient=False,
                    n_procs=int(n_procs_batch),
                    exec_mode=str(exec_mode),
                    n_steps=T,
                    mesh_n=int(default_mesh_n),
                    export_vtk=bool(export_vtk),
                    selected_indices_global=sel_idx_global,
                    timing_label=f"TSTEPS n_steps={T}",
                )
                tsteps_results[T] = out

                raw_csv = os.path.join(study_dir, f"tsteps_raw__T{T}.csv")
                write_raw_outputs_csv(raw_csv, sel_idx_global, out)
                print(f"[TSTEPS] Wrote {raw_csv}", flush=True)

            refT = int(max(tsteps_list))
            ref = tsteps_results[refT]

            summary_rows: List[Dict[str, float | int | str]] = []
            for T in tsteps_list:
                T = int(T)
                cur = tsteps_results[T]

                ref_dP = ref["dP"]
                ref_qmag = _qmag(ref["dQx"], ref["dQy"])
                cur_dP = cur["dP"]
                cur_qmag = _qmag(cur["dQx"], cur["dQy"])

                abs_dP = np.abs(cur_dP - ref_dP)
                abs_dQ = np.abs(cur_qmag - ref_qmag)

                ref_dP_fin = np.abs(ref_dP[np.isfinite(ref_dP)])
                ref_q_fin = np.abs(ref_qmag[np.isfinite(ref_qmag)])
                eps_dP = max(1e-30, float(np.percentile(ref_dP_fin, 5)) if ref_dP_fin.size else 1e-30)
                eps_dQ = max(1e-30, float(np.percentile(ref_q_fin, 5)) if ref_q_fin.size else 1e-30)

                rel_dP = _rel_err(abs_dP, ref_dP, eps_dP)
                rel_dQ = _rel_err(abs_dQ, ref_qmag, eps_dQ)

                symrel_dP = _sym_rel_err(cur_dP, ref_dP, eps_dP)
                symrel_dQ = _sym_rel_err(cur_qmag, ref_qmag, eps_dQ)

                dP_stats_abs = summarize_errors_vs_ref(cur_dP, ref_dP)
                dQ_stats_abs = summarize_errors_vs_ref(cur_qmag, ref_qmag)

                dP_stats_rel = summarize_errors_vs_ref(rel_dP, np.zeros_like(rel_dP))
                dQ_stats_rel = summarize_errors_vs_ref(rel_dQ, np.zeros_like(rel_dQ))
                dP_stats_symrel = summarize_errors_vs_ref(symrel_dP, np.zeros_like(symrel_dP))
                dQ_stats_symrel = summarize_errors_vs_ref(symrel_dQ, np.zeros_like(symrel_dQ))

                summary_rows.append(
                    {
                        "Tsteps": T,
                        "refT": refT,
                        "Ah_fixed": float(old_Ah),
                        "mesh_n_fixed": int(default_mesh_n),
                        "micro_dt": float(getattr(micro_solver, "dt", float("nan"))),
                        "n_tasks": int(sel_tasks.shape[0]),
                        "dP_err_p50": dP_stats_abs["p50"],
                        "dP_err_p90": dP_stats_abs["p90"],
                        "dP_err_p99": dP_stats_abs["p99"],
                        "dP_err_max": dP_stats_abs["max"],
                        "dQmag_err_p50": dQ_stats_abs["p50"],
                        "dQmag_err_p90": dQ_stats_abs["p90"],
                        "dQmag_err_p99": dQ_stats_abs["p99"],
                        "dQmag_err_max": dQ_stats_abs["max"],
                        "dP_relerr_p50": dP_stats_rel["p50"],
                        "dP_relerr_p90": dP_stats_rel["p90"],
                        "dP_relerr_p99": dP_stats_rel["p99"],
                        "dP_relerr_max": dP_stats_rel["max"],
                        "dQmag_relerr_p50": dQ_stats_rel["p50"],
                        "dQmag_relerr_p90": dQ_stats_rel["p90"],
                        "dQmag_relerr_p99": dQ_stats_rel["p99"],
                        "dQmag_relerr_max": dQ_stats_rel["max"],
                        "dP_symrelerr_p50": dP_stats_symrel["p50"],
                        "dP_symrelerr_p90": dP_stats_symrel["p90"],
                        "dP_symrelerr_p99": dP_stats_symrel["p99"],
                        "dP_symrelerr_max": dP_stats_symrel["max"],
                        "dQmag_symrelerr_p50": dQ_stats_symrel["p50"],
                        "dQmag_symrelerr_p90": dQ_stats_symrel["p90"],
                        "dQmag_symrelerr_p99": dQ_stats_symrel["p99"],
                        "dQmag_symrelerr_max": dQ_stats_symrel["max"],
                    }
                )

                err_csv = os.path.join(study_dir, f"tsteps_errors__T{T}_vs_refT{refT}.csv")
                write_errors_vs_ref_csv(err_csv, sel_idx_global, cur, ref)
                print(f"[TSTEPS] Wrote {err_csv}", flush=True)

            summary_csv = os.path.join(study_dir, "tsteps_summary_vs_ref.csv")
            write_summary_csv(summary_csv, summary_rows)
            print(f"[TSTEPS] Wrote {summary_csv}", flush=True)

        # --------------------------------------------------
        # 2) Mesh study
        # --------------------------------------------------
        if do_mesh_study:
            study_dir = os.path.join(out_dir, "mesh")
            _safe_mkdir(study_dir)

            ideal_film_thickness.Ah = float(meshstudy_Ah)

            mesh_results: Dict[int, Dict[str, np.ndarray]] = {}
            for nmesh in mesh_n_list:
                nmesh = int(nmesh)
                print(f"\n[MESH] Running mesh_n={nmesh} (n_steps={meshstudy_tsteps}, Ah={ideal_film_thickness.Ah:.6e}, exec_mode={exec_mode})", flush=True)
                out = run_batch(
                    sel_tasks,
                    transient=False,
                    n_procs=int(n_procs_batch),
                    exec_mode=str(exec_mode),
                    n_steps=int(meshstudy_tsteps),
                    mesh_n=nmesh,
                    export_vtk=bool(export_vtk),
                    selected_indices_global=sel_idx_global,
                    timing_label=f"MESH mesh_n={nmesh}",
                )
                mesh_results[nmesh] = out

                raw_csv = os.path.join(study_dir, f"mesh_raw__ncells{nmesh}.csv")
                write_raw_outputs_csv(raw_csv, sel_idx_global, out)
                print(f"[MESH] Wrote {raw_csv}", flush=True)

            refN = int(max(mesh_n_list))
            ref = mesh_results[refN]
            ref_label = f"refmesh{refN}"

            summary_rows: List[Dict[str, float | int | str]] = []
            for nmesh in mesh_n_list:
                nmesh = int(nmesh)
                cur = mesh_results[nmesh]

                dP_stats_abs = summarize_errors_vs_ref(cur["dP"], ref["dP"])
                dQ_stats_abs = summarize_errors_vs_ref(
                    _qmag(cur["dQx"], cur["dQy"]),
                    _qmag(ref["dQx"], ref["dQy"]),
                )

                summary_rows.append(
                    {
                        "ncells": nmesh,
                        "ref_label": ref_label,
                        "Tsteps_fixed": int(meshstudy_tsteps),
                        "Ah_fixed": float(ideal_film_thickness.Ah),
                        "micro_dt": float(getattr(micro_solver, "dt", float("nan"))),
                        "n_tasks": int(sel_tasks.shape[0]),
                        "dP_err_p50": dP_stats_abs["p50"],
                        "dP_err_p90": dP_stats_abs["p90"],
                        "dP_err_p99": dP_stats_abs["p99"],
                        "dP_err_max": dP_stats_abs["max"],
                        "dQmag_err_p50": dQ_stats_abs["p50"],
                        "dQmag_err_p90": dQ_stats_abs["p90"],
                        "dQmag_err_p99": dQ_stats_abs["p99"],
                        "dQmag_err_max": dQ_stats_abs["max"],
                    }
                )

                err_csv = os.path.join(study_dir, f"mesh_errors__ncells{nmesh}_vs_{ref_label}.csv")
                write_errors_vs_ref_csv(err_csv, sel_idx_global, cur, ref)
                print(f"[MESH] Wrote {err_csv}", flush=True)

            summary_csv = os.path.join(study_dir, "mesh_summary_vs_ref.csv")
            write_summary_csv(summary_csv, summary_rows)
            print(f"[MESH] Wrote {summary_csv}", flush=True)

        # --------------------------------------------------
        # 3) Ah-ratio sweep
        # --------------------------------------------------
        if do_ah_ratio_sweep:
            study_dir = os.path.join(out_dir, "ahratio")
            _safe_mkdir(study_dir)

            if ah_ratio_values_explicit:
                ratio_list = [float(r) for r in ah_ratio_values_explicit]
            else:
                ratio_list = np.linspace(float(ah_ratio_min), float(ah_ratio_max), int(ah_ratio_n)).tolist()
            ratio_list = [max(0.0, min(0.99, float(r))) for r in ratio_list]

            ratio_results: Dict[float, Dict[str, np.ndarray]] = {}
            for r in ratio_list:
                r = float(r)
                print(f"\n[AHRATIO] Running ratio={r:.4f} (Ah_task=r*H_task) (n_steps={fixed_tsteps_for_ratio}, mesh_n={default_mesh_n})", flush=True)

                # Per-task Ah override is applied inside the worker (Ah_task = ah_ratio * H_task).
                out = run_batch(
                    sel_tasks,
                    transient=False,
                    ah_ratio=r,
                    n_procs=int(n_procs_batch),
                    exec_mode=str(exec_mode),
                    n_steps=int(fixed_tsteps_for_ratio),
                    mesh_n=int(default_mesh_n),
                    export_vtk=bool(export_vtk),
                    selected_indices_global=sel_idx_global,
                    timing_label=f"AHRATIO r={r:.4f}",
                )
                # Note: Above doesn't actually apply per-task Ah without editing run_batch.
                # If you need per-task Ah_ratio, see comment near the end of this script.
                ratio_results[r] = out

                r_tag = str(r).replace(".", "p")
                raw_csv = os.path.join(study_dir, f"ahratio_raw__r{r_tag}.csv")
                write_raw_outputs_csv(raw_csv, sel_idx_global, out)
                print(f"[AHRATIO] Wrote {raw_csv}", flush=True)


            summary_rows = []
            for r in ratio_list:
                out = ratio_results[float(r)]
                dP = out["dP"]
                qmag = _qmag(out["dQx"], out["dQy"])

                def pct(x: np.ndarray, p: float) -> float:
                    x1 = x[np.isfinite(x)]
                    return float(np.percentile(x1, p)) if x1.size else float("nan")

                summary_rows.append(
                    {
                        "ah_ratio": float(r),
                        "Tsteps_fixed": int(fixed_tsteps_for_ratio),
                        "mesh_n_fixed": int(default_mesh_n),
                        "micro_dt": float(getattr(micro_solver, "dt", float("nan"))),
                        "n_tasks": int(sel_tasks.shape[0]),
                        "dP_p10": pct(dP, 10),
                        "dP_p50": pct(dP, 50),
                        "dP_p90": pct(dP, 90),
                        "dP_p99": pct(dP, 99),
                        "dQmag_p10": pct(qmag, 10),
                        "dQmag_p50": pct(qmag, 50),
                        "dQmag_p90": pct(qmag, 90),
                        "dQmag_p99": pct(qmag, 99),
                    }
                )

            summary_csv = os.path.join(study_dir, "ahratio_summary_percentiles.csv")
            write_summary_csv(summary_csv, summary_rows)
            print(f"[AHRATIO] Wrote {summary_csv}", flush=True)

    finally:
        ideal_film_thickness.Ah = old_Ah
        print("\n[RESTORE] Restored ideal_film_thickness.Ah", flush=True)


if __name__ == "__main__":
    main()
