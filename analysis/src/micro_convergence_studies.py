#!/usr/bin/env python3
"""
Run microscale convergence studies in parallel on a random subset of tasks.

python3 micro_convergence_studies.py \
  --tasks-file PenaltyDebug5/tasks.npy \
  --output-dir PenaltyDebug5/convergence \
  --no-tasks-selected 25 \
  --seed 1 \
  --transient \
  --Tf 0.05 \
  --nprocs 24


Studies (toggleable):
1) Mesh convergence: n_h = n_m in [5,10,20,40,80,160,320,640]
2) Time-step discretisation: n_steps in [2,4,8,16,32,64,128]
3) Ah-ratio sweep: Ah := H0*ratio, ratio in linspace(0.0, 0.95, 10)

Outputs (CSV, long-form, one row per (task,param)):
- selected_tasks.csv
- mesh_raw.csv
- tsteps_raw.csv
- ahratio_raw.csv
- meta.json

What we store per run:
- Qx_re, Qy_re, P_re  (per-task, per-run reference values)
- dP = Pst - P_re
- dQx, dQy = (Qx,Qy) - (Qx_re,Qy_re)
- dP/P_re, dQx/Qx_re, dQy/Qy_re  (per-run normalised increments)

Notes:
- Uses multiprocessing spawn (safe with dolfin in child processes)
- Each worker keeps a mesh cache keyed by mesh_n to avoid rebuilding meshes repeatedly
- Output is intentionally "raw"; plotting/summary (mean/std bands) comes later.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import csv
import argparse
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

import numpy as np

# Threading guards (important when using many processes)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---- Your imports (match run_microscale.py style) ----
from CONFIGPenalty import micro_physical, micro_solver, ideal_film_thickness

from microscale.src.functions.micro_mixed_transient_class import (
    MicroMixedSolver,
    MicroPhysicalParameters,
    MicroSolverSettings,
)

from dolfin import *
set_log_active(False)
set_log_level(LogLevel.ERROR)
# ---------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------
def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _print_job_breakdown(jobs: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for j in jobs:
        counts[j["study"]] = counts.get(j["study"], 0) + 1
    total = len(jobs)
    print("[INFO] Job breakdown:")
    for k in sorted(counts):
        print(f"  - {k:8s}: {counts[k]:6d}")
    print(f"  - {'TOTAL':8s}: {total:6d}")
    sys.stdout.flush()


# ---------------------------------------------------------------------
# Reference model (matches your micro_mixed_transient_class.py main() logic)
# ---------------------------------------------------------------------
def _roelands_eta(p: float) -> float:
    """
    Numpy Roelands viscosity model (dimensional).
    Returns eta [Pa·s].
    """
    eta_p = 6.31e-5
    eta00 = 1.0e-2
    p_r = 0.5e8
    z = 0.548
    expo_cap = 10.0

    p_pos = max(float(p), 0.0)
    expo = (1.0 + p_pos / p_r) ** z
    expo_c = min(expo, expo_cap)

    base_log = np.log(eta00 / eta_p)
    ln_eta = np.log(eta_p) + expo_c * base_log
    return float(np.exp(ln_eta))


def _q_re(
    transient: bool,
    H: float,
    P: float,
    Ux: float,
    Uy: float,
    gradpx: float,
    gradpy: float,
    *,
    Hdot: float = 0.0,
    Pdot: float = 0.0,
    dt: float = 0.0,
) -> Tuple[float, float]:
    """
    Analytical/reference flux (dimensional) used for delta_Q.
    Returns (Qx_re, Qy_re).
    """
    if transient:
        Hdt = float(H + Hdot * dt)
        Pdt = float(P + Pdot * dt)
    else:
        Hdt = float(H)
        Pdt = float(P)

    eta = _roelands_eta(Pdt)

    qx = float(Ux * Hdt - (Hdt**3) * gradpx / (12.0 * eta))
    qy = float(Uy * Hdt - (Hdt**3) * gradpy / (12.0 * eta))
    return qx, qy


def _safe_div(num: float, den: float, eps: float) -> float:
    """Return num/den, but NaN if |den| <= eps or nonfinite."""
    if not (np.isfinite(num) and np.isfinite(den)):
        return float("nan")
    if abs(den) <= eps:
        return float("nan")
    return float(num / den)


# ---------------------------------------------------------------------
# Per-worker mesh cache
# ---------------------------------------------------------------------
_MESH_CACHE: Dict[int, Tuple[Any, Any]] | None = None


def _get_worker_meshes(mesh_n: int):
    """Return (mesh_m, mesh_h) for this worker, cached per mesh_n."""
    global _MESH_CACHE
    if _MESH_CACHE is None:
        _MESH_CACHE = {}

    if mesh_n not in _MESH_CACHE:
        print(f"[PID {os.getpid()}] Building meshes for mesh_n={mesh_n}")
        sys.stdout.flush()
        from dolfin import UnitSquareMesh

        m = UnitSquareMesh(mesh_n, mesh_n)
        h = UnitSquareMesh(mesh_n, mesh_n)
        _MESH_CACHE[mesh_n] = (m, h)

    return _MESH_CACHE[mesh_n]


# ---------------------------------------------------------------------
# Task parsing (same structure as run_microscale.py)
# ---------------------------------------------------------------------
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
        (task_id, row_idx, H, P, Ux, Uy, _, gradp1, gradp2, _) = task
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


# ---------------------------------------------------------------------
# ht() helper (ratio override)
# ---------------------------------------------------------------------
def _make_ht_with_ratio(ratio: float | None):
    """
    Returns ht(x,y,xmax,ymax,H0,HT,T,Ux,Uy) using kx/ky from ideal_film_thickness.
    If ratio is not None: Ah := ratio * H0 (dimensional), overriding config Ah.
    """
    from dolfin import cos, pi, Constant

    kx = ideal_film_thickness.kx
    ky = ideal_film_thickness.ky

    def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        x_d, y_d = x * xmax, y * ymax

        if ratio is None:
            Ah = ideal_film_thickness.Ah
        else:
            Ah = Constant(float(ratio)) * H0  # UFL-safe

        # optional python-side safety check (best-effort)
        try:
            H0_val = float(H0)
            HT_val = float(HT)
            T_val = float(T)
            Ah_val = float(ideal_film_thickness.Ah) if ratio is None else float(ratio) * H0_val
            hmin0 = H0_val - Ah_val
            hmin1 = H0_val - Ah_val + HT_val * T_val
            if (hmin0 < 0.0) or (hmin1 < 0.0):
                print(
                    f"Warning: Negative film thickness at T={T_val:.6e}s, "
                    f"hmin0={hmin0:.6e}, hmin1={hmin1:.6e}."
                )
                sys.stdout.flush()
        except Exception:
            pass

        return H0 + HT * T + 0.5 * Ah * (
            cos(kx * 2 * pi * x_d / xmax) + cos(ky * 2 * pi * y_d / ymax)
        )

    return ht


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------
def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job keys:
      study: 'mesh' | 'tsteps' | 'ahratio'
      transient: bool
      mesh_n: int
      n_steps: int
      ratio: float|None
      macro_dt: float|None
      tend: float
      task: np.ndarray row
      task_index: int
    """
    t0 = time.time()
    pid = os.getpid()

    study = job["study"]
    transient = bool(job["transient"])
    mesh_n = int(job["mesh_n"])
    n_steps = int(job["n_steps"])
    ratio = job.get("ratio", None)
    macro_dt = job.get("macro_dt", None)
    tend = float(job["tend"])

    # division safety for normalised increments
    DIV_EPS = 1e-30

    task = job["task"]
    task_id, row_idx, H, P, Ux, Uy, gradp1, gradp2, Hdot, Pdot = _parse_task(task, transient=transient)

    mesh_m, mesh_h = _get_worker_meshes(mesh_n)
    ht = _make_ht_with_ratio(ratio)

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
        Tend=(macro_dt if (transient and macro_dt is not None) else (tend if transient else 0.0)),
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
        mesh_m=mesh_m,
        mesh_h=mesh_h,
        physical_params=phys,
        solver_settings=settings,
        k=1,
        ht=ht,
        export_vtk=False,
        output_dir=os.path.join(os.getcwd(), "output"),
        auto_solve=False,
    )

    # Run
    if transient:
        Tf = float(phys.Tend)
        if Tf <= 0.0:
            solver.run_steady()
        else:
            solver.run(Tf=Tf, n_steps=n_steps)
    else:
        solver.run_steady()

    # Only need Qx,Qy,Pst from homogenisation
    Qx, Qy, Pst, *_ = solver.spatial_homogenisation()

    # Per-run reference values (task-specific, uses this run's dt)
    dt = float(phys.Tend) if transient else 0.0
    Qx_re, Qy_re = _q_re(
        transient=transient,
        H=H,
        P=P,
        Ux=Ux,
        Uy=Uy,
        gradpx=gradp1,
        gradpy=gradp2,
        Hdot=Hdot,
        Pdot=Pdot,
        dt=dt,
    )
    P_re = float(P + Pdot * dt) if transient else float(P)

    # Increments
    dQx = float(Qx - Qx_re)
    dQy = float(Qy - Qy_re)
    dP = float(Pst - P_re)

    # Normalised increments (task-specific denominators)
    dQx_over_Qxre = _safe_div(dQx, Qx_re, DIV_EPS)
    dQy_over_Qyre = _safe_div(dQy, Qy_re, DIV_EPS)
    dP_over_Pre = _safe_div(dP, P_re, DIV_EPS)

    # cleanup
    del solver, phys, settings
    # gc.collect()

    wall = time.time() - t0

    return {
        # identifiers
        "study": study,
        "task_index": int(job["task_index"]),
        "task_id": task_id,
        "row_idx": row_idx,
        "pid": pid,

        # sweep parameters
        "mesh_n": mesh_n,
        "n_steps": n_steps,
        "ratio": ("" if ratio is None else float(ratio)),
        "transient": int(transient),
        "Tf": (tend if transient else 0.0),

        # task inputs
        "H0": H,
        "P0": P,
        "Ux": Ux,
        "Uy": Uy,
        "dpdx": gradp1,
        "dpdy": gradp2,
        "Hdot": Hdot,
        "Pdot": Pdot,

        # reference values (per task, per run)
        "Qx_re": float(Qx_re),
        "Qy_re": float(Qy_re),
        "P_re": float(P_re),

        # increments
        "dP": float(dP),
        "dQx": float(dQx),
        "dQy": float(dQy),

        # normalised increments (what you want to plot)
        "dP_over_Pre": float(dP_over_Pre) if np.isfinite(dP_over_Pre) else float("nan"),
        "dQx_over_Qxre": float(dQx_over_Qxre) if np.isfinite(dQx_over_Qxre) else float("nan"),
        "dQy_over_Qyre": float(dQy_over_Qyre) if np.isfinite(dQy_over_Qyre) else float("nan"),

        # timing
        "wall_time_s": float(wall),
    }


# ---------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------
def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_selected_tasks_csv(path: str, selected_tasks: np.ndarray, transient: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["sel_index", "task_id", "row_idx", "H0", "P0", "Ux", "Uy", "dpdx", "dpdy", "Hdot", "Pdot"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, task in enumerate(selected_tasks):
            task_id, row_idx, H, P, Ux, Uy, dpdx, dpdy, Hdot, Pdot = _parse_task(task, transient=transient)
            w.writerow([i, task_id, row_idx, H, P, Ux, Uy, dpdx, dpdy, Hdot, Pdot])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True, help="Path to tasks.npy")
    ap.add_argument("--output-dir", required=True, help="Output directory for CSV/meta")
    ap.add_argument("--no-tasks-selected", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--transient", action="store_true", help="Use transient solve (default: steady)")
    ap.add_argument("--Tf", type=float, default=0.05, help="Transient final time (used if transient and macro_dt not provided)")
    ap.add_argument("--macro-dt", type=float, default=None, help="If provided, overrides Tf with macro dt (like run_microscale macro_dt)")

    ap.add_argument("--nprocs", type=int, default=int(os.environ.get("MICRO_NPROCS", "24")))

    # toggles
    ap.add_argument("--mesh-study", action="store_true")
    ap.add_argument("--tsteps-study", action="store_true")
    ap.add_argument("--ahratio-study", action="store_true")

    args = ap.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    tasks = np.load(args.tasks_file, allow_pickle=True)
    N = len(tasks)

    run_mesh = args.mesh_study
    run_tsteps = args.tsteps_study
    run_ahratio = args.ahratio_study
    if not (run_mesh or run_tsteps or run_ahratio):
        run_mesh = run_tsteps = run_ahratio = True

    transient = bool(args.transient)

    rng = np.random.default_rng(args.seed)
    k = int(min(max(args.no_tasks_selected, 1), N))
    sel_idx = rng.choice(N, size=k, replace=False)
    selected = tasks[sel_idx]

    _write_selected_tasks_csv(os.path.join(out_dir, "selected_tasks.csv"), selected, transient=transient)

    # Sweep grids
    mesh_list = [5, 10, 20, 40, 80, 160]
    nsteps_list = [2, 4, 8, 16, 32, 64]
    ratios = np.linspace(0.0, 0.95, 10).tolist()

    jobs: List[Dict[str, Any]] = []
    tend = float(args.Tf)

    nsteps_fixed_for_non_tsteps = 10 if transient else 1

    if run_mesh:
        for ti, task in enumerate(selected):
            for mesh_n in mesh_list:
                jobs.append({
                    "study": "mesh",
                    "transient": transient,
                    "mesh_n": mesh_n,
                    "n_steps": nsteps_fixed_for_non_tsteps,
                    "ratio": None,
                    "macro_dt": args.macro_dt,
                    "tend": tend,
                    "task": task,
                    "task_index": ti,
                })

    if run_tsteps:
        for ti, task in enumerate(selected):
            for ns in nsteps_list:
                jobs.append({
                    "study": "tsteps",
                    "transient": True,
                    "mesh_n": 40,
                    "n_steps": ns,
                    "ratio": None,
                    "macro_dt": args.macro_dt,
                    "tend": tend,
                    "task": task,
                    "task_index": ti,
                })

    if run_ahratio:
        for ti, task in enumerate(selected):
            for r in ratios:
                jobs.append({
                    "study": "ahratio",
                    "transient": transient,
                    "mesh_n": 40,
                    "n_steps": nsteps_fixed_for_non_tsteps,
                    "ratio": float(r),
                    "macro_dt": args.macro_dt,
                    "tend": tend,
                    "task": task,
                    "task_index": ti,
                })

    meta = {
        "tasks_file": os.path.abspath(args.tasks_file),
        "output_dir": os.path.abspath(out_dir),
        "seed": args.seed,
        "no_tasks_selected": k,
        "selected_task_indices_in_tasks_npy": sel_idx.tolist(),
        "transient_default": int(transient),
        "Tf": tend,
        "macro_dt": args.macro_dt,
        "nprocs": args.nprocs,
        "run_mesh": bool(run_mesh),
        "run_tsteps": bool(run_tsteps),
        "run_ahratio": bool(run_ahratio),
        "mesh_list": mesh_list,
        "nsteps_list": nsteps_list,
        "ratios": ratios,
        "nsteps_fixed_for_non_tsteps": nsteps_fixed_for_non_tsteps,
        "outputs_saved": [
            "Qx_re", "Qy_re", "P_re",
            "dP", "dQx", "dQy",
            "dP_over_Pre", "dQx_over_Qxre", "dQy_over_Qyre",
        ],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Loaded {N} tasks from {args.tasks_file}")
    print(f"[INFO] Selected {k} tasks (seed={args.seed})")
    print(f"[INFO] Jobs to run: {len(jobs)} (mesh={run_mesh}, tsteps={run_tsteps}, ahratio={run_ahratio})")
    _print_job_breakdown(jobs)
    sys.stdout.flush()

    # Parallel run
    ctx = mp.get_context("spawn")
    start = time.time()

    rows_mesh: List[Dict[str, Any]] = []
    rows_tsteps: List[Dict[str, Any]] = []
    rows_ahr: List[Dict[str, Any]] = []

    progress_every = max(1, len(jobs) // 50)   # ~2% updates
    flush_every = max(1, len(jobs) // 10)      # write partial CSVs ~10 times
    track_slowest = 5

    done = 0
    done_by_study = {"mesh": 0, "tsteps": 0, "ahratio": 0}
    slowest: List[tuple] = []

    def _maybe_track_slowest(row: Dict[str, Any]) -> None:
        nonlocal slowest
        if track_slowest <= 0:
            return
        tup = (
            float(row.get("wall_time_s", 0.0)),
            row.get("study", ""),
            int(row.get("task_index", -1)),
            int(row.get("mesh_n", -1)),
            int(row.get("n_steps", -1)),
            row.get("ratio", ""),
        )
        slowest.append(tup)
        slowest.sort(key=lambda x: x[0], reverse=True)
        del slowest[track_slowest:]

    def _flush_partials() -> None:
        if rows_mesh:
            _write_csv(os.path.join(out_dir, "mesh_raw__partial.csv"), rows_mesh)
        if rows_tsteps:
            _write_csv(os.path.join(out_dir, "tsteps_raw__partial.csv"), rows_tsteps)
        if rows_ahr:
            _write_csv(os.path.join(out_dir, "ahratio_raw__partial.csv"), rows_ahr)

    with ctx.Pool(processes=int(max(1, args.nprocs))) as pool:
        for row in pool.imap_unordered(_run_job, jobs, chunksize=5):
            done += 1
            st = row["study"]
            done_by_study[st] = done_by_study.get(st, 0) + 1

            if st == "mesh":
                rows_mesh.append(row)
            elif st == "tsteps":
                rows_tsteps.append(row)
            elif st == "ahratio":
                rows_ahr.append(row)

            _maybe_track_slowest(row)

            if (done % progress_every == 0) or (done == len(jobs)):
                now = time.time()
                elapsed = now - start
                rate = done / elapsed if elapsed > 1e-12 else 0.0
                remaining = len(jobs) - done
                eta = remaining / rate if rate > 1e-12 else float("inf")

                print(
                    "[INFO] "
                    f"{done:6d}/{len(jobs):6d} "
                    f"({100.0*done/len(jobs):5.1f}%) | "
                    f"mesh {done_by_study.get('mesh',0):5d} "
                    f"tsteps {done_by_study.get('tsteps',0):5d} "
                    f"ahr {done_by_study.get('ahratio',0):5d} | "
                    f"elapsed {_fmt_hms(elapsed)} | "
                    f"{rate:6.3f} jobs/s | "
                    f"ETA {_fmt_hms(eta)}"
                )
                sys.stdout.flush()

                if track_slowest > 0 and slowest:
                    print("[INFO] Slowest jobs so far:")
                    for w, s, ti, mn, ns, rr in slowest:
                        rr_s = rr if rr != "" else "None"
                        print(f"   - {w:8.2f}s | {s:7s} | task={ti:3d} | mesh_n={mn:4d} | n_steps={ns:4d} | ratio={rr_s}")
                    sys.stdout.flush()

            if (done % flush_every == 0) and (done != len(jobs)):
                print(f"[INFO] Writing partial CSVs at {done}/{len(jobs)} ...")
                sys.stdout.flush()
                _flush_partials()

    # Sort rows deterministically
    def _sort_key(r):
        return (
            r["task_index"],
            r["task_id"],
            r["mesh_n"],
            r["n_steps"],
            (999 if r["ratio"] == "" else float(r["ratio"])),
        )

    rows_mesh.sort(key=_sort_key)
    rows_tsteps.sort(key=_sort_key)
    rows_ahr.sort(key=_sort_key)

    if rows_mesh:
        _write_csv(os.path.join(out_dir, "mesh_raw.csv"), rows_mesh)
    if rows_tsteps:
        _write_csv(os.path.join(out_dir, "tsteps_raw.csv"), rows_tsteps)
    if rows_ahr:
        _write_csv(os.path.join(out_dir, "ahratio_raw.csv"), rows_ahr)

    total = time.time() - start
    print(f"[DONE] All studies completed in {_fmt_hms(total)}")
    print(f"[DONE] Wrote outputs to: {out_dir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
