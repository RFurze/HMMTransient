#!/usr/bin/env python3
"""
Plot microscale sensitivity results from exported CSVs (Plotly + house style).

ASSUMPTION (as requested):
- ALL CSVs live in a SINGLE directory ROOT (no subfolders).

Expected files in ROOT (depending on which studies you ran):
  meta.json
  selected_tasks.csv

  # Tsteps study (optional)
  tsteps_summary_vs_ref.csv
  tsteps_errors__T*_vs_refT*.csv

  # Mesh study (optional)
  mesh_summary_vs_ref.csv
  mesh_errors__n*_vs_refn*.csv

  # Ah-ratio sweep (optional; replaces Ah sweep)
  ahratio_summary_percentiles.csv
  ahratio_raw__r*.csv

Outputs:
  ROOT/plots_plotly/
    relerr_dP_vs_dt.png
    relerr_dQ_vs_dt.png
    relerr_dP_vs_meshN.png
    relerr_dQ_vs_meshN.png
    dP_vs_ah_ratio_mean_std.png
    dQmag_vs_ah_ratio_mean_std.png

What we plot (simplified, per your request):
  1) Relative error in dP and dQ vs dt (effective dt = micro_dt / Tsteps)
     - Uses per-task relative errors from tsteps_errors__... CSVs
     - Shows mean ± std across tasks
  2) Relative error in dP and dQ vs N (mesh nodes)
     - Uses per-task relative errors from mesh_errors__... CSVs
     - Shows mean ± std across tasks
  3) dP and dQ vs Ah_ratio
     - Uses ahratio_raw__r*.csv to compute mean ± std across tasks

Notes on data availability:
- For relative error plots we need per-task rel errors, which are in:
    rel_err_dP, rel_err_dQmag columns
  produced by your runner in write_errors_vs_ref_csv().
- mesh_summary_vs_ref.csv currently does NOT include relative errors (only abs).
  That's fine: we plot rel errors from mesh_errors__... CSVs.
- For dt on the timestep study, we use:
    dt_eff = micro_dt / Tsteps
  where micro_dt is read from meta.json. If meta.json is missing, we fall back to dt_eff = 1/Tsteps.

Requires:
- plotly
- kaleido (for saving png via fig.write_image)
- your plotly_style module (the code you shared) available as `plotly_style.py` on PYTHONPATH
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from plotly_formats import (
    PALETTE,
    AxisSpec,
    FigureSpec,
    apply_house_style,
    add_line,
    add_envelope_fill,
    save_png,
)

# ----------------------------
# Utilities
# ----------------------------
def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def col_as_float(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    out: List[float] = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def parse_first_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_first_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def load_meta_micro_dt(root: str) -> Optional[float]:
    meta_path = os.path.join(root, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dt = meta.get("micro_dt", None)
        return float(dt) if dt is not None else None
    except Exception:
        return None


# ----------------------------
# Core aggregators
# ----------------------------
@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float
    n: int


def mean_std(x: np.ndarray) -> MeanStd:
    v = finite(x)
    if v.size == 0:
        return MeanStd(mean=float("nan"), std=float("nan"), n=0)
    return MeanStd(mean=float(np.mean(v)), std=float(np.std(v)), n=int(v.size))


def load_per_task_rel_errors(error_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (rel_err_dP, rel_err_dQmag) arrays.
    """
    rows = read_csv_rows(error_csv_path)
    rel_dP = col_as_float(rows, "rel_err_dP")
    rel_dQ = col_as_float(rows, "rel_err_dQmag")
    return rel_dP, rel_dQ


def aggregate_rel_error_series(
    csv_paths: List[str],
    *,
    x_from_path: str,   # "tsteps" or "mesh"
    micro_dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build series:
      x, dP_mean, dP_std, dQ_mean, dQ_std

    For x_from_path == "tsteps":
      Extract T from filename and compute x = dt_eff = micro_dt / T if micro_dt else 1/T.
    For x_from_path == "mesh":
      Extract n from filename and use x = n.
    """
    xs: List[float] = []
    dP_means: List[float] = []
    dP_stds: List[float] = []
    dQ_means: List[float] = []
    dQ_stds: List[float] = []

    for p in sorted(csv_paths):
        base = os.path.basename(p)

        if x_from_path == "tsteps":
            T = parse_first_int(r"__T(\d+)_vs_refT\d+", base)
            if T is None:
                # try a bit more permissive
                T = parse_first_int(r"T(\d+)", base)
            if T is None or T <= 0:
                continue
            if micro_dt is not None and np.isfinite(micro_dt):
                x = float(micro_dt) / float(T)
            else:
                x = 1.0 / float(T)

        elif x_from_path == "mesh":
            n = parse_first_int(r"__n(\d+)_vs_", base)
            if n is None:
                n = parse_first_int(r"n(\d+)", base)
            if n is None or n <= 0:
                continue
            x = float(n)**2

        else:
            raise ValueError(f"Unknown x_from_path={x_from_path}")

        rel_dP, rel_dQ = load_per_task_rel_errors(p)
        msP = mean_std(rel_dP)
        msQ = mean_std(rel_dQ)

        xs.append(x)
        dP_means.append(msP.mean)
        dP_stds.append(msP.std)
        dQ_means.append(msQ.mean)
        dQ_stds.append(msQ.std)

    if not xs:
        return (
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
        )

    xarr = np.asarray(xs, float)
    # sort by x
    order = np.argsort(xarr)
    xarr = xarr[order]
    dP_mean = np.asarray(dP_means, float)[order]
    dP_std = np.asarray(dP_stds, float)[order]
    dQ_mean = np.asarray(dQ_means, float)[order]
    dQ_std = np.asarray(dQ_stds, float)[order]
    return xarr, dP_mean, dP_std, dQ_mean, dQ_std


def aggregate_ahratio_raw_series(root: str) -> Tuple[np.ndarray, MeanStd, MeanStd, MeanStd, MeanStd, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads ahratio_raw__r*.csv files and aggregates:
      x = ratio
      dP_mean ± std
      dQmag_mean ± std
    Also returns the raw per-ratio mean/std arrays for plotting.
    """
    paths = sorted(glob.glob(os.path.join(root, "ahratio_raw__r*.csv")))
    ratios: List[float] = []
    dP_mean: List[float] = []
    dP_std: List[float] = []
    dQ_mean: List[float] = []
    dQ_std: List[float] = []

    for p in paths:
        base = os.path.basename(p)
        # tag: ahratio_raw__r0p100000.csv etc.
        # We'll parse from the CSV by reconstituting from filename:
        # Extract after "__r" up to ".csv" and reverse safe formatting:
        m = re.search(r"__r(.+)\.csv$", base)
        if not m:
            continue
        tag = m.group(1)
        # tag was created by f"{r:.6f}".replace(".", "p")
        # Convert back:
        tag2 = tag.replace("p", ".")
        try:
            r = float(tag2)
        except Exception:
            # fallback: try to parse a float in the base name
            r2 = parse_first_float(r"r([0-9]+\.[0-9]+)", base)
            if r2 is None:
                continue
            r = r2

        rows = read_csv_rows(p)
        dP = col_as_float(rows, "dP")
        dQmag = col_as_float(rows, "dQmag")

        msP = mean_std(dP)
        msQ = mean_std(dQmag)

        ratios.append(float(r))
        dP_mean.append(msP.mean)
        dP_std.append(msP.std)
        dQ_mean.append(msQ.mean)
        dQ_std.append(msQ.std)

    if not ratios:
        return (
            np.asarray([], float),
            MeanStd(np.nan, np.nan, 0),
            MeanStd(np.nan, np.nan, 0),
            MeanStd(np.nan, np.nan, 0),
            MeanStd(np.nan, np.nan, 0),
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
        )

    rarr = np.asarray(ratios, float)
    order = np.argsort(rarr)
    rarr = rarr[order]
    dP_mean_arr = np.asarray(dP_mean, float)[order]
    dP_std_arr = np.asarray(dP_std, float)[order]
    dQ_mean_arr = np.asarray(dQ_mean, float)[order]
    dQ_std_arr = np.asarray(dQ_std, float)[order]

    return (
        rarr,
        MeanStd(float(np.mean(dP_mean_arr)), float(np.std(dP_mean_arr)), int(dP_mean_arr.size)),
        MeanStd(float(np.mean(dP_std_arr)), float(np.std(dP_std_arr)), int(dP_std_arr.size)),
        MeanStd(float(np.mean(dQ_mean_arr)), float(np.std(dQ_mean_arr)), int(dQ_mean_arr.size)),
        MeanStd(float(np.mean(dQ_std_arr)), float(np.std(dQ_std_arr)), int(dQ_std_arr.size)),
        dP_mean_arr,
        dP_std_arr,
        dQ_mean_arr,
        dQ_std_arr,
    )


# ----------------------------
# Plot helpers (mean ± std band)
# ----------------------------
def plot_mean_std_band(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    out_png: str,
    x_log: bool = False,
    y_log: bool = False,
) -> None:

    if x.size == 0:
        print(f"[SKIP] No data for {out_png}")
        return

    fig = go.Figure()

    # envelope mean-std .. mean+std
    y_lo = y_mean - y_std
    y_hi = y_mean + y_std
    x_env = np.concatenate([x, x[::-1]])
    y_env = np.concatenate([y_lo, y_hi[::-1]])
    add_envelope_fill(fig, x_env, y_env, name="mean ± std", fillcolor=PALETTE["blue_fill"])

    # mean line
    add_line(fig, x, y_mean, name="mean", color=PALETTE["blue_dark"], width=3.0)

    fig.update_layout(title=title)
    apply_house_style(
        fig,
        xaxis=AxisSpec(title=xaxis_title, tickformat=".2e" if x_log else ".2g"),
        yaxis=AxisSpec(title=yaxis_title, tickformat=".2e"),
        legend=True,
        hover=False,
    )

    if x_log:
        # Force decade ticks only
        xmin = float(np.nanmin(x[np.isfinite(x)]))
        xmax = float(np.nanmax(x[np.isfinite(x)]))
        if xmin > 0 and xmax > 0:
            pmin = int(np.floor(np.log10(xmin)))
            pmax = int(np.ceil(np.log10(xmax)))
            tickvals = [10.0**p for p in range(pmin, pmax + 1)]

            fig.update_xaxes(
                type="log",
                tickmode="array",
                tickvals=tickvals,
                tickformat=".0e",
                exponentformat="e",
                showexponent="all",
            )
        else:
            fig.update_xaxes(type="log", nticks=5, tickformat=".0e")
    
    if y_log:
        fig.update_yaxes(
            type="log",
            tickformat=".0e",
            exponentformat="e",
            showexponent="all",
        )


    save_png(fig, out_png)


# ----------------------------
# Main plotting pipelines
# ----------------------------
def make_tsteps_relerr_plots_single_dir(root: str, out_dir: str) -> None:
    micro_dt = load_meta_micro_dt(root)

    tstep_err_paths = sorted(glob.glob(os.path.join(root, "tsteps_errors__T*_vs_refT*.csv")))
    if not tstep_err_paths:
        print("[INFO] No tsteps_errors__... files found; skipping dt plots.")
        return

    x_dt, dP_mean, dP_std, dQ_mean, dQ_std = aggregate_rel_error_series(
        tstep_err_paths,
        x_from_path="tsteps",
        micro_dt=micro_dt,
    )

    plot_mean_std_band(
        x_dt, dP_mean, dP_std,
        title="Relative error in dP vs Δt (mean ± std over tasks)",
        xaxis_title="Δt_eff (micro_dt / Tsteps)",
        yaxis_title="rel_err_dP",
        out_png=os.path.join(out_dir, "relerr_dP_vs_dt.png"),
        x_log=True,
    )

    plot_mean_std_band(
        x_dt, dQ_mean, dQ_std,
        title="Relative error in ||dQ|| vs Δt (mean ± std over tasks)",
        xaxis_title="Δt_eff (micro_dt / Tsteps)",
        yaxis_title="rel_err_dQmag",
        out_png=os.path.join(out_dir, "relerr_dQ_vs_dt.png"),
        x_log=True,
    )


def make_mesh_relerr_plots_single_dir(root: str, out_dir: str) -> None:
    mesh_err_paths = sorted(glob.glob(os.path.join(root, "mesh_errors__n*_vs_*.csv")))
    if not mesh_err_paths:
        print("[INFO] No mesh_errors__... files found; skipping mesh relerr plots.")
        return

    x_n, dP_mean, dP_std, dQ_mean, dQ_std = aggregate_rel_error_series(
        mesh_err_paths,
        x_from_path="mesh",
        micro_dt=None,
    )

    plot_mean_std_band(
        x_n, dP_mean, dP_std,
        title="Relative error in dP vs mesh N (mean ± std over tasks)",
        xaxis_title="mesh N (n)",
        yaxis_title="rel_err_dP",
        out_png=os.path.join(out_dir, "relerr_dP_vs_meshN.png"),
        x_log=True,
        y_log=True,
    )

    plot_mean_std_band(
        x_n, dQ_mean, dQ_std,
        title="Relative error in ||dQ|| vs mesh N (mean ± std over tasks)",
        xaxis_title="mesh N (n)",
        yaxis_title="rel_err_dQmag",
        out_png=os.path.join(out_dir, "relerr_dQ_vs_meshN.png"),
        x_log=True,
        y_log=True,
    )


def make_ahratio_plots_single_dir(root: str, out_dir: str) -> None:
    # Prefer raw files because they give mean/std directly from task outputs
    raw_paths = sorted(glob.glob(os.path.join(root, "ahratio_raw__r*.csv")))
    if not raw_paths:
        print("[INFO] No ahratio_raw__... files found; skipping Ah-ratio plots.")
        return

    # Aggregate means/stds
    ratios: List[float] = []
    dP_mean: List[float] = []
    dP_std: List[float] = []
    dQ_mean: List[float] = []
    dQ_std: List[float] = []

    for p in sorted(raw_paths):
        base = os.path.basename(p)
        m = re.search(r"__r(.+)\.csv$", base)
        if not m:
            continue
        tag = m.group(1).replace("p", ".")
        try:
            r = float(tag)
        except Exception:
            continue

        rows = read_csv_rows(p)
        dP = col_as_float(rows, "dP")
        dQmag = col_as_float(rows, "dQmag")

        msP = mean_std(dP)
        msQ = mean_std(dQmag)

        ratios.append(r)
        dP_mean.append(msP.mean)
        dP_std.append(msP.std)
        dQ_mean.append(msQ.mean)
        dQ_std.append(msQ.std)

    if not ratios:
        print("[INFO] ahratio_raw__... found but could not parse; skipping Ah-ratio plots.")
        return

    rarr = np.asarray(ratios, float)
    order = np.argsort(rarr)
    rarr = rarr[order]
    dP_mean_arr = np.asarray(dP_mean, float)[order]
    dP_std_arr = np.asarray(dP_std, float)[order]
    dQ_mean_arr = np.asarray(dQ_mean, float)[order]
    dQ_std_arr = np.asarray(dQ_std, float)[order]

    plot_mean_std_band(
        rarr, dP_mean_arr, dP_std_arr,
        title="dP vs Ah ratio (Ah_task = r·H_task) (mean ± std over tasks)",
        xaxis_title="Ah ratio r",
        yaxis_title="dP",
        out_png=os.path.join(out_dir, "dP_vs_ah_ratio_mean_std.png"),
        x_log=False,
    )

    plot_mean_std_band(
        rarr, dQ_mean_arr, dQ_std_arr,
        title="||dQ|| vs Ah ratio (Ah_task = r·H_task) (mean ± std over tasks)",
        xaxis_title="Ah ratio r",
        yaxis_title="||dQ||",
        out_png=os.path.join(out_dir, "dQmag_vs_ah_ratio_mean_std.png"),
        x_log=False,
    )


# ----------------------------
# Diagnostics (data availability)
# ----------------------------
def print_data_diagnostics(root: str) -> None:
    """
    Prints a quick report of what we found / what might be missing for the plots.
    """
    want = {
        "tsteps errors": sorted(glob.glob(os.path.join(root, "tsteps_errors__T*_vs_refT*.csv"))),
        "mesh errors": sorted(glob.glob(os.path.join(root, "mesh_errors__n*_vs_*.csv"))),
        "ahratio raw": sorted(glob.glob(os.path.join(root, "ahratio_raw__r*.csv"))),
        "meta.json": [os.path.join(root, "meta.json")] if os.path.exists(os.path.join(root, "meta.json")) else [],
    }

    print("\n=== Data diagnostics (single-dir mode) ===")
    for k, v in want.items():
        if v:
            print(f"- {k}: {len(v)} file(s)")
        else:
            print(f"- {k}: MISSING")

    if not want["meta.json"]:
        print("  * Note: meta.json missing -> dt plots will use Δt_eff = 1/Tsteps (relative scale only).")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # EDIT THIS: directory containing all exported CSVs (single folder)
    ROOT = "."

    if not ROOT:
        raise RuntimeError("Please set ROOT to the directory containing your exported CSVs.")

    out_dir = os.path.join(ROOT, "plots_plotly")
    mkdir(out_dir)

    print_data_diagnostics(ROOT)

    make_tsteps_relerr_plots_single_dir(ROOT, out_dir)
    make_mesh_relerr_plots_single_dir(ROOT, out_dir)
    make_ahratio_plots_single_dir(ROOT, out_dir)

    print(f"\nDone. Wrote plots into: {out_dir}")


if __name__ == "__main__":
    main()