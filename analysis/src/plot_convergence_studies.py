#!/usr/bin/env python3
"""
Plot convergence study outputs produced by micro_convergence_studies.py.

Reads from:
  - mesh_raw.csv        (optional)
  - tsteps_raw.csv      (optional)
  - ahratio_raw.csv     (optional)

Writes plots to:
  - <OUTPUT_DIR>/mesh/*.png
  - <OUTPUT_DIR>/tsteps/*.png
  - <OUTPUT_DIR>/ahratio/*.png

Uses house style helpers from plotly_formats.py (UPDATED VERSION).
Mesh x-axis uses mesh_n^2 as requested.

Plots (per-run, task-specific normalisation; no "ref run" selection):
  - dP_over_Pre   = dP / P_re
  - dQx_over_Qxre = dQx / Qx_re
  - dQy_over_Qyre = dQy / Qy_re
  - abs_dQ_over_abs_Qre = |dQ| / |Q_re|

Also plots ABS(values) with log-y:
  abs(dP/P_re), abs(dQx/Qx_re), abs(dQy/Qy_re), abs(|dQ|/|Q_re|)

Notes:
- Ratios may be NaN when denominators are ~0 in the studies script.
- For log-y plots we require strictly positive values (we mask <= 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly_formats import (
    PALETTE,
    AxisSpec,
    apply_house_style,
    add_envelope_fill,
    add_line,
    save_png,
    PRESETS,
)

# ---------------------------------------------------------------------
# CONFIG (edit these)
# ---------------------------------------------------------------------

INPUT_DIR = Path(".")         # folder containing mesh_raw.csv etc.
OUTPUT_DIR = None             # default: INPUT_DIR / "plots"

PLOT_MESH = True
PLOT_TSTEPS = True
PLOT_AHRATIO = True

# Choose style preset for saved figures (THESIS / PPT_16_9 / DRAFT / THESIS_SQUARE)
FIG_PRESET_NAME = "THESIS"
FIGSPEC = PRESETS.get(FIG_PRESET_NAME, PRESETS["THESIS"])

# PNG export scale (kaleido)
PNG_SCALE = 3

# Small epsilon used only for log-y masking
POS_EPS = 1e-30

# What we plot (must exist or be derivable)
NORM_VARS_TO_PLOT = [
    "dP_over_Pre",
    "dQx_over_Qxre",
    "dQy_over_Qyre",
    "abs_dQ_over_abs_Qre",
]

ABS_NORM_VARS_TO_PLOT = [
    "abs_dP_over_Pre",
    "abs_dQx_over_Qxre",
    "abs_dQy_over_Qyre",
    "abs_abs_dQ_over_abs_Qre",
]

# Signed normalised quantities can cross 0 -> linear-y
LOGY_SIGNED = False

# Abs versions -> log-y
LOGY_ABS = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _nice_linear_ticks(vmin: float, vmax: float, nticks: int = 7) -> np.ndarray:
    """
    Generate "nice-ish" linear ticks using a simple linspace.
    Ensures monotonic increasing and returns unique ticks.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([])
    if vmin == vmax:
        return np.array([vmin])
    ticks = np.linspace(vmin, vmax, nticks)
    # de-dup in case of tiny ranges
    ticks = np.unique(ticks)
    return ticks


def _apply_zero_tick_label(
    fig: go.Figure,
    *,
    y_values: np.ndarray,
    nticks: int = 7,
    zero_tol: float = 1e-12,
    fmt: str = ".3g",
) -> None:
    """
    For linear y-axes: enforce a 0 tick and replace any tick values with |v|<zero_tol with '0'.
    Avoids -4.44e-17 style labels from Plotly's auto ticks.
    """
    yv = np.asarray(y_values, dtype=float)
    yv = yv[np.isfinite(yv)]
    if yv.size == 0:
        return

    vmin = float(np.min(yv))
    vmax = float(np.max(yv))

    ticks = _nice_linear_ticks(vmin, vmax, nticks=nticks)

    # If the range crosses 0 (or is very close), ensure 0 is included
    if (vmin <= 0.0 <= vmax) or (abs(vmin) < zero_tol) or (abs(vmax) < zero_tol):
        ticks = np.unique(np.sort(np.concatenate([ticks, np.array([0.0])])))

    # Build tick text; snap near-zero to "0"
    ticktext = []
    for t in ticks:
        if abs(float(t)) < zero_tol:
            ticktext.append("0")
        else:
            # d3-format mini-language
            ticktext.append(f"{t:{fmt}}")

    fig.update_yaxes(
        tickmode="array",
        tickvals=ticks.tolist(),
        ticktext=ticktext,
    )


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # indices / sweep params
    for c in ["mesh_n", "n_steps", "task_index"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c]).astype("Int64")

    if "ratio" in df.columns:
        df["ratio"] = _safe_numeric(df["ratio"])

    # required raw columns (some may not exist depending on run)
    for c in [
        "dP", "dQx", "dQy",
        "P_re", "Qx_re", "Qy_re",
        "dP_over_Pre", "dQx_over_Qxre", "dQy_over_Qyre",
    ]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c])

    # derive |dQ| and |Q_re| and their ratio
    if ("dQx" in df.columns) and ("dQy" in df.columns):
        df["abs_dQ"] = np.sqrt(df["dQx"].astype(float) ** 2 + df["dQy"].astype(float) ** 2)
    if ("Qx_re" in df.columns) and ("Qy_re" in df.columns):
        df["abs_Qre"] = np.sqrt(df["Qx_re"].astype(float) ** 2 + df["Qy_re"].astype(float) ** 2)

    if ("abs_dQ" in df.columns) and ("abs_Qre" in df.columns):
        # keep NaN if denominator is ~0
        den = df["abs_Qre"].astype(float)
        num = df["abs_dQ"].astype(float)
        out = np.full(len(df), np.nan, dtype=float)
        good = np.isfinite(num.to_numpy()) & np.isfinite(den.to_numpy()) & (np.abs(den.to_numpy()) > POS_EPS)
        out[good] = num.to_numpy()[good] / den.to_numpy()[good]
        df["abs_dQ_over_abs_Qre"] = out

    # Abs variants for log-y plotting
    for c in ["dP_over_Pre", "dQx_over_Qxre", "dQy_over_Qyre", "abs_dQ_over_abs_Qre"]:
        if c in df.columns:
            df[f"abs_{c}"] = np.abs(df[c].astype(float))
            # log safety: kill zeros / negatives (abs should be >=0)
            df.loc[df[f"abs_{c}"].astype(float) <= POS_EPS, f"abs_{c}"] = np.nan

    # For readability, match requested ABS list names:
    # abs_abs_dQ_over_abs_Qre corresponds to abs(abs_dQ_over_abs_Qre)
    if "abs_abs_dQ_over_abs_Qre" not in df.columns and "abs_abs_dQ_over_abs_Qre" in ABS_NORM_VARS_TO_PLOT:
        if "abs_abs_dQ_over_abs_Qre" not in df.columns and "abs_abs_dQ_over_abs_Qre" not in df.columns:
            pass  # handled by abs_ prefix rule above

    return df


def _add_task_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    group_col: str = "task_index",
    opacity: float = 0.25,
) -> None:
    for _, g in df.groupby(group_col):
        gg = g.sort_values(xcol)
        fig.add_trace(go.Scatter(
            x=gg[xcol].to_numpy(),
            y=gg[ycol].to_numpy(),
            mode="lines+markers",
            line=dict(width=1.2, color="rgba(0,0,0,0.25)"),
            marker=dict(size=4, color="rgba(0,0,0,0.25)"),
            showlegend=False,
            hoverinfo="skip",
            opacity=opacity,
        ))


def _add_mean_std(
    fig: go.Figure,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    name_mean: str = "Mean",
    name_band: str = "±1 std",
) -> None:
    x_env = np.concatenate([x, x[::-1]])
    lo = mean - std
    hi = mean + std
    y_env = np.concatenate([hi, lo[::-1]])

    add_envelope_fill(fig, x_env, y_env, name=name_band)
    add_line(fig, x, mean, name=name_mean, color=PALETTE["red_main"], width=2.6)


def _ylabel(ycol: str) -> str:
    mapping: Dict[str, str] = {
        "dP_over_Pre": "dP / P_re",
        "dQx_over_Qxre": "dQx / Qx_re",
        "dQy_over_Qyre": "dQy / Qy_re",
        "abs_dQ_over_abs_Qre": "|dQ| / |Q_re|",
        "abs_dP_over_Pre": "|dP / P_re|",
        "abs_dQx_over_Qxre": "|dQx / Qx_re|",
        "abs_dQy_over_Qyre": "|dQy / Qy_re|",
        "abs_abs_dQ_over_abs_Qre": "||dQ| / |Q_re||",
        "abs_abs_dQ_over_abs_Qre": "| |dQ| / |Q_re| |",
        "abs_abs_dQ_over_abs_Qre": "| |dQ| / |Q_re| |",
        "abs_abs_dQ_over_abs_Qre": "| |dQ| / |Q_re| |",
        "abs_abs_dQ_over_abs_Qre": "| |dQ| / |Q_re| |",
        "abs_abs_dQ_over_abs_Qre": "| |dQ| / |Q_re| |",
    }
    if ycol in mapping:
        return mapping[ycol]
    # generic abs_ prefix
    if ycol.startswith("abs_"):
        base = ycol[len("abs_"):]
        return f"|{mapping.get(base, base)}|"
    return ycol


def _write_study_plots(
    df: pd.DataFrame,
    *,
    study: str,
    xcol: str,
    xaxis_title: str,
    out_dir: Path,
    ycols_to_plot: List[str],
    logx: bool,
    logy: bool,
    title_prefix: str,
) -> None:
    study_dir = out_dir / study
    ensure_dir(study_dir)

    df = df.copy()
    df = df[np.isfinite(df[xcol].to_numpy(dtype=float))]
    if df.empty:
        return

    for ycol in ycols_to_plot:
        if ycol not in df.columns:
            continue

        dff = df[[xcol, "task_index", ycol]].dropna()
        if dff.empty:
            continue

        if logy:
            dff = dff[dff[ycol].astype(float) > 0.0]
            if dff.empty:
                continue

        stats = dff.groupby(xcol)[ycol].agg(["mean", "std"]).reset_index().sort_values(xcol)
        x = stats[xcol].to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        std = stats["std"].fillna(0.0).to_numpy(dtype=float)

        fig = go.Figure()
        _add_task_traces(fig, dff, xcol=xcol, ycol=ycol, group_col="task_index")
        _add_mean_std(fig, x, mean, std)

        xspec = AxisSpec(title=xaxis_title, tickformat=".2g", log=logx, nticks=(6 if logx else None))
        yspec = AxisSpec(title=_ylabel(ycol), tickformat=".3g", log=logy, nticks=(6 if logy else None))

        apply_house_style(
            fig,
            figspec=FIGSPEC,
            title=f"{title_prefix} | {study} | {ycol}",
            xaxis=xspec,
            yaxis=yspec,
            legend=True,
            hover=False,
        )

        # Fix near-zero tick labels on linear-y axes (avoids -4.44e-17 instead of 0)
        if not logy:
            _apply_zero_tick_label(
                fig,
                y_values=dff[ycol].to_numpy(dtype=float),
                nticks=7,
                zero_tol=1e-12,
                fmt=".3g",
            )


        suffix = f"logx{int(logx)}_logy{int(logy)}"
        out_path = study_dir / f"{study}__{ycol}__{suffix}.png"
        save_png(fig, str(out_path), scale=PNG_SCALE)


def _load_if_exists(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    return _prep_df(pd.read_csv(p))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR if OUTPUT_DIR is not None else (in_dir / "plots")
    ensure_dir(out_dir)

    mesh = _load_if_exists(in_dir / "mesh_raw.csv")
    tsteps = _load_if_exists(in_dir / "tsteps_raw.csv")
    ahr = _load_if_exists(in_dir / "ahratio_raw.csv")

    title_prefix = FIG_PRESET_NAME

    # --------------------
    # Mesh study
    # --------------------
    if PLOT_MESH and (mesh is not None) and (not mesh.empty):
        mesh = mesh.copy()
        mesh["mesh_n2"] = (mesh["mesh_n"].astype(float) ** 2)

        # Signed (linear y)
        _write_study_plots(
            mesh,
            study="mesh",
            xcol="mesh_n2",
            xaxis_title="Mesh nodes per cell (n²)",
            out_dir=out_dir,
            ycols_to_plot=NORM_VARS_TO_PLOT,
            logx=True,
            logy=LOGY_SIGNED,
            title_prefix=title_prefix,
        )

        # Abs (log y)
        _write_study_plots(
            mesh,
            study="mesh",
            xcol="mesh_n2",
            xaxis_title="Mesh nodes per cell (n²)",
            out_dir=out_dir,
            ycols_to_plot=[f"abs_{c}" for c in NORM_VARS_TO_PLOT],
            logx=True,
            logy=LOGY_ABS,
            title_prefix=title_prefix,
        )

    # --------------------
    # Time-step study
    # --------------------
    if PLOT_TSTEPS and (tsteps is not None) and (not tsteps.empty):
        tsteps = tsteps.copy()
        tsteps["n_steps_f"] = tsteps["n_steps"].astype(float)

        _write_study_plots(
            tsteps,
            study="tsteps",
            xcol="n_steps_f",
            xaxis_title="Number of time steps (n_steps)",
            out_dir=out_dir,
            ycols_to_plot=NORM_VARS_TO_PLOT,
            logx=True,
            logy=LOGY_SIGNED,
            title_prefix=title_prefix,
        )

        _write_study_plots(
            tsteps,
            study="tsteps",
            xcol="n_steps_f",
            xaxis_title="Number of time steps (n_steps)",
            out_dir=out_dir,
            ycols_to_plot=[f"abs_{c}" for c in NORM_VARS_TO_PLOT],
            logx=True,
            logy=LOGY_ABS,
            title_prefix=title_prefix,
        )

    # --------------------
    # Ah-ratio sweep
    # --------------------
    if PLOT_AHRATIO and (ahr is not None) and (not ahr.empty):
        ahr = ahr.copy()
        ahr["ratio_f"] = ahr["ratio"].astype(float)

        _write_study_plots(
            ahr,
            study="ahratio",
            xcol="ratio_f",
            xaxis_title="Ah ratio (Ah = ratio · H0)",
            out_dir=out_dir,
            ycols_to_plot=NORM_VARS_TO_PLOT,
            logx=False,
            logy=LOGY_SIGNED,
            title_prefix=title_prefix,
        )

        _write_study_plots(
            ahr,
            study="ahratio",
            xcol="ratio_f",
            xaxis_title="Ah ratio (Ah = ratio · H0)",
            out_dir=out_dir,
            ycols_to_plot=[f"abs_{c}" for c in NORM_VARS_TO_PLOT],
            logx=False,
            logy=LOGY_ABS,
            title_prefix=title_prefix,
        )

    print(f"[DONE] Plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
