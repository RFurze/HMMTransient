#!/usr/bin/env python3
"""Plot time-series from live_history.jsonl collected during a run.

Usage::

    python diagnostics/plot_live_history.py --case BaseCase020326

Generates ``<case>/diagnostics/live_history_plots.png``.
"""

import argparse
import json
import os
import sys

import numpy as np


def load_history(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--base", default="data/output/cases")
    args = parser.parse_args()

    history_path = os.path.join(args.base, args.case, "diagnostics", "live_history.jsonl")
    if not os.path.exists(history_path):
        print(f"No history file found at {history_path}")
        sys.exit(1)

    records = load_history(history_path)
    if not records:
        print("No records found.")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        sys.exit(1)

    # Build arrays
    idx = list(range(len(records)))
    mls_max = [r.get("mls_max_error") for r in records]
    coupling = [r.get("coupling_error") for r in records]
    n_train = [r.get("n_training") for r in records]
    n_tasks = [r.get("n_new_tasks") for r in records]
    n_fb = [r.get("n_fallback_dP") for r in records]

    # Labels for x-axis: T/c/r
    labels = [f"T{r.get('T','?')}\nc{r.get('c_iter','?')}\nr{r.get('refine_iter','?')}" for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(records) * 0.3), 12), sharex=True)

    # Panel 1: MLS max error + coupling error
    ax = axes[0]
    valid_mls = [(i, v) for i, v in zip(idx, mls_max) if v is not None]
    valid_coupling = [(i, v) for i, v in zip(idx, coupling) if v is not None]
    if valid_mls:
        ax.semilogy(*zip(*valid_mls), "o-", label="MLS max error", color="coral")
    if valid_coupling:
        ax.semilogy(*zip(*valid_coupling), "s-", label="Coupling error", color="steelblue")
    ax.axhline(0.05, color="green", linestyle="--", alpha=0.5, label="EDAS target")
    ax.axhline(1e-5, color="blue", linestyle="--", alpha=0.5, label="Coupling tol")
    ax.set_ylabel("Error")
    ax.legend(fontsize=8)
    ax.set_title("Error Convergence Over EDAS Iterations")
    ax.grid(True, alpha=0.3)

    # Panel 2: Training set size
    ax = axes[1]
    valid_train = [(i, v) for i, v in zip(idx, n_train) if v is not None]
    if valid_train:
        ax.plot(*zip(*valid_train), "D-", color="teal", label="Training set size")
    ax2 = ax.twinx()
    valid_tasks = [(i, v) for i, v in zip(idx, n_tasks) if v is not None]
    if valid_tasks:
        ax2.bar([v[0] for v in valid_tasks], [v[1] for v in valid_tasks],
                alpha=0.3, color="orange", label="New tasks")
        ax2.set_ylabel("New tasks per pass", color="orange")
    ax.set_ylabel("Total training points")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Fallback rate
    ax = axes[2]
    valid_fb = [(i, v) for i, v in zip(idx, n_fb) if v is not None]
    if valid_fb:
        ax.bar([v[0] for v in valid_fb], [v[1] for v in valid_fb], color="red", alpha=0.5)
    ax.set_ylabel("dP fallback nodes")
    ax.set_xlabel("EDAS iteration")
    ax.grid(True, alpha=0.3)

    # Use abbreviated labels for sparse ticks
    if len(records) <= 30:
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=6, rotation=90)

    plt.tight_layout()
    out_path = os.path.join(args.base, args.case, "diagnostics", "live_history_plots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
