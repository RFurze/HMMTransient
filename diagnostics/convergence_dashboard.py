#!/usr/bin/env python3
"""EDAS Convergence Dashboard — post-run diagnostic and per-iteration monitor.

Usage
-----
Post-run analysis of a completed/aborted case::

    python diagnostics/convergence_dashboard.py --case BaseCase020326

Live monitoring (call from RUN_PENALTY.sh after each EDAS pass)::

    python diagnostics/convergence_dashboard.py \
        --case BaseCase020326 --live \
        --T 0.05 --lb_iter 1 --c_iter 2 --refine_iter 1

Outputs
-------
Writes a JSON summary to ``<case>/diagnostics/convergence_summary.json`` and,
when matplotlib is available, a multi-panel PNG to
``<case>/diagnostics/convergence_dashboard.png``.
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load(path):
    """Load a numpy file, returning None if it does not exist."""
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def _safe_load_txt(path):
    """Load a text file of floats, one per line."""
    if not os.path.exists(path):
        return np.array([])
    try:
        return np.loadtxt(path, ndmin=1)
    except Exception:
        return np.array([])


def _safe_load_pickle(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_coupling_convergence(exchange_dir):
    """Parse d_coupling_errs.txt and detect oscillation/stagnation."""
    errs = _safe_load_txt(os.path.join(exchange_dir, "d_coupling_errs.txt"))
    if errs.size == 0:
        return {}

    ratios = []
    for i in range(1, len(errs)):
        if abs(errs[i - 1]) > 1e-30:
            ratios.append(float(errs[i] / errs[i - 1]))
        else:
            ratios.append(float("nan"))

    divergence_events = [
        {"iteration": i + 1, "ratio": r, "error": float(errs[i + 1])}
        for i, r in enumerate(ratios)
        if r > 1.5
    ]
    stagnation_events = [
        {"iteration": i + 1, "ratio": r, "error": float(errs[i + 1])}
        for i, r in enumerate(ratios)
        if 0.9 < r <= 1.0
    ]

    return {
        "n_iterations": int(len(errs)),
        "errors": [float(e) for e in errs],
        "ratios": ratios,
        "final_error": float(errs[-1]),
        "divergence_events": divergence_events,
        "stagnation_events": stagnation_events,
        "monotone_from": _find_monotone_start(errs),
    }


def _find_monotone_start(errs):
    """Find the first iteration from which errors decrease monotonically."""
    for start in range(len(errs)):
        monotone = True
        for i in range(start + 1, len(errs)):
            if errs[i] >= errs[i - 1]:
                monotone = False
                break
        if monotone:
            return int(start)
    return int(len(errs))


def analyse_mls_error(exchange_dir):
    """Analyse the MLS error indicator distribution."""
    indicators = _safe_load(os.path.join(exchange_dir, "mls_error_indicators.npy"))
    if indicators is None:
        return {}

    pcts = [0.5, 0.9, 0.95, 0.99, 1.0]
    quantiles = {f"p{int(p*100)}": float(np.quantile(indicators, p)) for p in pcts}

    return {
        "n_nodes": int(len(indicators)),
        "max": float(indicators.max()),
        "mean": float(indicators.mean()),
        "std": float(indicators.std()),
        "quantiles": quantiles,
        "max_node": int(np.argmax(indicators)),
        "n_above_0.3": int(np.sum(indicators > 0.3)),
        "n_above_0.1": int(np.sum(indicators > 0.1)),
        "n_above_target": int(np.sum(indicators > 0.05)),
    }


def analyse_training_data(exchange_dir):
    """Characterise the EDAS training set — size, staleness, feature spread."""
    state = _safe_load_pickle(os.path.join(exchange_dir, "edas_state.npy"))
    if state is None:
        return {}

    existing = state.get("existing_xi_d")
    timestamps = state.get("timestamps")
    norm_state = state.get("normaliser", {})

    info = {}
    if existing is not None:
        info["n_training_points"] = int(existing.shape[1])
        feature_idx = [0, 1, 5, 6, 11, 12]
        feature_names = ["H", "P", "dPdx", "dPdy", "Hdot", "Pdot"]
        features = np.column_stack([existing[i] for i in feature_idx])
        for j, name in enumerate(feature_names):
            col = features[:, j]
            info[f"feature_{name}_range"] = [float(col.min()), float(col.max())]
            info[f"feature_{name}_std"] = float(col.std())

    if timestamps is not None:
        info["timestamp_min"] = float(timestamps.min())
        info["timestamp_max"] = float(timestamps.max())
        info["n_unique_timestamps"] = int(len(np.unique(timestamps)))

    rmin = norm_state.get("running_min")
    rmax = norm_state.get("running_max")
    if rmin is not None and rmax is not None:
        rng = rmax - rmin
        info["normaliser_ranges"] = [float(r) for r in rng]
        info["normaliser_min"] = [float(r) for r in rmin]
        info["normaliser_max"] = [float(r) for r in rmax]

    # Relevance weight analysis
    weights = _safe_load(os.path.join(exchange_dir, "edas_relevance_weights.npy"))
    if weights is not None and weights.size > 0:
        info["relevance_weight_min"] = float(weights.min())
        info["relevance_weight_mean"] = float(weights.mean())
        info["relevance_weight_max"] = float(weights.max())
        info["n_below_prune_threshold"] = int(np.sum(weights < 0.01))
        info["n_below_0.1"] = int(np.sum(weights < 0.1))
        info["n_below_0.5"] = int(np.sum(weights < 0.5))

    return info


def analyse_mls_flags(exchange_dir):
    """Check fallback rates for the primary correction fields."""
    results = {}
    for name in ("dP", "dQx", "dQy"):
        flag_path = os.path.join(exchange_dir, f"{name}_mls_flag.npy")
        flags = _safe_load(flag_path)
        if flags is not None:
            n_total = int(len(flags))
            n_fallback = int(flags.sum())
            results[name] = {
                "n_total": n_total,
                "n_fallback": n_fallback,
                "fallback_rate": float(n_fallback / n_total) if n_total > 0 else 0.0,
            }
    return results


def analyse_normaliser_stability(exchange_dir):
    """Check whether the SharedNormaliser bounds are stable or still expanding."""
    state = _safe_load_pickle(os.path.join(exchange_dir, "edas_state.npy"))
    if state is None:
        return {}

    norm = state.get("normaliser", {})
    rmin = norm.get("running_min")
    rmax = norm.get("running_max")
    if rmin is None or rmax is None:
        return {}

    xi_rot = _safe_load(os.path.join(exchange_dir, "xi_rot.npy"))
    if xi_rot is None:
        return {}

    feature_idx = [0, 1, 5, 6, 11, 12]
    feature_names = ["H", "P", "dPdx", "dPdy", "Hdot", "Pdot"]
    current_features = np.column_stack([xi_rot[i] for i in feature_idx])
    current_min = current_features.min(axis=0)
    current_max = current_features.max(axis=0)

    rng = rmax - rmin
    rng[rng < 1e-15] = 1.0

    stability = {}
    for j, name in enumerate(feature_names):
        # How much of the normaliser range is actually used by current data?
        used_fraction = (current_max[j] - current_min[j]) / rng[j]
        # Would current data expand the bounds?
        would_expand_min = current_min[j] < rmin[j]
        would_expand_max = current_max[j] > rmax[j]
        stability[name] = {
            "normaliser_range": float(rng[j]),
            "current_data_range": float(current_max[j] - current_min[j]),
            "utilisation": float(used_fraction),
            "would_expand_min": bool(would_expand_min),
            "would_expand_max": bool(would_expand_max),
        }
    return stability


def analyse_select_samples_efficiency(exchange_dir):
    """Estimate how many candidates were rejected by delta_min spacing."""
    indicators = _safe_load(os.path.join(exchange_dir, "mls_error_indicators.npy"))
    selected = _safe_load(os.path.join(exchange_dir, "edas_selected_indices.npy"))
    if indicators is None or selected is None:
        return {}

    n_query = len(indicators)
    n_selected = len(selected)

    # Check: are the selected points actually the highest-error ones?
    if n_selected > 0 and n_query > 0:
        top_k_indices = np.argsort(-indicators)[:n_selected]
        overlap = len(set(selected) & set(top_k_indices))
        top_k_overlap_frac = overlap / n_selected if n_selected > 0 else 0.0

        # Error at selected vs top-k
        err_at_selected = float(indicators[selected].mean()) if n_selected > 0 else 0.0
        err_at_top_k = float(indicators[top_k_indices].mean()) if n_selected > 0 else 0.0
    else:
        top_k_overlap_frac = 0.0
        err_at_selected = 0.0
        err_at_top_k = 0.0

    return {
        "n_query_nodes": n_query,
        "n_selected": n_selected,
        "selection_rate": float(n_selected / n_query) if n_query > 0 else 0.0,
        "top_k_overlap": float(top_k_overlap_frac),
        "mean_error_at_selected": err_at_selected,
        "mean_error_at_ideal_top_k": err_at_top_k,
        "error_efficiency_ratio": float(err_at_selected / err_at_top_k) if err_at_top_k > 0 else 0.0,
    }


def analyse_hdot_pdot_drift(exchange_dir):
    """Quantify how much Hdot/Pdot change between coupling iterations.

    This is critical because training data from earlier coupling iterations
    has stale Hdot/Pdot values while H, P, U, V, dpdx, dpdy remain fixed.
    """
    state = _safe_load_pickle(os.path.join(exchange_dir, "edas_state.npy"))
    xi_rot = _safe_load(os.path.join(exchange_dir, "xi_rot.npy"))
    if state is None or xi_rot is None:
        return {}

    existing = state.get("existing_xi_d")
    if existing is None:
        return {}

    # Current Hdot, Pdot from macroscale
    current_hdot = xi_rot[11]
    current_pdot = xi_rot[12]

    # Training Hdot, Pdot
    train_hdot = existing[11]
    train_pdot = existing[12]

    return {
        "current_hdot_range": [float(current_hdot.min()), float(current_hdot.max())],
        "current_pdot_range": [float(current_pdot.min()), float(current_pdot.max())],
        "training_hdot_range": [float(train_hdot.min()), float(train_hdot.max())],
        "training_pdot_range": [float(train_pdot.min()), float(train_pdot.max())],
        "hdot_current_std": float(current_hdot.std()),
        "pdot_current_std": float(current_pdot.std()),
        "hdot_training_std": float(train_hdot.std()),
        "pdot_training_std": float(train_pdot.std()),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_summary(exchange_dir):
    summary = {
        "coupling": analyse_coupling_convergence(exchange_dir),
        "mls_error": analyse_mls_error(exchange_dir),
        "training_data": analyse_training_data(exchange_dir),
        "mls_flags": analyse_mls_flags(exchange_dir),
        "normaliser_stability": analyse_normaliser_stability(exchange_dir),
        "sample_selection": analyse_select_samples_efficiency(exchange_dir),
        "hdot_pdot_drift": analyse_hdot_pdot_drift(exchange_dir),
    }
    return summary


def save_summary(summary, diag_dir):
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "convergence_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to {path}")
    return path


def print_report(summary):
    """Print a human-readable report to stdout."""
    print("\n" + "=" * 72)
    print("  EDAS CONVERGENCE DIAGNOSTIC REPORT")
    print("=" * 72)

    # Coupling convergence
    c = summary.get("coupling", {})
    if c:
        print(f"\n--- Coupling Convergence ({c.get('n_iterations', '?')} iterations) ---")
        print(f"  Final error:       {c.get('final_error', '?'):.4e}")
        print(f"  Divergence events: {len(c.get('divergence_events', []))}")
        for ev in c.get("divergence_events", []):
            print(f"    iter {ev['iteration']}: ratio={ev['ratio']:.2f}, err={ev['error']:.4e}")
        print(f"  Stagnation events: {len(c.get('stagnation_events', []))}")
        print(f"  Monotone from:     iter {c.get('monotone_from', '?')}")

    # MLS error
    m = summary.get("mls_error", {})
    if m:
        print(f"\n--- MLS Error Indicators ({m.get('n_nodes', '?')} nodes) ---")
        print(f"  Max:    {m.get('max', '?'):.4e}  (node {m.get('max_node', '?')})")
        print(f"  Mean:   {m.get('mean', '?'):.4e}")
        print(f"  Std:    {m.get('std', '?'):.4e}")
        print(f"  >0.3:   {m.get('n_above_0.3', '?')}")
        print(f"  >0.1:   {m.get('n_above_0.1', '?')}")
        print(f"  >target: {m.get('n_above_target', '?')}")
        q = m.get("quantiles", {})
        if q:
            print(f"  Quantiles: p50={q.get('p50','?'):.4f}  p90={q.get('p90','?'):.4f}  "
                  f"p95={q.get('p95','?'):.4f}  p99={q.get('p99','?'):.4f}")

    # Training data
    t = summary.get("training_data", {})
    if t:
        print(f"\n--- Training Data ---")
        print(f"  N training points:    {t.get('n_training_points', '?')}")
        print(f"  Unique timestamps:    {t.get('n_unique_timestamps', '?')}")
        print(f"  Relevance wt range:   [{t.get('relevance_weight_min', '?'):.4f}, "
              f"{t.get('relevance_weight_max', '?'):.4f}]")
        print(f"  Below prune thresh:   {t.get('n_below_prune_threshold', '?')}")
        print(f"  Below 0.5:            {t.get('n_below_0.5', '?')}")

    # Normaliser stability
    ns = summary.get("normaliser_stability", {})
    if ns:
        print(f"\n--- Normaliser Stability ---")
        for name, info in ns.items():
            util = info.get("utilisation", 0)
            flag = " ** LOW" if util < 0.3 else ""
            print(f"  {name:8s}: utilisation={util:.2%}, "
                  f"range={info.get('normaliser_range', 0):.4e}{flag}")

    # Sample selection efficiency
    ss = summary.get("sample_selection", {})
    if ss:
        print(f"\n--- Sample Selection Efficiency ---")
        print(f"  Selected:           {ss.get('n_selected', '?')}/{ss.get('n_query_nodes', '?')}")
        print(f"  Top-k overlap:      {ss.get('top_k_overlap', 0):.1%}")
        print(f"  Error efficiency:   {ss.get('error_efficiency_ratio', 0):.2f}")

    # Hdot/Pdot drift
    hp = summary.get("hdot_pdot_drift", {})
    if hp:
        print(f"\n--- Hdot/Pdot Drift ---")
        print(f"  Current Hdot range:  {hp.get('current_hdot_range', '?')}")
        print(f"  Training Hdot range: {hp.get('training_hdot_range', '?')}")
        print(f"  Current Pdot range:  {hp.get('current_pdot_range', '?')}")
        print(f"  Training Pdot range: {hp.get('training_pdot_range', '?')}")

    # MLS flags
    fl = summary.get("mls_flags", {})
    if fl:
        print(f"\n--- MLS Fallback Rates ---")
        for name, info in fl.items():
            print(f"  {name}: {info.get('n_fallback', 0)}/{info.get('n_total', 0)} "
                  f"({info.get('fallback_rate', 0):.1%} fallback)")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Plotting (optional, degrades gracefully)
# ---------------------------------------------------------------------------

def try_plot(summary, diag_dir):
    """Generate multi-panel dashboard PNG if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation.")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EDAS Convergence Dashboard", fontsize=14, fontweight="bold")

    # 1. Coupling error history
    ax = axes[0, 0]
    c = summary.get("coupling", {})
    errs = c.get("errors", [])
    if errs:
        ax.semilogy(range(1, len(errs) + 1), errs, "o-", color="steelblue")
        for ev in c.get("divergence_events", []):
            ax.axvline(ev["iteration"], color="red", alpha=0.3, linestyle="--")
        ax.set_xlabel("Coupling Iteration")
        ax.set_ylabel("Coupling Error")
        ax.set_title("Coupling Convergence")
        ax.grid(True, alpha=0.3)

    # 2. MLS error distribution
    ax = axes[0, 1]
    m = summary.get("mls_error", {})
    q = m.get("quantiles", {})
    if q:
        labels = list(q.keys())
        vals = list(q.values())
        ax.bar(labels, vals, color="coral")
        ax.axhline(0.05, color="green", linestyle="--", label="target")
        ax.set_ylabel("Error Indicator")
        ax.set_title("MLS Error Quantiles")
        ax.legend()

    # 3. Normaliser utilisation
    ax = axes[0, 2]
    ns = summary.get("normaliser_stability", {})
    if ns:
        names = list(ns.keys())
        utils = [ns[n]["utilisation"] for n in names]
        colors = ["red" if u < 0.3 else "orange" if u < 0.6 else "green" for u in utils]
        ax.barh(names, utils, color=colors)
        ax.set_xlabel("Utilisation (current / normaliser range)")
        ax.set_title("Normaliser Range Utilisation")
        ax.set_xlim(0, 1.1)

    # 4. Coupling error ratios
    ax = axes[1, 0]
    ratios = c.get("ratios", [])
    if ratios:
        ax.plot(range(2, len(ratios) + 2), ratios, "s-", color="purple")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Coupling Iteration")
        ax.set_ylabel("Error Ratio (iter / prev)")
        ax.set_title("Convergence Rate")
        ax.grid(True, alpha=0.3)

    # 5. Training data feature spreads
    ax = axes[1, 1]
    td = summary.get("training_data", {})
    feature_names = ["H", "P", "dPdx", "dPdy", "Hdot", "Pdot"]
    stds = [td.get(f"feature_{n}_std", 0) for n in feature_names]
    norm_ranges = td.get("normaliser_ranges", [1] * 6)
    if any(s > 0 for s in stds):
        relative_stds = [s / r if r > 0 else 0 for s, r in zip(stds, norm_ranges)]
        ax.bar(feature_names, relative_stds, color="teal")
        ax.set_ylabel("Std / Normaliser Range")
        ax.set_title("Feature Spread (relative)")
        ax.tick_params(axis="x", rotation=45)

    # 6. Sample selection efficiency
    ax = axes[1, 2]
    ss = summary.get("sample_selection", {})
    if ss:
        metrics = {
            "Selection\nRate": ss.get("selection_rate", 0),
            "Top-k\nOverlap": ss.get("top_k_overlap", 0),
            "Error\nEfficiency": ss.get("error_efficiency_ratio", 0),
        }
        ax.bar(metrics.keys(), metrics.values(), color="goldenrod")
        ax.set_ylim(0, 1.1)
        ax.set_title("Sample Selection Quality")

    plt.tight_layout()
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "convergence_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard plot saved to {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDAS Convergence Dashboard")
    parser.add_argument("--case", required=True, help="Case directory name (e.g. BaseCase020326)")
    parser.add_argument("--base", default="data/output/cases", help="Cases base directory")
    parser.add_argument("--live", action="store_true", help="Live monitoring mode (append to history)")
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--lb_iter", type=int, default=None)
    parser.add_argument("--c_iter", type=int, default=None)
    parser.add_argument("--refine_iter", type=int, default=None)
    args = parser.parse_args()

    case_root = os.path.join(args.base, args.case)
    exchange_dir = os.path.join(case_root, "run", "exchange")
    diag_dir = os.path.join(case_root, "diagnostics")

    if not os.path.isdir(exchange_dir):
        print(f"Exchange directory not found: {exchange_dir}")
        sys.exit(1)

    summary = build_summary(exchange_dir)

    if args.live and args.T is not None:
        summary["_meta"] = {
            "T": args.T,
            "lb_iter": args.lb_iter,
            "c_iter": args.c_iter,
            "refine_iter": args.refine_iter,
        }
        # Append to history file
        history_path = os.path.join(diag_dir, "convergence_history.jsonl")
        os.makedirs(diag_dir, exist_ok=True)
        with open(history_path, "a") as f:
            f.write(json.dumps(summary, default=str) + "\n")
        print(f"Appended to {history_path}")

    save_summary(summary, diag_dir)
    print_report(summary)
    try_plot(summary, diag_dir)


if __name__ == "__main__":
    main()
