#!/usr/bin/env python3
"""EDAS parameter sensitivity analyser.

Reads an existing EDAS state and the current macroscale state, then
evaluates how the error indicator and sample selection change under
different parameter choices.  No simulations are run — this is a
purely offline "what-if" tool.

Usage
-----
    python diagnostics/edas_parameter_sensitivity.py --case BaseCase020326

Outputs a JSON report and optional plots to ``<case>/diagnostics/``.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coupling.src.functions.edas import (
    SharedNormaliser,
    compute_error_indicators,
    compute_relevance_weights,
    select_samples,
    initial_coverage_sample,
)


def _safe_load(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def _safe_load_pickle(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


FEATURE_IDX = [0, 1, 5, 6, 11, 12]
FEATURE_NAMES = ["H", "P", "dPdx", "dPdy", "Hdot", "Pdot"]


def load_data(exchange_dir):
    """Load everything needed for offline sensitivity analysis."""
    xi_rot = _safe_load(os.path.join(exchange_dir, "xi_rot.npy"))
    state = _safe_load_pickle(os.path.join(exchange_dir, "edas_state.npy"))
    mls_errors = _safe_load(os.path.join(exchange_dir, "mls_error_indicators.npy"))

    if xi_rot is None or state is None:
        return None

    norm = SharedNormaliser.from_state(state["normaliser"])
    existing = state.get("existing_xi_d")
    timestamps = state.get("timestamps")

    X_query_raw = np.column_stack([xi_rot[i] for i in FEATURE_IDX])
    norm.update(X_query_raw)
    X_query_norm = norm.transform(X_query_raw)

    X_train_norm = None
    if existing is not None and existing.shape[1] > 0:
        X_train_raw = np.column_stack([existing[i] for i in FEATURE_IDX])
        X_train_norm = norm.transform(X_train_raw)

    return {
        "X_query_norm": X_query_norm,
        "X_train_norm": X_train_norm,
        "mls_errors": mls_errors,
        "timestamps": timestamps,
        "existing": existing,
        "norm": norm,
    }


def sweep_alpha_blend(data, alphas=None):
    """Sweep alpha_blend to see how coverage vs prediction weighting affects errors."""
    if alphas is None:
        alphas = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]

    X_train = data["X_train_norm"]
    X_query = data["X_query_norm"]
    if X_train is None or X_train.shape[0] < 4:
        return {}

    results = {}
    for alpha in alphas:
        eps = compute_error_indicators(
            X_train, np.zeros(X_train.shape[0]),
            X_query, theta=10000.0, degree=2,
            alpha_blend=alpha,
        )
        results[f"alpha={alpha:.1f}"] = {
            "max": float(eps.max()),
            "mean": float(eps.mean()),
            "p95": float(np.quantile(eps, 0.95)),
            "n_above_0.05": int(np.sum(eps > 0.05)),
        }
    return results


def sweep_delta_min_quantile(data, quantiles=None, batch_size=500):
    """Sweep delta_min_quantile to see how spacing constraints affect selection."""
    if quantiles is None:
        quantiles = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    X_query = data["X_query_norm"]
    X_train = data["X_train_norm"]
    mls_errors = data["mls_errors"]

    if mls_errors is None:
        return {}

    results = {}
    for q in quantiles:
        if q > 0 and X_query.shape[0] >= 2:
            tree = cKDTree(X_query)
            d_nn, _ = tree.query(X_query, k=2)
            delta_min = float(np.quantile(d_nn[:, 1], q))
        else:
            delta_min = 0.0

        selected = select_samples(
            mls_errors, X_query, X_train,
            batch_size=batch_size,
            delta_min=delta_min,
        )
        mean_err = float(mls_errors[selected].mean()) if len(selected) > 0 else 0.0
        results[f"q={q:.2f}"] = {
            "delta_min": delta_min,
            "n_selected": int(len(selected)),
            "mean_error_selected": mean_err,
        }
    return results


def sweep_lambda_decay(data, lambdas=None, current_time=0.05):
    """Sweep lambda_decay to see how time-decay affects training data pruning."""
    if lambdas is None:
        lambdas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    timestamps = data["timestamps"]
    X_train = data["X_train_norm"]
    X_query = data["X_query_norm"]

    if timestamps is None or X_train is None:
        return {}

    results = {}
    for lam in lambdas:
        weights = compute_relevance_weights(
            timestamps, current_time,
            X_train, X_query,
            lambda_decay=lam, sigma_spatial=0.3,
        )
        results[f"lambda={lam:.1f}"] = {
            "weight_min": float(weights.min()),
            "weight_mean": float(weights.mean()),
            "n_below_0.01": int(np.sum(weights < 0.01)),
            "n_below_0.1": int(np.sum(weights < 0.1)),
            "n_below_0.5": int(np.sum(weights < 0.5)),
            "effective_n": float(weights.sum()),
        }
    return results


def sweep_sigma_spatial(data, sigmas=None, current_time=0.05):
    """Sweep sigma_spatial to see how spatial relevance affects pruning."""
    if sigmas is None:
        sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    timestamps = data["timestamps"]
    X_train = data["X_train_norm"]
    X_query = data["X_query_norm"]

    if timestamps is None or X_train is None:
        return {}

    results = {}
    for sigma in sigmas:
        weights = compute_relevance_weights(
            timestamps, current_time,
            X_train, X_query,
            lambda_decay=2.0, sigma_spatial=sigma,
        )
        results[f"sigma={sigma:.2f}"] = {
            "weight_min": float(weights.min()),
            "weight_mean": float(weights.mean()),
            "n_below_0.01": int(np.sum(weights < 0.01)),
            "n_below_0.5": int(np.sum(weights < 0.5)),
            "effective_n": float(weights.sum()),
        }
    return results


def analyse_feature_importance(data):
    """Estimate which features dominate the error indicator via ablation.

    For each feature, zero it out in the normalised space and measure the
    change in error indicator distribution.
    """
    X_train = data["X_train_norm"]
    X_query = data["X_query_norm"]
    if X_train is None or X_train.shape[0] < 4:
        return {}

    # Baseline
    eps_base = compute_error_indicators(
        X_train, np.zeros(X_train.shape[0]),
        X_query, theta=10000.0, degree=2, alpha_blend=1.0,
    )
    base_max = float(eps_base.max())
    base_mean = float(eps_base.mean())

    results = {"baseline_max": base_max, "baseline_mean": base_mean}

    for j, name in enumerate(FEATURE_NAMES):
        # Zero out feature j in both train and query
        X_train_mod = X_train.copy()
        X_query_mod = X_query.copy()
        X_train_mod[:, j] = 0.5  # set to midpoint
        X_query_mod[:, j] = 0.5

        eps_mod = compute_error_indicators(
            X_train_mod, np.zeros(X_train_mod.shape[0]),
            X_query_mod, theta=10000.0, degree=2, alpha_blend=1.0,
        )
        results[name] = {
            "max_without": float(eps_mod.max()),
            "mean_without": float(eps_mod.mean()),
            "max_change": float(eps_mod.max() - base_max),
            "mean_change": float(eps_mod.mean() - base_mean),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="EDAS Parameter Sensitivity")
    parser.add_argument("--case", required=True)
    parser.add_argument("--base", default="data/output/cases")
    parser.add_argument("--current_time", type=float, default=0.05)
    args = parser.parse_args()

    exchange_dir = os.path.join(args.base, args.case, "run", "exchange")
    diag_dir = os.path.join(args.base, args.case, "diagnostics")

    data = load_data(exchange_dir)
    if data is None:
        print(f"Could not load EDAS state from {exchange_dir}")
        sys.exit(1)

    print("Running parameter sensitivity sweeps...")
    report = {
        "alpha_blend_sweep": sweep_alpha_blend(data),
        "delta_min_sweep": sweep_delta_min_quantile(data),
        "lambda_decay_sweep": sweep_lambda_decay(data, current_time=args.current_time),
        "sigma_spatial_sweep": sweep_sigma_spatial(data, current_time=args.current_time),
        "feature_importance": analyse_feature_importance(data),
    }

    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "parameter_sensitivity.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSensitivity report written to {path}")

    # Print key findings
    print("\n=== Key Findings ===")

    # Lambda decay
    ld = report.get("lambda_decay_sweep", {})
    if ld:
        print("\nLambda decay → effective training set size:")
        for k, v in ld.items():
            print(f"  {k}: effective_n={v['effective_n']:.0f}, "
                  f"below_0.01={v['n_below_0.01']}")

    # Delta min
    dm = report.get("delta_min_sweep", {})
    if dm:
        print("\nDelta min quantile → samples selected (of 500 budget):")
        for k, v in dm.items():
            print(f"  {k}: n_selected={v['n_selected']}, "
                  f"mean_err={v['mean_error_selected']:.4f}")

    # Feature importance
    fi = report.get("feature_importance", {})
    if fi:
        print(f"\nFeature importance (baseline coverage max={fi.get('baseline_max', 0):.4f}):")
        for name in FEATURE_NAMES:
            info = fi.get(name, {})
            print(f"  {name:8s}: max_change={info.get('max_change', 0):+.4f}")


if __name__ == "__main__":
    main()
