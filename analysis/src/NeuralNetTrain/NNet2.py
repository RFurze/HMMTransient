#!/usr/bin/env python3
"""
Train/test an sklearn neural network (MLPRegressor) on transient micro training data.

Inputs:
  transient_existing_xi_d.npy   -> features (xi_d)
  transient_existing_dp.npy     -> scalar target dp
  transient_existing_dq.npy     -> vector target dq (typically 2 components: dQx,dQy)

Key features:
- Hold-out evaluation with X% omitted from training
- Robust metrics that don't look artificially good due to "easy" regions
  (macro-averaged over bins, tail errors, optional cluster macro-averaging)
- Feature checks / ablation (e.g. "zeta" on/off) and permutation importance

Run:
  python3 nn_sklearn_test.py --holdout 0.2 --zeta-index 8
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import MiniBatchKMeans


# ----------------------------
# Utilities: loading + shaping
# ----------------------------
def _as_samples_by_features(X: np.ndarray) -> np.ndarray:
    """
    Ensure X is (n_samples, n_features).
    Accepts (n_features, n_samples) or (n_samples, n_features).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    # Heuristic: if rows << cols and you expect many samples, transpose
    if X.shape[0] < X.shape[1]:
        # Could be (n_features, n_samples)
        # But could also be (n_samples, n_features) with n_samples < n_features (rare here).
        # We'll assume your usual layout is features x samples.
        return X.T
    return X


def _as_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and 1 in y.shape:
        y = y.reshape(-1)
    elif y.ndim != 1:
        raise ValueError(f"dp must be 1D (or squeezable), got shape {y.shape}")
    return y


def _as_samples_by_outputs(Y: np.ndarray) -> np.ndarray:
    """
    Ensure Y is (n_samples, n_outputs).
    Accepts (n_outputs, n_samples) or (n_samples, n_outputs) or (n_samples,) -> (n_samples,1).
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        return Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 1D or 2D, got shape {Y.shape}")
    if Y.shape[0] < Y.shape[1]:
        # likely (n_outputs, n_samples)
        return Y.T
    return Y


# ----------------------------
# Robust / "not easy-skewed" metrics
# ----------------------------
def _bin_edges_from_quantiles(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bins; duplicates handled by unique."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(v, qs)
    edges = np.unique(edges)
    # Need at least 2 edges to form bins
    if edges.size < 2:
        # fallback: single bin
        edges = np.array([v.min(), v.max() + 1e-12])
    return edges


def macro_mae_over_magnitude_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 8,
    eps: float = 1e-30,
) -> float:
    """
    Macro-average MAE over bins of |y_true|.
    Each bin contributes equally (so "easy small-magnitude" regions don't dominate).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mag = np.abs(y_true)
    edges = _bin_edges_from_quantiles(mag, n_bins=n_bins)

    # digitize into bins 0..(k-2), where k=len(edges)
    idx = np.clip(np.digitize(mag, edges[1:-1], right=False), 0, len(edges) - 2)

    maes = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        maes.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    if not maes:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(maes))


def tail_abs_error(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.95) -> float:
    """Quantile of absolute error (e.g. 95th percentile)."""
    err = np.abs(np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1))
    return float(np.quantile(err, q))


def cluster_macro_mae(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_clusters: int = 16,
    random_state: int = 0,
) -> float:
    """
    Macro-average MAE over clusters in feature space (equal weight per cluster),
    to avoid being dominated by dense "easy" regions.
    """
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=4096)
    labels = km.fit_predict(X)

    maes = []
    for c in range(n_clusters):
        mask = labels == c
        if not np.any(mask):
            continue
        maes.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    if not maes:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(maes))


# ----------------------------
# Training / evaluation
# ----------------------------
@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float
    macro_mae_magbins: float
    p95_abs_err: float
    cluster_macro_mae: float


def compute_metrics(
    X_eval: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mag_bins: int,
    cluster_k: int,
) -> Metrics:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    macro_bins = macro_mae_over_magnitude_bins(y_true, y_pred, n_bins=mag_bins)
    p95 = tail_abs_error(y_true, y_pred, q=0.95)
    cmae = cluster_macro_mae(X_eval, y_true, y_pred, n_clusters=cluster_k)
    return Metrics(rmse, mae, r2, macro_bins, p95, cmae)


def build_model(hidden=(256, 256), alpha=1e-4, max_iter=2000, random_state=0) -> Pipeline:
    """
    StandardScaler + MLPRegressor.
    - 'adam' solver works well generally.
    - early_stopping uses an internal validation split from the training data.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
        verbose=False,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", default="analysis/src/NeuralNetTrain/transient_existing_xi_d.npy")
    ap.add_argument("--dp", default="analysis/src/NeuralNetTrain/transient_existing_dp.npy")
    ap.add_argument("--dq", default="analysis/src/NeuralNetTrain/transient_existing_dq.npy")

    ap.add_argument("--holdout", type=float, default=0.2, help="fraction held out for evaluation")
    ap.add_argument("--seed", type=int, default=0)

    # Robust-metric knobs
    ap.add_argument("--mag-bins", type=int, default=8, help="bins for macro MAE by |target|")
    ap.add_argument("--cluster-k", type=int, default=16, help="clusters for cluster macro MAE")

    # Feature checks
    ap.add_argument("--zeta-index", type=int, default=None, help="column index of 'zeta' in X (0-based)")
    ap.add_argument(
        "--ablate-zeta",
        action="store_true",
        help="run an extra training run with zeta omitted and compare metrics",
    )
    ap.add_argument(
        "--perm-importance",
        action="store_true",
        help="compute permutation importance on holdout for each output (slow-ish)",
    )

    # Model knobs
    ap.add_argument("--hidden", default="256,256", help="hidden sizes, e.g. 128,128")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--max-iter", type=int, default=2000)

    args = ap.parse_args()

    X_raw = np.load(args.xi)
    dp_raw = np.load(args.dp)
    dq_raw = np.load(args.dq)

    X = _as_samples_by_features(X_raw)

    dp = _as_1d(dp_raw)
    dq = _as_samples_by_outputs(dq_raw)  # (n_samples, n_out)

    if X.shape[0] != dp.shape[0] or X.shape[0] != dq.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X {X.shape}, dp {dp.shape}, dq {dq.shape}"
        )

    # Compose multi-output target: [dp, dq...]
    Y = np.concatenate([dp.reshape(-1, 1), dq], axis=1)

    hidden = tuple(int(s) for s in args.hidden.split(",") if s.strip())
    model = build_model(hidden=hidden, alpha=args.alpha, max_iter=args.max_iter, random_state=args.seed)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=args.holdout, random_state=args.seed, shuffle=True
    )

    model.fit(X_tr, Y_tr)
    Y_hat = model.predict(X_te)

    # Report per-output metrics (dp, dq1, dq2,...)
    names = ["dp"] + [f"dq{i}" for i in range(1, Y.shape[1])]
    report = {}

    for j, name in enumerate(names):
        m = compute_metrics(
            X_eval=X_te,
            y_true=Y_te[:, j],
            y_pred=Y_hat[:, j],
            mag_bins=args.mag_bins,
            cluster_k=args.cluster_k,
        )
        report[name] = m.__dict__

    print("\n=== Holdout evaluation (robust) ===")
    print(json.dumps(report, indent=2))

    # Optional: ablation test (omit zeta)
    if args.ablate_zeta:
        if args.zeta_index is None:
            raise ValueError("--ablate-zeta requires --zeta-index")
        if not (0 <= args.zeta_index < X.shape[1]):
            raise ValueError(f"zeta-index {args.zeta_index} out of range for X with {X.shape[1]} features")

        keep = np.ones(X.shape[1], dtype=bool)
        keep[args.zeta_index] = False

        Xk = X[:, keep]
        Xk_tr, Xk_te, Yk_tr, Yk_te = train_test_split(
            Xk, Y, test_size=args.holdout, random_state=args.seed, shuffle=True
        )

        model2 = build_model(hidden=hidden, alpha=args.alpha, max_iter=args.max_iter, random_state=args.seed)
        model2.fit(Xk_tr, Yk_tr)
        Yk_hat = model2.predict(Xk_te)

        report2 = {}
        for j, name in enumerate(names):
            m = compute_metrics(
                X_eval=Xk_te,
                y_true=Yk_te[:, j],
                y_pred=Yk_hat[:, j],
                mag_bins=args.mag_bins,
                cluster_k=args.cluster_k,
            )
            report2[name] = m.__dict__

        print("\n=== Ablation: zeta removed ===")
        print(json.dumps(report2, indent=2))

        # Simple “did it help?” summary using your robust metric
        print("\n=== Delta (zeta-removed minus baseline) on macro_mae_magbins (lower is better) ===")
        for name in names:
            base = report[name]["macro_mae_magbins"]
            ablt = report2[name]["macro_mae_magbins"]
            print(f"{name:>4s}: {ablt - base:+.6e}")

    # Optional: permutation importance on holdout
    if args.perm_importance:
        # Permutation importance is defined for single-output, so do it per output.
        # We'll use the same fitted model; for each output we compute a score drop.
        # Note: This can be slow on big datasets.
        print("\n=== Permutation importance (holdout) ===")
        for j, name in enumerate(names):
            def score_fn(est, Xp, yp):
                pred = est.predict(Xp)[:, j]
                # robust-ish scoring: negative macro MAE (higher is better)
                return -macro_mae_over_magnitude_bins(yp, pred, n_bins=args.mag_bins)

            pi = permutation_importance(
                model, X_te, Y_te[:, j],
                scoring=score_fn,
                n_repeats=10,
                random_state=args.seed,
                n_jobs=-1,
            )
            importances = pi.importances_mean
            order = np.argsort(importances)[::-1]
            topk = min(15, len(importances))
            print(f"\nTop features for {name} (by importance on -macroMAE):")
            for k in range(topk):
                i = int(order[k])
                print(f"  feat[{i:02d}]  importance={importances[i]:+.4e}")

    # Save model
    out = Path("mlp_model.joblib")
    try:
        import joblib
        joblib.dump(model, out)
        print(f"\nSaved trained model to: {out}")
    except Exception as e:
        print(f"\nNote: couldn't save model (joblib missing?). Error: {e}")


if __name__ == "__main__":
    main()