#!/usr/bin/env python3
"""Grid-search ND_FACTOR and MLS_THETA using prepared 1:1 microscale results.

Workflow:
1. Build downsampled points from ``xi_rot.npy`` using repository ``MetaModel3``.
2. Map those downsampled points back to indices in the full rotated dataset.
3. Train MLS surrogates from the downsampled subset.
4. Predict all non-downsampled points.
5. Compare against full 1:1 values and report best ND/theta combinations.

python3 -m analysis.src.optimise_ndtheta_mls \
  --output_dir data/input/OneToOne/ \
  --transient \
  --nd_factors "0.2, 0.5, 0.6" \
  --theta_values "1000, 5000, 10000" \
  --k_neighbors 500 \
  --w_thresh 1e-3 \
  --match_tol 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from CONFIGPenalty import MLS_DEGREE
from coupling.src.functions.coupling_classes import MetaModel3 as SteadyMetaModel
from coupling.src.functions.multinom_MLS_par import multinom_coeffs
from coupling.src.functions.transient_coupling_classes import (
    MetaModel3 as TransientMetaModel,
)


@dataclass
class VariableSpec:
    """Metadata for each output variable predicted by MLS."""

    name: str
    values: np.ndarray
    degree_idx: int
    feature_idx: list[int]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(" ", "").split(",") if x]


def _as_feature_matrix(xi: np.ndarray, transient: bool) -> np.ndarray:
    """Feature extraction mirroring generate_MLS_tasks input construction."""
    base_idx = [0, 1, 5, 6]
    if transient:
        base_idx.extend([11, 12])
    return np.vstack([xi[i] for i in base_idx]).T


def _match_downsample_indices(X_full: np.ndarray, X_down: np.ndarray, tol: float) -> np.ndarray:
    """Match downsampled rows to full rows using nearest-neighbour lookup."""
    tree = cKDTree(X_full)
    dists, idx = tree.query(X_down, k=1)
    max_dist = float(np.max(dists))
    if np.any(dists > tol):
        raise ValueError(
            "Could not map some downsampled points back to xi_rot indices. "
            f"max_dist={max_dist:.3e}, tolerance={tol:.3e}."
        )
    if len(np.unique(idx)) != len(idx):
        raise ValueError(
            "Downsampled points mapped to duplicate full indices; "
            "try tighter tolerance or ensure unique source points."
        )
    print(f"Matched downsample indices (max distance {max_dist:.3e})")
    return idx.astype(int)


def _build_design(X: np.ndarray, Xi: np.ndarray, degree: int):
    """Build normalized design matrices for polynomial MLS."""
    n_train, n_features = X.shape

    X_norm = np.zeros_like(X, dtype=float)
    Xi_norm = np.zeros_like(Xi, dtype=float)
    for j in range(n_features):
        xmin = X[:, j].min()
        xmax = X[:, j].max()
        rng = (xmax - xmin) if xmax > xmin else 1.0
        X_norm[:, j] = (X[:, j] - xmin) / rng
        Xi_norm[:, j] = (Xi[:, j] - xmin) / rng

    coeffs, n_terms = multinom_coeffs(degree, n_features)
    mat_train = np.ones((n_train, n_terms), dtype=float)
    mat_query = np.ones((Xi.shape[0], n_terms), dtype=float)

    for i_term in range(n_terms):
        for j in range(n_features):
            exp_j = coeffs[i_term, j]
            if exp_j:
                mat_train[:, i_term] *= X_norm[:, j] ** exp_j
                mat_query[:, i_term] *= Xi_norm[:, j] ** exp_j

    return X_norm, Xi_norm, mat_train, mat_query


def _mls_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    degree: int,
    theta: float,
    k_neighbors: int,
    w_thresh: float,
) -> np.ndarray:
    """Evaluate MLS predictions for query points."""
    Xn, Xin, mat, mati = _build_design(X_train, X_query, degree)

    ymin = float(y_train.min())
    ymax = float(y_train.max())
    yrng = (ymax - ymin) if ymax > ymin else 1.0
    y_norm = (y_train - ymin) / yrng

    tree = cKDTree(Xn)
    k = min(k_neighbors, len(X_train))
    dist_all, idx_all = tree.query(Xin, k=k)
    if k == 1:
        dist_all = dist_all[:, None]
        idx_all = idx_all[:, None]

    n_query, n_terms = mati.shape
    coeff_pred = np.zeros((n_query, n_terms), dtype=float)

    for i in range(n_query):
        idx = idx_all[i]
        dist = dist_all[i]
        w = np.exp(-theta * dist**2)
        w_max = w.max()
        if w_max < 1e-15:
            continue

        mask = w >= (w_thresh * w_max)
        if np.count_nonzero(mask) < n_terms:
            mask = np.ones_like(mask, dtype=bool)

        mat_red = mat[idx][mask]
        y_red = y_norm[idx][mask]
        w_red = w[mask]

        matw = mat_red * w_red[:, None]
        pw = y_red * w_red
        alpha, *_ = np.linalg.lstsq(matw, pw, rcond=None)
        coeff_pred[i] = alpha

    y_query = np.sum(mati * coeff_pred, axis=1) * yrng + ymin
    return y_query


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true))
    if denom < 1e-15:
        return _rmse(y_true, y_pred)
    return _rmse(y_true, y_pred) / denom


def _max_abs_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def _max_rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe = np.where(np.abs(y_true) < 1e-15, 1.0, np.abs(y_true))
    return float(np.max(np.abs(y_true - y_pred) / safe))


def _iter_grid(nd_factors: Iterable[float], thetas: Iterable[float]):
    for nd in nd_factors:
        for theta in thetas:
            yield nd, theta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize ND_FACTOR and MLS_THETA on known 1:1 results"
    )
    parser.add_argument("--output_dir", type=str, default="OneToOne/")
    parser.add_argument(
        "--transient", action="store_true", help="Use transient xi layout"
    )
    parser.add_argument(
        "--nd_factors", type=str, required=True, help="Comma-separated ND_FACTOR values"
    )
    parser.add_argument(
        "--theta_values", type=str, required=True, help="Comma-separated theta values"
    )
    parser.add_argument("--k_neighbors", type=int, default=200)
    parser.add_argument("--w_thresh", type=float, default=1e-3)
    parser.add_argument("--match_tol", type=float, default=1e-12)
    parser.add_argument(
        "--prefer_high_nd_within",
        type=float,
        default=0.02,
        help="Prefer higher ND when mean NRMSE is within this relative margin of min.",
    )
    args = parser.parse_args()

    nd_factors = _parse_csv_floats(args.nd_factors)
    theta_values = _parse_csv_floats(args.theta_values)
    odir = args.output_dir

    xi_rot = np.load(os.path.join(odir, "xi_rot.npy"))
    X_full = _as_feature_matrix(xi_rot, transient=args.transient)
    n_full = X_full.shape[0]

    dq = np.load(os.path.join(odir, "dq_results.npy"))
    dp = np.load(os.path.join(odir, "dp_results.npy"))
    taust = np.load(os.path.join(odir, "tau_results.npy"))
    pmax = np.load(os.path.join(odir, "pmax_results.npy"))
    pmin = np.load(os.path.join(odir, "pmin_results.npy"))
    hmax = np.load(os.path.join(odir, "hmax_results.npy"))
    hmin = np.load(os.path.join(odir, "hmin_results.npy"))

    default_features = list(range(X_full.shape[1]))
    variables = [
        VariableSpec("dQx", dq[:, 0], 0, [0, 1, 2] + ([4, 5] if args.transient else [])),
        VariableSpec("dQy", dq[:, 1], 1, [0, 1, 3] + ([4, 5] if args.transient else [])),
        VariableSpec("dP", dp, 2, [0, 1, 2, 3] + ([4, 5] if args.transient else [])),
        VariableSpec("taustx", taust[:, 0], 3, default_features),
        VariableSpec("tausty", taust[:, 1], 4, default_features),
        VariableSpec("pmax", pmax, 5, default_features),
        VariableSpec("pmin", pmin, 6, default_features),
        VariableSpec("hmax", hmax, 7, default_features),
        VariableSpec("hmin", hmin, 8, default_features),
    ]

    model_cls = TransientMetaModel if args.transient else SteadyMetaModel
    records: list[dict] = []

    for nd, theta in _iter_grid(nd_factors, theta_values):
        print(f"Evaluating ND_FACTOR={nd}, theta={theta}")
        metamodel = model_cls(Nd_factor=nd)
        _, xi_d = metamodel.build(xi_rot, order=None, init=True, theta=None)

        X_down = _as_feature_matrix(xi_d, transient=args.transient)
        idx_down = _match_downsample_indices(X_full, X_down, tol=args.match_tol)

        mask_down = np.zeros(n_full, dtype=bool)
        mask_down[idx_down] = True
        idx_query = np.where(~mask_down)[0]
        if len(idx_query) == 0:
            raise ValueError(
                "Downsampling selected all points; cannot evaluate on held-out points."
            )

        per_var: dict[str, dict[str, float]] = {}
        nrmse_values = []
        max_abs_values = []
        max_rel_values = []

        for var in variables:
            y_train = var.values[idx_down]
            y_true = var.values[idx_query]
            X_train = X_down[:, var.feature_idx]
            X_query = X_full[idx_query][:, var.feature_idx]

            y_pred = _mls_predict(
                X_train,
                y_train,
                X_query,
                degree=int(MLS_DEGREE[var.degree_idx]),
                theta=float(theta),
                k_neighbors=args.k_neighbors,
                w_thresh=args.w_thresh,
            )

            rmse = _rmse(y_true, y_pred)
            nrmse = _nrmse(y_true, y_pred)
            max_abs_err = _max_abs_err(y_true, y_pred)
            max_rel_err = _max_rel_err(y_true, y_pred)

            per_var[var.name] = {
                "rmse": rmse,
                "nrmse": nrmse,
                "max_abs_err": max_abs_err,
                "max_rel_err": max_rel_err,
            }
            nrmse_values.append(nrmse)
            max_abs_values.append(max_abs_err)
            max_rel_values.append(max_rel_err)

            print(
                f"  {var.name}: rmse={rmse:.6e}, nrmse={nrmse:.6e}, "
                f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}"
            )

        mean_nrmse = float(np.mean(nrmse_values))
        downsample_ratio = float(len(idx_down) / n_full)
        record = {
            "nd_factor": float(nd),
            "theta": float(theta),
            "n_downsampled": int(len(idx_down)),
            "n_query": int(len(idx_query)),
            "downsample_ratio": downsample_ratio,
            "mean_nrmse": mean_nrmse,
            "max_abs_err": float(np.max(max_abs_values)),
            "max_rel_err": float(np.max(max_rel_values)),
            "per_variable": per_var,
        }
        records.append(record)
        print(
            f"Summary: downsample={downsample_ratio:.3f}, "
            f"mean_nrmse={mean_nrmse:.6e}\n"
        )

    records_sorted = sorted(records, key=lambda r: (r["mean_nrmse"], -r["nd_factor"]))
    global_best = records_sorted[0]

    min_err = global_best["mean_nrmse"]
    cutoff = min_err * (1.0 + args.prefer_high_nd_within)
    near_best = [r for r in records if r["mean_nrmse"] <= cutoff]
    preferred_high_nd = sorted(near_best, key=lambda r: (-r["nd_factor"], r["mean_nrmse"]))[0]

    payload = {
        "global_best": global_best,
        "preferred_high_nd": preferred_high_nd,
        "all_results": records_sorted,
        "settings": {
            "transient": bool(args.transient),
            "k_neighbors": int(args.k_neighbors),
            "w_thresh": float(args.w_thresh),
            "match_tol": float(args.match_tol),
            "prefer_high_nd_within": float(args.prefer_high_nd_within),
        },
    }

    out_json = os.path.join(odir, "nd_theta_optimization.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_csv = os.path.join(odir, "nd_theta_optimization.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "nd_factor,theta,n_downsampled,n_query,downsample_ratio,"
            "mean_nrmse,max_abs_err,max_rel_err\n"
        )
        for r in records_sorted:
            f.write(
                f"{r['nd_factor']},{r['theta']},{r['n_downsampled']},{r['n_query']},"
                f"{r['downsample_ratio']},{r['mean_nrmse']},{r['max_abs_err']},{r['max_rel_err']}\n"
            )

    print("=== Optimization summary ===")
    print(
        f"Global best: ND_FACTOR={global_best['nd_factor']}, theta={global_best['theta']}, "
        f"downsample_ratio={global_best['downsample_ratio']:.3f}, "
        f"mean_nrmse={global_best['mean_nrmse']:.6e}"
    )
    print(
        f"Preferred high-ND (within {args.prefer_high_nd_within*100:.1f}%): "
        f"ND_FACTOR={preferred_high_nd['nd_factor']}, theta={preferred_high_nd['theta']}, "
        f"downsample_ratio={preferred_high_nd['downsample_ratio']:.3f}, "
        f"mean_nrmse={preferred_high_nd['mean_nrmse']:.6e}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()