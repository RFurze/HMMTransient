#!/usr/bin/env python3
"""Tune MLS theta values via k-fold cross validation.

This standalone script loads the transient ``existing`` arrays from
``--output_dir`` and performs k-fold validation to determine the best
MLS ``theta`` parameter for each predicted quantity (dQx, dQy, dP, sst,
taustx, tausty).
"""

import argparse
import os
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from CONFIG import MLS_THETA, MLS_DEGREE
from coupling.src.functions.transient_coupling_classes import MetaModel3
from coupling.src.functions.multinom_MLS_par import multinom_coeffs


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _normalize_columns(X: np.ndarray, Xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Sequence[Tuple[float, float]]]:
    """Normalize X and Xi column wise returning the scaled arrays and
    list of (min,max) per column."""
    m = X.shape[1]
    Xn = np.zeros_like(X, dtype=float)
    Xin = np.zeros_like(Xi, dtype=float)
    ranges = []
    for j in range(m):
        xmin, xmax = X[:, j].min(), X[:, j].max()
        rng = (xmax - xmin) if (xmax > xmin) else 1.0
        Xn[:, j] = (X[:, j] - xmin) / rng
        Xin[:, j] = (Xi[:, j] - xmin) / rng
        ranges.append((xmin, xmax))
    return Xn, Xin, ranges


def _design_matrices(X: np.ndarray, Xi: np.ndarray, degree: int):
    Xn, Xin, _ = _normalize_columns(X, Xi)
    C, Nt = multinom_coeffs(degree, X.shape[1])
    Mat = np.ones((X.shape[0], Nt), dtype=float)
    Mati = np.ones((Xi.shape[0], Nt), dtype=float)
    for iexp in range(Nt):
        for j in range(X.shape[1]):
            e = C[iexp, j]
            if e:
                Mat[:, iexp] *= Xn[:, j] ** e
                Mati[:, iexp] *= Xin[:, j] ** e
    return Xn, Xin, Mat, Mati


def _normalize_y(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    ymin = y.min()
    ymax = y.max()
    yrng = (ymax - ymin) if (ymax > ymin) else 1.0
    return (y - ymin) / yrng, ymin, yrng


def mls_predict(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, theta: float, degree: int, k_neighbors: int = 200, w_thresh: float = 1e-3) -> np.ndarray:
    """Predict ``y`` for ``X_val`` using weighted least squares MLS."""
    Xn_tr, Xn_val, Mat, Mati = _design_matrices(X_train, X_val, degree)
    y_norm, y_min, y_rng = _normalize_y(y_train)

    tree = cKDTree(Xn_tr)
    k = min(len(Xn_tr), k_neighbors)
    dist_all, idx_all = tree.query(Xn_val, k=k)

    Nt = Mat.shape[1]
    coeffs = np.zeros((Xn_val.shape[0], Nt), dtype=float)
    for i in range(Xn_val.shape[0]):
        idx = idx_all[i]
        dist = dist_all[i]
        w = np.exp(-theta * dist ** 2)
        w_max = w.max()
        if w_max < 1e-15:
            continue
        mask = w >= w_thresh * w_max
        if np.count_nonzero(mask) < Nt:
            mask = np.ones_like(mask, dtype=bool)
        Mat_red = Mat[idx[mask]]
        y_red = y_norm[idx[mask]]
        w_red = w[mask]
        Matw = Mat_red * w_red[:, None]
        Pw = y_red * w_red
        alpha, *_ = np.linalg.lstsq(Matw, Pw, rcond=None)
        coeffs[i] = alpha

    y_pred_norm = np.sum(Mati * coeffs, axis=1)
    return y_pred_norm * y_rng + y_min


def k_fold_theta(X: np.ndarray, y: np.ndarray, thetas: Sequence[float], degree: int, k_folds: int = 5, k_neighbors: int = 200) -> Tuple[float, dict]:
    """Return best theta and mapping of theta->mean r2 score."""
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[: n % k_folds] += 1
    folds = []
    current = 0
    for fs in fold_sizes:
        folds.append(indices[current : current + fs])
        current += fs

    scores = {}
    best_theta = None
    best_score = -np.inf

    for theta in thetas:
        fold_scores = []
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            preds = mls_predict(X_tr, y_tr, X_val, theta, degree, k_neighbors)
            y_mean = np.mean(y_val)
            ss_res = np.sum((y_val - preds) ** 2)
            ss_tot = np.sum((y_val - y_mean) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fold_scores.append(r2)
        mean_r2 = float(np.mean(fold_scores))
        scores[theta] = mean_r2
        if mean_r2 > best_score:
            best_score = mean_r2
            best_theta = theta
    return best_theta, scores


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune MLS theta using k-fold validation")
    parser.add_argument("--output_dir", type=str, default="data/output/hmm_job", help="Directory containing existing transient data")
    parser.add_argument("--thetas", type=str, default=",".join(str(t) for t in MLS_THETA), help="Comma separated list of theta values to try")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--k_neighbors", type=int, default=200, help="Number of nearest neighbors")
    args = parser.parse_args()

    theta_list = [float(t) for t in args.thetas.replace(" ", ",").split(",") if t]

    odir = args.output_dir
    xi_d = np.load(os.path.join(odir, "transient_existing_xi_d.npy"))
    dq = np.load(os.path.join(odir, "transient_existing_dq.npy"))
    dp = np.load(os.path.join(odir, "transient_existing_dp.npy"))
    sst = np.load(os.path.join(odir, "transient_existing_sst.npy"))
    taust = np.load(os.path.join(odir, "transient_existing_taust.npy"))

    metamodel = MetaModel3()
    metamodel.existing_xi_d = xi_d
    metamodel.load_results(dq, dp, sst, taust)

    X = metamodel.get_training_matrix()

    variables = [
        ("dQx", dq[:, 0], MLS_DEGREE[0]),
        ("dQy", dq[:, 1], MLS_DEGREE[1]),
        ("dP", dp, MLS_DEGREE[2]),
        ("sst", sst, MLS_DEGREE[3]),
        ("taustx", taust[:, 0], MLS_DEGREE[4]),
        ("tausty", taust[:, 1], MLS_DEGREE[5]),
    ]

    best = []
    for name, y, deg in variables:
        theta, table = k_fold_theta(X, y, theta_list, deg, k_folds=args.k_folds, k_neighbors=args.k_neighbors)
        best.append(theta)
        print(f"{name}: best theta = {theta}")
        for t, score in table.items():
            print(f"  theta={t}: mean R2={score:.4f}")

    np.save(os.path.join(odir, "optimal_theta.npy"), np.array(best))
    print("Saved best theta values to optimal_theta.npy")


if __name__ == "__main__":
    main()