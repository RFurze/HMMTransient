#!/usr/bin/env python3
"""Grid-search ND_FACTOR with sklearn surrogates using prepared 1:1 microscale results.

Workflow:
1. Build downsampled points from ``xi_rot.npy`` using repository ``MetaModel3``.
2. Map those downsampled points back to indices in the full rotated dataset.
3. Train sklearn surrogates from the downsampled subset (one model per output).
4. Predict all non-downsampled points.
5. Compare against full 1:1 values and report best ND/model/hyperparameter combinations.

Example:
python3 -m analysis.src.optimise_ndtheta_sklearn \
  --output_dir data/input/OneToOne/ \
  --transient \
  --nd_factors "0.2,0.5,0.6" \
  --models "knn,rf,gpr" \
  --knn_neighbors "20,50,100" \
  --rf_trees "200,500" \
  --gpr_alpha "1e-8,1e-6"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coupling.src.functions.coupling_classes import MetaModel3 as SteadyMetaModel
from coupling.src.functions.transient_coupling_classes import (
    MetaModel3 as TransientMetaModel,
)


@dataclass
class VariableSpec:
    """Metadata for each output variable predicted by a surrogate."""

    name: str
    values: np.ndarray
    feature_idx: list[int]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(" ", "").split(",") if x]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _parse_csv_strings(text: str) -> list[str]:
    return [x for x in text.replace(" ", "").split(",") if x]


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


def _iter_grid(nd_factors: Iterable[float], model_grid: list[dict]):
    for nd in nd_factors:
        for model_cfg in model_grid:
            yield nd, model_cfg


def _build_model_grid(args: argparse.Namespace) -> list[dict]:
    model_names = _parse_csv_strings(args.models)
    grid: list[dict] = []

    for model_name in model_names:
        if model_name == "knn":
            for n_neighbors in _parse_csv_ints(args.knn_neighbors):
                grid.append({"model": "knn", "n_neighbors": n_neighbors})
        elif model_name == "rf":
            for n_estimators in _parse_csv_ints(args.rf_trees):
                for max_depth in _parse_csv_strings(args.rf_depths):
                    depth_val = None if max_depth.lower() == "none" else int(max_depth)
                    grid.append(
                        {
                            "model": "rf",
                            "n_estimators": n_estimators,
                            "max_depth": depth_val,
                        }
                    )
        elif model_name == "gpr":
            for alpha in _parse_csv_floats(args.gpr_alpha):
                for length_scale in _parse_csv_floats(args.gpr_length_scale):
                    grid.append(
                        {
                            "model": "gpr",
                            "alpha": alpha,
                            "length_scale": length_scale,
                        }
                    )
        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported: knn, rf, gpr"
            )

    if not grid:
        raise ValueError("No model configurations were generated from --models and grids.")
    return grid


def _build_regressor(model_cfg: dict):
    model_name = model_cfg["model"]

    if model_name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=int(model_cfg["n_neighbors"]),
                weights="distance",
            ),
        )

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=model_cfg["max_depth"],
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "gpr":
        length_scale = float(model_cfg["length_scale"])
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=float(model_cfg["alpha"]), noise_level_bounds="fixed")
        )
        return make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                normalize_y=True,
                random_state=42,
            ),
        )

    raise ValueError(f"Unknown model configuration: {model_cfg}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize ND_FACTOR and sklearn surrogate settings on known 1:1 results"
    )
    parser.add_argument("--output_dir", type=str, default="OneToOne/")
    parser.add_argument(
        "--transient", action="store_true", help="Use transient xi layout"
    )
    parser.add_argument(
        "--nd_factors", type=str, required=True, help="Comma-separated ND_FACTOR values"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="knn,rf,gpr",
        help="Comma-separated model families to evaluate: knn, rf, gpr",
    )

    parser.add_argument("--knn_neighbors", type=str, default="20,50,100")
    parser.add_argument("--rf_trees", type=str, default="200,500")
    parser.add_argument("--rf_depths", type=str, default="none,20")
    parser.add_argument("--gpr_alpha", type=str, default="1e-8,1e-6")
    parser.add_argument("--gpr_length_scale", type=str, default="0.5,1.0,2.0")

    parser.add_argument("--match_tol", type=float, default=1e-12)
    parser.add_argument(
        "--prefer_high_nd_within",
        type=float,
        default=0.02,
        help="Prefer higher ND when mean NRMSE is within this relative margin of min.",
    )
    args = parser.parse_args()

    nd_factors = _parse_csv_floats(args.nd_factors)
    model_grid = _build_model_grid(args)
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
        VariableSpec("dQx", dq[:, 0], [0, 1, 2] + ([4, 5] if args.transient else [])),
        VariableSpec("dQy", dq[:, 1], [0, 1, 3] + ([4, 5] if args.transient else [])),
        VariableSpec("dP", dp, [0, 1, 2, 3] + ([4, 5] if args.transient else [])),
        VariableSpec("taustx", taust[:, 0], default_features),
        VariableSpec("tausty", taust[:, 1], default_features),
        VariableSpec("pmax", pmax, default_features),
        VariableSpec("pmin", pmin, default_features),
        VariableSpec("hmax", hmax, default_features),
        VariableSpec("hmin", hmin, default_features),
    ]

    model_cls = TransientMetaModel if args.transient else SteadyMetaModel
    records: list[dict] = []

    for nd, model_cfg in _iter_grid(nd_factors, model_grid):
        print(f"Evaluating ND_FACTOR={nd}, model_cfg={model_cfg}")
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

            regressor = _build_regressor(model_cfg)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_query)

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
            "model": model_cfg["model"],
            "model_cfg": model_cfg,
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
            "match_tol": float(args.match_tol),
            "prefer_high_nd_within": float(args.prefer_high_nd_within),
        },
    }

    out_json = os.path.join(odir, "nd_sklearn_optimization.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_csv = os.path.join(odir, "nd_sklearn_optimization.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "nd_factor,model,model_cfg,n_downsampled,n_query,downsample_ratio,"
            "mean_nrmse,max_abs_err,max_rel_err\n"
        )
        for r in records_sorted:
            f.write(
                f"{r['nd_factor']},{r['model']},\"{json.dumps(r['model_cfg'])}\","
                f"{r['n_downsampled']},{r['n_query']},{r['downsample_ratio']},"
                f"{r['mean_nrmse']},{r['max_abs_err']},{r['max_rel_err']}\n"
            )

    print("=== Optimization summary ===")
    print(
        f"Global best: ND_FACTOR={global_best['nd_factor']}, model_cfg={global_best['model_cfg']}, "
        f"downsample_ratio={global_best['downsample_ratio']:.3f}, "
        f"mean_nrmse={global_best['mean_nrmse']:.6e}"
    )
    print(
        f"Preferred high-ND (within {args.prefer_high_nd_within*100:.1f}%): "
        f"ND_FACTOR={preferred_high_nd['nd_factor']}, model_cfg={preferred_high_nd['model_cfg']}, "
        f"downsample_ratio={preferred_high_nd['downsample_ratio']:.3f}, "
        f"mean_nrmse={preferred_high_nd['mean_nrmse']:.6e}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()