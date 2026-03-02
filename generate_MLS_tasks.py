"""Merge microscale results into the MLS training set and build MLS query tasks.

The script reads ``*_results.npy`` produced by :mod:`2_micro_sims`, appends
them to existing training arrays and emits ``*_tasks.npz`` bundles.  Each
bundle contains the task list and normalised matrices required by
:mod:`4_run_MLS.py` to evaluate MLS predictions.

**Transient path** — uses the EDAS shared normaliser for consistent
feature scaling between the sampler and MLS.  Also saves relevance
weights and normaliser state for the MLS error indicator computation.

**Steady path** — unchanged from original behaviour.

Inputs read from ``--output_dir``:
    ``dq_results.npy`` - flux increments ``dQ``.
    ``dp_results.npy`` - pressure corrections ``dP``.
    ``Fst.npy`` - fluid fraction increments.
    ``tau_results.npy`` - shear stress increments.
    ``pmax_results.npy``, ``pmin_results.npy``, ``hmax_results.npy`` and
    ``hmin_results.npy``.

Outputs written back to ``--output_dir``:
    ``existing_*.npy`` - accumulated training data.
    ``*_tasks.npz`` - per-variable MLS query tasks.
    (transient) ``edas_relevance_weights.npy`` - per-training-point weights.

Command line options mirror those of :func:`utils.cli.parse_common_args`.
"""

import os
import numpy as np
import sys

from utils.cli import parse_common_args
from CONFIGPenalty import MLS_THETA, MLS_DEGREE


def create_multinom_tasks(X, Y, Xi, theta, n, normaliser_state=None, verbose=False):
    """Create tasks for parallel MLS solution.

    If *normaliser_state* is provided (from EDAS), the feature normalisation
    uses the shared bounds so it is consistent with the sampler.  Otherwise
    the original per-column min/max normalisation is used.
    """
    N, m = X.shape
    Ni = Xi.shape[0]

    if N != len(Y):
        raise ValueError(
            f"Training inputs (X) have {N} samples but Y has {len(Y)} entries"
        )

    # --- Feature normalisation -------------------------------------------
    if normaliser_state is not None:
        # Use EDAS shared normaliser bounds
        rmin = normaliser_state["running_min"]
        rmax = normaliser_state["running_max"]
        rng = rmax - rmin
        rng[rng < 1e-15] = 1.0
        X_ = (X - rmin) / rng
        Xi_ = (Xi - rmin) / rng
    else:
        # Original per-column normalisation
        X_ = np.zeros_like(X, dtype=float)
        Xi_ = np.zeros_like(Xi, dtype=float)
        for j in range(m):
            xcol = X[:, j]
            xmin, xmax = xcol.min(), xcol.max()
            rng = (xmax - xmin) if (xmax > xmin) else 1.0
            X_[:, j] = (xcol - xmin) / rng
            Xi_[:, j] = (Xi[:, j] - xmin) / rng

    # --- Response normalisation ------------------------------------------
    Ymin, Ymax = Y.min(), Y.max()
    Yrng = (Ymax - Ymin) if (Ymax > Ymin) else 1.0
    Y_ = (Y - Ymin) / Yrng

    # --- Polynomial basis ------------------------------------------------
    from coupling.src.functions.multinom_MLS_par import multinom_coeffs

    C, Nt = multinom_coeffs(n, m, verbose=verbose)

    Mat = np.ones((N, Nt), dtype=float)
    Mati = np.ones((Ni, Nt), dtype=float)
    for i_exp in range(Nt):
        for j_col in range(m):
            exp_j = C[i_exp, j_col]
            if exp_j != 0:
                Mat[:, i_exp] *= X_[:, j_col] ** exp_j
                Mati[:, i_exp] *= Xi_[:, j_col] ** exp_j

    # Construct the task list by pairing each query point with the training data.
    tasks = []
    for i_q in range(Ni):
        tasks.append((i_q, X_, Y_, Mat, theta))

    return tasks, Mati, Ymin, Yrng, Xi_


def main():
    verbose = False
    args = parse_common_args("Build Microscale Task List", with_time=True)

    lb_iter = args.lb_iter
    c_iter = args.c_iter
    T = args.Time
    DT = args.DT
    transient = args.transient

    if transient:
        prefix = "transient_"
        print(
            f"Starting build_task_list.py for T={T}, lb_iter={lb_iter}, c_iter={c_iter}"
        )
    else:
        from coupling.src.functions.coupling_classes import MetaModel3 as MetaModel

        prefix = ""
        print(f"Starting build_task_list.py for lb_iter={lb_iter}, c_iter={c_iter}")

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load microscale results and the rotated input vector xi.
    result_files = {
        "dq": "dq_results.npy",
        "dp": "dp_results.npy",
        "taust": "tau_results.npy",
        "pmax": "pmax_results.npy",
        "pmin": "pmin_results.npy",
        "hmax": "hmax_results.npy",
        "hmin": "hmin_results.npy",
    }
    results = {
        name: np.load(os.path.join(output_dir, fname))
        for name, fname in result_files.items()
    }
    existing_xi_d = np.load(os.path.join(output_dir, f"{prefix}existing_xi_d.npy"))
    xi_rot = np.load(os.path.join(output_dir, "xi_rot.npy"))

    data_keys = ["dq", "dp", "taust", "pmax", "pmin", "hmax", "hmin"]
    existing = {}
    if transient:
        first_file = os.path.join(output_dir, f"{prefix}existing_{data_keys[0]}.npy")
        if not os.path.exists(first_file):
            print("No existing results to concatenate, creating new files.")
            sys.stdout.flush()
            existing = {k: results[k].copy() for k in data_keys}
        else:
            existing = {
                k: np.load(os.path.join(output_dir, f"{prefix}existing_{k}.npy"))
                for k in data_keys
            }
            for k in data_keys:
                existing[k] = np.concatenate((existing[k], results[k]), axis=0)
    else:
        if c_iter == 1 and lb_iter == 1:
            existing = {k: results[k].copy() for k in data_keys}
        else:
            existing = {
                k: np.load(os.path.join(output_dir, f"existing_{k}.npy"))
                for k in data_keys
            }
            for k in data_keys:
                existing[k] = np.concatenate((existing[k], results[k]), axis=0)

    for k, arr in existing.items():
        np.save(os.path.join(output_dir, f"{prefix}existing_{k}.npy"), arr)

    # --- Build training matrix -------------------------------------------
    if transient:
        # Use EDAS training matrix: columns [H, P, dPdx, dPdy, Hdot, Pdot]
        from coupling.src.functions.edas import ErrorDrivenSampler

        edas_state_file = os.path.join(output_dir, "edas_state.npy")
        edas_state = np.load(edas_state_file, allow_pickle=True).item()
        sampler = ErrorDrivenSampler.from_state(edas_state)
        X = sampler.get_training_matrix()

        # Load shared normaliser state for consistent feature normalisation
        norm_state_file = os.path.join(output_dir, "edas_normaliser_state.npy")
        normaliser_state = np.load(norm_state_file, allow_pickle=True).item()

        # Compute and save relevance weights for MLS weighting
        timestamps_file = os.path.join(output_dir, "edas_timestamps.npy")
        if os.path.exists(timestamps_file):
            timestamps = np.load(timestamps_file)
            from coupling.src.functions.edas import (
                SharedNormaliser,
                compute_relevance_weights,
            )

            norm = SharedNormaliser.from_state(normaliser_state)
            rot_indices = [0, 1, 5, 6, 11, 12]
            X_query_raw = np.vstack([xi_rot[i] for i in rot_indices]).T
            norm.update(X_query_raw)
            X_query_norm = norm.transform(X_query_raw)
            X_train_norm = norm.transform(X)

            rel_weights = compute_relevance_weights(
                timestamps,
                T,
                X_train_norm,
                X_query_norm,
                lambda_decay=sampler.lambda_decay,
                sigma_spatial=sampler.sigma_spatial,
            )
            np.save(os.path.join(output_dir, "edas_relevance_weights.npy"), rel_weights)
        else:
            rel_weights = np.ones(X.shape[0])
            np.save(os.path.join(output_dir, "edas_relevance_weights.npy"), rel_weights)
    else:
        metamodel = MetaModel()
        metamodel.existing_xi_d = existing_xi_d
        metamodel.load_results(
            existing["dq"],
            existing["dp"],
            existing["taust"],
            existing["pmax"],
            existing["pmin"],
            existing["hmax"],
            existing["hmin"],
        )
        X = metamodel.get_training_matrix()
        normaliser_state = None

    print("Assembling training and evaluation matrices")
    rot_indices = [0, 1, 5, 6]
    if transient:
        rot_indices.extend([11, 12])
    X_rot = np.vstack([xi_rot[i] for i in rot_indices]).T
    print(f'Size of training data: {X.shape}')

    feature_names = ["H", "P", "dPdx", "dPdy"]
    dQx_feature_names = ["H", "P", "dPdx", "dPdy"]
    dQy_feature_names = ["H", "P", "dPdx", "dPdy"]
    dP_feature_names = ["H", "P", "dPdx", "dPdy"]
    if transient:
        feature_names.extend(["Hdot", "Pdot"])
        dQx_feature_names.extend(["Hdot", "Pdot"])
        dQy_feature_names.extend(["Hdot", "Pdot"])
        dP_feature_names.extend(["Hdot", "Pdot"])

    default_features = list(range(len(feature_names)))
    feature_sets = {
        "dQx": dQx_feature_names,
        "dQy": dQy_feature_names,
        "dP": dP_feature_names,
    }

    def resolve_feature_indices(name):
        requested = feature_sets.get(name, default_features)
        if not requested:
            raise ValueError(f"Feature set for {name} is empty.")
        if isinstance(requested[0], str):
            feature_map = {feat: idx for idx, feat in enumerate(feature_names)}
            try:
                indices = [feature_map[feat] for feat in requested]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown feature '{exc.args[0]}' for {name}. "
                    f"Available: {feature_names}"
                ) from exc
        else:
            indices = list(requested)
        invalid = [idx for idx in indices if idx < 0 or idx >= len(feature_names)]
        if invalid:
            raise ValueError(
                f"Invalid feature indices for {name}: {invalid}. "
                f"Valid range: 0-{len(feature_names) - 1}"
            )
        return indices

    theta = MLS_THETA
    degree = MLS_DEGREE

    np.save(os.path.join(output_dir, "theta.npy"), theta)
    np.save(os.path.join(output_dir, "degree.npy"), degree)

    specs = [
        ("dQx", existing["dq"][:, 0], 0),
        ("dQy", existing["dq"][:, 1], 1),
        ("dP", existing["dp"], 2),
        ("taustx", existing["taust"][:, 0], 3),
        ("tausty", existing["taust"][:, 1], 4),
        ("pmax", existing["pmax"], 5),
        ("pmin", existing["pmin"], 6),
        ("hmax", existing["hmax"], 7),
        ("hmin", existing["hmin"], 8),
    ]

    # For transient: prepare per-feature normaliser states
    # The normaliser_state has all 6 features; we need sub-states for feature subsets
    def get_subset_normaliser_state(feature_idx):
        if normaliser_state is None:
            return None
        return {
            "running_min": normaliser_state["running_min"][feature_idx],
            "running_max": normaliser_state["running_max"][feature_idx],
        }

    task_data = {}
    for name, y, idx in specs:
        feature_idx = resolve_feature_indices(name)
        X_sel = X[:, feature_idx]
        X_rot_sel = X_rot[:, feature_idx]
        sub_norm = get_subset_normaliser_state(feature_idx) if transient else None
        task_data[name] = create_multinom_tasks(
            X_sel, y, X_rot_sel, theta=theta[idx], n=degree[idx],
            normaliser_state=sub_norm,
            verbose=verbose,
        )
        task_data[name] += (feature_idx, )

    for name, (task_list, Mati, Ymin, Yrng, Xi, feature_idx) in task_data.items():
        file_path = os.path.join(output_dir, f"{name}_tasks.npz")
        save_dict = dict(
            tasks=np.array(task_list, dtype=object),
            Mati=Mati,
            Xi=Xi,
            Ymin=Ymin,
            Yrng=Yrng,
            feature_idx=np.array(feature_idx, dtype=int),
            feature_names=np.array(feature_names),
        )
        np.savez(file_path, **save_dict)

    print("update_metamodel.py completed successfully.")


if __name__ == "__main__":
    main()
