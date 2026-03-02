"""Evaluate MLS to obtain macroscale corrections.

This step reads the ``*_tasks.npz`` bundles prepared by
``3_update_metamodel`` and solves them in parallel using MPI.  The predicted
corrections ``dQx.npy``, ``dQy.npy``, ``dP.npy`` and the remaining six
correction variables are written to ``--output_dir`` for the subsequent
macroscale solve.

For the three primary correction fields (dQx, dQy, dP) a companion binary
flag array ``<name>_mls_flag.npy`` is also written:

    0 → full MLS solve succeeded (n_eff ≥ Nt polynomial terms)
    1 → any fallback was used (k-NN, Tikhonov regularisation, or
        nearest-neighbour constant)

When ``np.linalg.lstsq`` raises ``LinAlgError`` (the SVD did not converge),
the solve is retried with Tikhonov regularisation.  If that also fails the
nearest-neighbour training value is substituted.  In all non-standard cases
the flag is set to 1.  Diagnostic output is printed to stdout so the exact
failing query points and their local geometry can be inspected.

Key command line options
------------------------
``--k_neighbors`` and ``--chunk_size`` control MLS workload.
``--w_thresh``    weight-fraction threshold (default 1e-3); a neighbour is
                  considered effective if its Gaussian weight ≥ w_thresh × w_max.
``--output_dir``  points to input bundles and is where outputs are saved.
"""

import os
import time
from concurrent.futures import as_completed

import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from scipy.spatial import cKDTree

from CONFIGPenalty import MLS_THETA, MLS_DEGREE
from utils.cli import parse_common_args

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
root = rank == 0

# Variables for which a binary fallback-flag array is saved alongside the
# prediction.  Extend this set if you need flags for other outputs.
TRACK_FLAG = {"dQx", "dQy", "dP"}

# ---------------------------------------------------------------------------
# Global training data cached on each worker rank
# ---------------------------------------------------------------------------

G_MAT:   np.ndarray | None = None   # (N_train, N_poly)
G_Y:     np.ndarray | None = None   # (N_train,)
G_THETA: float       | None = None


def init_worker():
    """Executed once on each worker when the pool starts."""
    global G_MAT, G_Y, G_THETA
    G_MAT   = None
    G_Y     = None
    G_THETA = None


def update_globals(Y: np.ndarray, Mat: np.ndarray, theta: float):
    """Replace the global training data on a worker rank."""
    global G_MAT, G_Y, G_THETA
    G_MAT   = Mat
    G_Y     = Y
    G_THETA = float(theta)
    return MPI.COMM_WORLD.Get_rank()


# ---------------------------------------------------------------------------
# Per-query solver – executed inside worker ranks
# ---------------------------------------------------------------------------

def _solve_one(
    i_q:     int,
    idx:     np.ndarray,
    dist:    np.ndarray,
    w_thresh: float = 1e-3,
):
    """Weighted least-squares MLS for a single query point.

    Returns
    -------
    alpha     : np.ndarray, shape (Nt,)
    fallback  : bool – True whenever a non-standard path was taken
    fail_info : None, or dict with diagnostic data if lstsq raised LinAlgError
    """
    Nt    = G_MAT.shape[1]
    wght  = np.exp(-G_THETA * dist ** 2)
    w_max = wght.max()

    # --- degenerate: all neighbours have negligible weight -------------------
    if w_max < 1e-15:
        # Nearest-neighbour constant: alpha[0] = Y[nn] exploits that
        # column 0 of the polynomial basis is always 1.
        alpha_nn    = np.zeros(Nt)
        alpha_nn[0] = G_Y[idx[0]]
        return alpha_nn, True, None

    # --- standard mask: neighbours above the weight threshold ----------------
    mask  = wght >= w_thresh * w_max
    n_eff = int(np.count_nonzero(mask))
    is_knn = n_eff < Nt          # True → too few effective neighbours

    if is_knn:
        mask = np.ones(len(idx), dtype=bool)   # use all k neighbours

    Mat_red = G_MAT[idx][mask]
    Y_red   = G_Y[idx][mask]
    w_red   = wght[mask]
    Matw    = Mat_red * w_red[:, None]
    Pw      = Y_red   * w_red

    # --- primary lstsq -------------------------------------------------------
    try:
        alpha, *_ = np.linalg.lstsq(Matw, Pw, rcond=None)
        return alpha, is_knn, None

    except np.linalg.LinAlgError as _lstsq_err:
        # Build diagnostic record (Xi coords added by root after collection)
        fail_info = dict(
            i_q      = i_q,
            n_eff    = n_eff,
            is_knn   = is_knn,
            nn_dist  = float(dist[0]),
            lstsq_err= str(_lstsq_err),
            tikh_ok  = False,
            tikh_err = None,
            lam      = None,
        )

        # --- Tikhonov regularisation fallback --------------------------------
        try:
            lam   = 1e-10 * float((Matw * Matw).sum()) / Nt
            A     = Matw.T @ Matw + lam * np.eye(Nt)
            alpha = np.linalg.solve(A, Matw.T @ Pw)
            fail_info["tikh_ok"] = True
            fail_info["lam"]     = lam
            return alpha, True, fail_info

        except (np.linalg.LinAlgError, Exception) as _tikh_err:
            fail_info["tikh_err"] = str(_tikh_err)

        # --- nearest-neighbour fallback (last resort) ------------------------
        alpha_nn    = np.zeros(Nt)
        alpha_nn[0] = G_Y[idx[0]]
        return alpha_nn, True, fail_info


# ---------------------------------------------------------------------------
# Batch worker – dispatched to the MPI pool
# ---------------------------------------------------------------------------

def batch_worker(
    i_q_batch:  np.ndarray,
    idx_batch:  np.ndarray,
    dist_batch: np.ndarray,
    w_thresh:   float = 1e-3,
):
    """Solve MLS for a batch of query indices on one worker rank.

    Returns
    -------
    out  : list of (i_q, alpha, fallback, fail_info)
    rank : int – worker rank (for load-balance logging)
    """
    out = []
    for local_idx, i_q in enumerate(i_q_batch):
        alpha, fallback, fail_info = _solve_one(
            int(i_q),
            idx_batch[local_idx],
            dist_batch[local_idx],
            w_thresh,
        )
        out.append((i_q, alpha, fallback, fail_info))
    return out, MPI.COMM_WORLD.Get_rank()


# ---------------------------------------------------------------------------
# Root-side helpers
# ---------------------------------------------------------------------------

def ensure_all_workers_update(
    pool: MPIPoolExecutor,
    Y:    np.ndarray,
    Mat:  np.ndarray,
    theta: float,
):
    """Guarantee that every worker rank runs ``update_globals``."""
    futures = [pool.submit(update_globals, Y, Mat, theta) for _ in range(size - 1)]
    for fut in as_completed(futures):
        fut.result()


def process_prediction(
    pool:            MPIPoolExecutor,
    tasks:           np.ndarray,
    Mati:            np.ndarray,
    Xi:              np.ndarray,
    Ymin:            float,
    Yrng:            float,
    output_filename: str,
    output_dir:      str,
    theta:           float,
    k_neighbors:     int,
    chunk_size:      int   = 64,
    w_thresh:        float = 1e-3,
    track_flag:      bool  = False,
):
    """Dispatch MLS solves to the worker pool and write the final prediction.

    Executed **only on the root rank**.  Workers sit in the MPI pool server
    loop.

    When ``track_flag=True`` a companion ``<name>_mls_flag.npy`` file is
    written (0 = full MLS, 1 = any fallback).  LinAlgError events are
    printed immediately with per-point diagnostics and summarised in a table
    at the end of the variable.
    """
    name    = output_filename.replace(".npy", "")
    Ni      = Xi.shape[0]
    Nt      = Mati.shape[1]

    placeholder = np.empty((Nt, Ni), dtype=float)
    flag_arr    = np.zeros(Ni, dtype=np.int8) if track_flag else None
    fail_infos  = []      # collected from workers, Xi coords added here on root
    rank_counts = {}
    completed   = 0

    # --- extract training arrays (identical across all tasks) ----------------
    _, X_train, Y_train, Mat_train, _ = tasks[0]

    # --- distribute training data to every worker ----------------------------
    ensure_all_workers_update(pool, Y_train, Mat_train, theta)

    # --- KD-tree neighbour search (on root, workers=-1 uses all local CPUs) --
    tree               = cKDTree(X_train)
    dist_all, idx_all  = tree.query(Xi, k=k_neighbors, workers=-1)

    # --- submit batches to the pool ------------------------------------------
    futures = []
    for start in range(0, Ni, chunk_size):
        sli       = slice(start, min(start + chunk_size, Ni))
        i_q_batch = np.arange(start, min(start + chunk_size, Ni))
        futures.append(
            pool.submit(batch_worker, i_q_batch, idx_all[sli], dist_all[sli], w_thresh)
        )

    # --- collect results as they arrive --------------------------------------
    for fut in as_completed(futures):
        results, wrk = fut.result()
        rank_counts[wrk] = rank_counts.get(wrk, 0) + len(results)
        for i_q, alpha, fallback, fail_info in results:
            placeholder[:, i_q] = alpha
            if track_flag and fallback:
                flag_arr[i_q] = 1
            if fail_info is not None:
                # Root attaches Xi coordinates for the diagnostic summary
                fail_info["xi_coords"] = Xi[i_q].tolist()
                # Print immediately so failures appear as they are detected
                tikh_str = (
                    f"Tikhonov OK (lam={fail_info['lam']:.3e})"
                    if fail_info["tikh_ok"]
                    else f"Tikhonov FAIL ('{fail_info['tikh_err']}') → nearest-neighbour"
                )
                print(
                    f"  [LinAlgError] {name} i_q={i_q}  "
                    f"n_eff={fail_info['n_eff']}  is_knn={fail_info['is_knn']}  "
                    f"nn_dist={fail_info['nn_dist']:.4e}  "
                    f"Xi={np.array2string(Xi[i_q], precision=4, suppress_small=True)}  "
                    f"err='{fail_info['lstsq_err']}'  → {tikh_str}",
                    flush=True,
                )
                fail_infos.append(fail_info)
        completed += len(results)

    # --- reconstruct physical prediction -------------------------------------
    Yi = (Mati * placeholder.T).sum(axis=1) * Yrng + Ymin
    np.save(os.path.join(output_dir, output_filename), Yi)

    # --- save binary flag array ----------------------------------------------
    if track_flag and flag_arr is not None:
        flag_path = os.path.join(output_dir, f"{name}_mls_flag.npy")
        np.save(flag_path, flag_arr)
        n_flagged = int(flag_arr.sum())
        print(
            f"[root] {name}: flag array saved — "
            f"{n_flagged}/{Ni} nodes flagged (0=MLS, 1=fallback)",
            flush=True,
        )

    # --- LinAlgError summary table -------------------------------------------
    n_linalg = len(fail_infos)
    if n_linalg:
        n_tikh = sum(1 for e in fail_infos if e["tikh_ok"])
        n_nn   = n_linalg - n_tikh
        print(
            f"\n[root] {name}: LinAlgError summary — "
            f"{n_linalg} failures  (Tikhonov={n_tikh} OK, NN={n_nn})",
            flush=True,
        )
        hdr = (
            f"  {'i_q':>7}  {'n_eff':>5}  {'knn':>3}  "
            f"{'nn_dist':>9}  {'tikh':>4}  Xi (normalised)"
        )
        print(hdr, flush=True)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        for e in sorted(fail_infos, key=lambda x: x["i_q"]):
            xi_str = "  ".join(f"{v:+.3f}" for v in e["xi_coords"])
            tikh   = "OK" if e["tikh_ok"] else "FAIL"
            print(
                f"  {e['i_q']:>7}  {e['n_eff']:>5}  "
                f"{'Y' if e['is_knn'] else 'N':>3}  "
                f"{e['nn_dist']:>9.3e}  {tikh:>4}  [{xi_str}]",
                flush=True,
            )
        print(flush=True)

    print(
        f"[root] {name}: done — LinAlgError={n_linalg}  "
        f"output range [{Yi.min():.4e}, {Yi.max():.4e}]",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main (root-only)
# ---------------------------------------------------------------------------

def main():
    args       = parse_common_args("Run MLS", MLS=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()

    # --- load all nine task bundles ------------------------------------------
    def load(name):
        path = os.path.join(args.output_dir, f"{name}_tasks.npz")
        data = np.load(path, allow_pickle=True)
        return (
            data["tasks"],
            data["Mati"],
            data["Xi"],
            data["Ymin"].item(),
            data["Yrng"].item(),
            data["feature_idx"].tolist()   if "feature_idx"   in data.files else None,
            data["feature_names"].tolist() if "feature_names" in data.files else None,
        )

    dQx    = load("dQx")
    dQy    = load("dQy")
    dP     = load("dP")
    taustx = load("taustx")
    tausty = load("tausty")
    pmax   = load("pmax")
    pmin   = load("pmin")
    hmax   = load("hmax")
    hmin   = load("hmin")

    if root:
        with MPIPoolExecutor(initializer=init_worker) as pool:
            for var_idx, (name, pack) in enumerate(zip(
                ("dQx", "dQy", "dP", "taustx", "tausty", "pmax", "pmin", "hmax", "hmin"),
                (dQx, dQy, dP, taustx, tausty, pmax, pmin, hmax, hmin),
            )):
                tasks, Mati, Xi, Ymin, Yrng, feature_idx, feature_names = pack
                t_var = time.time()
                process_prediction(
                    pool,
                    tasks,
                    Mati,
                    Xi,
                    Ymin,
                    Yrng,
                    f"{name}.npy",
                    args.output_dir,
                    MLS_THETA[var_idx],
                    k_neighbors = args.k_neighbors,
                    chunk_size  = args.chunk_size,
                    w_thresh    = args.w_thresh,
                    track_flag  = (name in TRACK_FLAG),
                )
                print(f"[root] {name} finished in {time.time() - t_var:.2f}s", flush=True)

        print(f"[root] MLS evaluations finished in {time.time() - t0:.2f}s", flush=True)

    else:
        # Worker ranks idle here; the MPIPool server loop runs inside
        # mpi4py.futures runtime.
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
