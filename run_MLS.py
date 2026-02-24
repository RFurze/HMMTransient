"""Evaluate MLS to obtain macroscale corrections.

This step reads the ``*_tasks.npz`` bundles prepared by
``3_update_metamodel`` and solves them in parallel using MPI. The predicted
corrections ``dQx.npy``, ``dQy.npy``, ``dP.npy`` and ``Fst.npy`` are written
to ``--output_dir`` for the subsequent macroscale solve.

Key command line options:
    ``--k_neighbors`` and ``--chunk_size`` control MLS workload.
    ``--output_dir`` points to input bundles and is where outputs are saved.
"""

import os
from utils.cli import parse_common_args
import time
from concurrent.futures import as_completed
from CONFIGPenalty import MLS_THETA, MLS_DEGREE

import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from scipy.spatial.distance import cdist

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
root = rank == 0
if root:
    print(
        f"[root] Master rank {rank} initialised with {size - 1} worker(s).", flush=True
    )


# Global training data cached on each worker
G_MAT: np.ndarray | None = None  # (Ntrain, Npoly)
G_Y: np.ndarray | None = None  # (Ntrain,)
G_THETA: float | None = None
TRANSIENT_MODE = False

# --------------------------------------------------------------------------------------
# Worker initialiser - run only once on pool creation
# --------------------------------------------------------------------------------------

def init_worker():
    """Executed once on each worker when the pool starts to avoid memory overflow later"""
    global G_MAT, G_Y, G_THETA
    G_MAT = None
    G_Y = None
    G_THETA = None
    print(
        f"[worker-init] Rank {MPI.COMM_WORLD.Get_rank()} initialised globals",
        flush=True,
    )

# --------------------------------------------------------------------------------------
# Helpers executed by workers
# --------------------------------------------------------------------------------------

def update_globals(Y: np.ndarray, Mat: np.ndarray, theta: float):
    """Replace the global training data on a worker rank."""
    global G_MAT, G_Y, G_THETA
    G_MAT = Mat  # view, not copy (already local on the worker)
    G_Y = Y
    G_THETA = float(theta)
    return MPI.COMM_WORLD.Get_rank()


def _solve_one(idx: np.ndarray, dist: np.ndarray, w_thresh: float = 1e-1):
    """Weighted least squares for one query; executed inside the batch loop."""
    wght = np.exp(-G_THETA * dist**2)
    w_max = wght.max()

    if w_max < 1e-15:
        return np.zeros(G_MAT.shape[1])

    mask = wght >= w_thresh * w_max
    if np.count_nonzero(mask) < G_MAT.shape[1]:
        mask = np.ones_like(mask, dtype=bool)

    Mat_red = G_MAT[idx][mask]
    Y_red = G_Y[idx][mask]
    w_red = wght[mask]

    Matw = Mat_red * w_red[:, None]
    Pw = Y_red * w_red
    alpha, *_ = np.linalg.lstsq(Matw, Pw, rcond=None)
    return alpha


def batch_worker(
    i_q_batch: np.ndarray,
    idx_batch: np.ndarray,
    dist_batch: np.ndarray,
    w_thresh: float = 1e-3,
):
    """Executes on a worker: solves MLS for a *batch* of query indices."""
    out = []
    for local_idx, i_q in enumerate(i_q_batch):
        alpha = _solve_one(idx_batch[local_idx], dist_batch[local_idx], w_thresh)
        out.append((i_q, alpha))
    return out, MPI.COMM_WORLD.Get_rank()


# --------------------------------------------------------------------------------------
# Controller-side helpers
# --------------------------------------------------------------------------------------

def ensure_all_workers_update(
    pool: MPIPoolExecutor, Y: np.ndarray, Mat: np.ndarray, theta: float
):
    """Guarantee that *every* worker rank runs `update_globals`."""
    futures = [pool.submit(update_globals, Y, Mat, theta) for _ in range(size - 1)]
    for fut in as_completed(futures):
        fut.result()  # re-raise errors immediately
    if root:
        print("[root] All workers updated their training data", flush=True)

# --------------------------------------------------------------------------------------
# Other helper functions 
# --------------------------------------------------------------------------------------

def process_prediction(
    pool: MPIPoolExecutor,
    tasks: np.ndarray,
    Mati: np.ndarray,
    Xi: np.ndarray,
    Ymin: float,
    Yrng: float,
    output_filename: str,
    output_dir: str,
    theta: float,
    k_neighbors: int,
    chunk_size: int = 64,
):
    """Dispatches MLS solves to the worker pool and writes final prediction.
    *Executed **only on the root rank***. Workers sit in the pool.
    """

    Ni, Nt = len(tasks), Mati.shape[1]
    print(
        f"[root] process_prediction: {output_filename} with {Ni} query points",
        flush=True,
    )

    placeholder = np.empty((Nt, Ni), dtype=float)
    completed, rank_counts = 0, {}

    # --- extract training arrays once (they are identical for every task) ----------
    _, X_train, Y_train, Mat_train, _ = tasks[0]

    # ---------- distribute training data to workers --------------------------------
    ensure_all_workers_update(pool, Y_train, Mat_train, theta)

    # ---------- Mahalanobis neighbour search ---------------------------------------
    centered = X_train - X_train.mean(axis=0)
    if X_train.shape[0] < 2:
        cov = np.eye(X_train.shape[1])
    else:
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
    cov += 1e-10 * np.eye(cov.shape[0])
    cov_inv = np.linalg.pinv(cov)

    dmat = cdist(Xi, X_train, metric="mahalanobis", VI=cov_inv)
    k_use = min(k_neighbors, X_train.shape[0])
    if k_use < 1:
        raise ValueError("No training points available for MLS prediction.")
    idx_all = np.argpartition(dmat, kth=k_use - 1, axis=1)[:, :k_use]
    dist_all = np.take_along_axis(dmat, idx_all, axis=1)
    ord_local = np.argsort(dist_all, axis=1)
    idx_all = np.take_along_axis(idx_all, ord_local, axis=1)
    dist_all = np.take_along_axis(dist_all, ord_local, axis=1)
    print("[root] Mahalanobis neighbour search done", flush=True)

    # -----------------------------------------------------------------------------
    # Fire off batches
    # -----------------------------------------------------------------------------
    submit_t0 = time.time()
    futures = []
    for start in range(0, Ni, chunk_size):
        sli = slice(start, min(start + chunk_size, Ni))
        i_q_batch = np.arange(start, min(start + chunk_size, Ni))
        futures.append(
            pool.submit(batch_worker, i_q_batch, idx_all[sli], dist_all[sli])
        )
    print(
        f"[root] Submitted {len(futures)} batch tasks in {time.time() - submit_t0:.2f}s",
        flush=True,
    )

    # collect results
    for fut in as_completed(futures):
        results, wrk = fut.result()
        rank_counts[wrk] = rank_counts.get(wrk, 0) + len(results)
        for i_q, alpha in results:
            placeholder[:, i_q] = alpha
        completed += len(results)
        if completed % 500 == 0:
            print(
                f"[root] Completed {completed}/{Ni}  rank_counts={rank_counts}",
                flush=True,
            )

    # ----------------------------- gather & write output -------------------------
    Mi = Mati * placeholder.T  # (Ni, Nt)
    Yi = Mi.sum(axis=1) * Yrng + Ymin
    print(f"Minimum value of Yi: {np.min(Yi)}", flush=True)
    np.save(os.path.join(output_dir, output_filename), Yi)
    print(f"[root] Saved {output_filename} - shape {Yi.shape}", flush=True)


def main():
    args = parse_common_args("Run MLS", MLS=True)
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    global TRANSIENT_MODE
    TRANSIENT_MODE = args.transient
    print(f'k_neighbors: {args.k_neighbors}')
    t0 = time.time()

    # --------------------------------------------------------------------------
    # Load *.npz bundles per variable - identical structure for each.
    # --------------------------------------------------------------------------
    def load(name):
        path = os.path.join(args.output_dir, f"{name}_tasks.npz")
        data = np.load(path, allow_pickle=True)
        feature_idx = None
        feature_names = None
        if "feature_idx" in data.files:
            feature_idx = data["feature_idx"].tolist()
        if "feature_names" in data.files:
            feature_names = data["feature_names"].tolist()
        return (
            data["tasks"],
            data["Mati"],
            data["Xi"],
            data["Ymin"].item(),
            data["Yrng"].item(),
            feature_idx,
            feature_names,
        )

    dQx = load("dQx")
    dQy = load("dQy")
    dP = load("dP")
    taustx = load("taustx")
    tausty = load("tausty")
    pmax = load("pmax")
    pmin = load("pmin")
    hmax = load("hmax")
    hmin = load("hmin")

    if root:
        Ni = len(dQx[0])
        Nt = dQx[1].shape[1]
        print(f"[root] Loaded {Ni} query points, {Nt} polynomial terms.", flush=True)

    # --------------------------------------------------------------------------
    # Root rank drives the pool; workers do *not* enter this block.
    # --------------------------------------------------------------------------
    if root:
        with MPIPoolExecutor(initializer=init_worker) as pool:
            for idx, (name, pack) in enumerate(
                zip(
                    (
                        "dQx",
                        "dQy",
                        "dP",
                        "taustx",
                        "tausty",
                        "pmax",
                        "pmin",
                        "hmax",
                        "hmin",
                    ),
                    (dQx, dQy, dP, taustx, tausty, pmax, pmin, hmax, hmin),
                )
            ):
                tasks, Mati, Xi, Ymin, Yrng, feature_idx, feature_names = pack
                if feature_idx is not None and feature_names is not None:
                    selected = [feature_names[i] for i in feature_idx]
                    print(
                        f"[root] ==== {name} features: {selected}",
                        flush=True,
                    )
                print(
                    f"[root] ==== {name} === (theta={MLS_THETA[idx]}, degree={MLS_DEGREE[idx]})",
                    flush=True,
                )
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
                    MLS_THETA[idx],
                    k_neighbors=args.k_neighbors,
                    chunk_size=args.chunk_size,
                )
                print(f"[root] {name} done in {time.time() - t_var:.2f}s", flush=True)

        print(f"[root] MLS evaluations finished in {time.time() - t0:.2f}s", flush=True)
    else:
        # Worker ranks idle here; the MPIPool server loop is already running inside
        # mpi4py.futures runtime.
        while True:
            time.sleep(3600)  # effectively wait forever; Ctrl+C / MPI abort ends run


if __name__ == "__main__":
    main()
