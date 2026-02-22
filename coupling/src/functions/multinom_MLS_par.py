# run_multinom_mls.py
import numpy as np
import itertools
import tempfile
import subprocess
import argparse
import os
import logging
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from coupling.src.functions.coupling_helper_fns import predict_mls_parallel


# (Re-use your polynomial exponent generator)
def multinom_coeffs(n, m, verbose=False):
    all_exps = []
    total = (n + 1) ** m
    step = max(1, total // 10)
    count = 0
    for exps in itertools.product(range(n + 1), repeat=m):
        count += 1
        if verbose and count % step == 0:
            print(f"Processed {count}/{total} combos ({100*count/total:.2f}%)")
        if sum(exps) <= n:
            all_exps.append(exps)
    C = np.array(all_exps, dtype=int)
    return C, C.shape[0]


# The single-query worker code (import from a separate file if desired)
def worker_multinom_query_point(args, w_thresh=1e-1):
    """Solve the weighted least-squares system for one query index using
    precomputed weights."""
    i_q, wght, Y_, Mat = args

    # if i_q == 10 or i_q == 200 or i_q == 7000:
    #     print(f'Shape of X_: {X_.shape}')
    #     print(f'shape of Xi_q: {Xi_q.shape}')
    #     print(f'Shape of Y_: {Y_.shape}')
    #     print(f'Shape of Mat: {Mat.shape}')
    #     print(f'Shape of wght: {wght.shape}')
    #     print(f'Maximum weight for query {i_q}: {wght.max()}')
    #     print(f'Minimum weight for query {i_q}: {wght.min()}')
    #     print(f'Number of points with weight above threshold for query {i_q}: {np.count_nonzero(wght >= w_thresh * wght.max())}, length of weights: {len(wght)}')
    #     print(f'Mean weight for query {i_q}: {np.mean(wght)}')

    # Find the maximum weight
    w_max = wght.max()

    # If the maximum weight is extremely small, you may want a fallback path.
    # For instance, return zeros, or skip.
    # This is an application-specific choice.
    if w_max < 1e-15:
        # e.g. fallback to all zeros or a direct guess
        alpha = np.zeros(Mat.shape[1])
        from mpi4py import MPI

        worker_rank = MPI.COMM_WORLD.Get_rank()
        return (i_q, alpha, worker_rank)

    # Mask out training points whose weight is below threshold fraction of w_max
    mask = wght >= w_thresh * w_max

    # If the mask is too small (e.g., fewer points than needed to solve),
    # you can also decide to skip or reduce the threshold as a fallback.
    # For demonstration, we'll just check we have enough points:
    num_points = np.count_nonzero(mask)
    if num_points < Mat.shape[1]:
        # If not enough points to form a solvable system, fallback
        mask = np.ones_like(mask, dtype=bool)  # revert to using all points

    # Apply the mask to reduce the system
    Mat_reduced = Mat[mask, :]
    w_reduced = wght[mask]
    Y_reduced = Y_[mask]

    Matw = Mat_reduced * w_reduced[:, None]
    P = Y_reduced * w_reduced

    # Solve the smaller least-squares problem
    alpha, _, _, _ = np.linalg.lstsq(Matw, P, rcond=None)

    # Get MPI rank to track distribution (optional)
    from mpi4py import MPI

    worker_rank = MPI.COMM_WORLD.Get_rank()

    # Return the result
    return (i_q, alpha, worker_rank)


from mpi4py.futures import MPIPoolExecutor, as_completed
from mpi4py import MPI


def multinom_MLS_parallel(X, Y, Xi, theta, n, verbose=False):
    """
    Parallel version of multinom_MLS that distributes the per-query local
    MLS solve (loop over i_q in Xi) to multiple ranks via MPIPoolExecutor.
    """
    # Basic sizes
    N, m = X.shape
    Ni = Xi.shape[0]
    # print(f'Number of training points: {N}, Number of query points: {Ni}')

    # Normalize each column of X and Xi
    X_ = np.zeros_like(X, dtype=float)
    Xi_ = np.zeros_like(Xi, dtype=float)
    for j in range(m):
        xcol = X[:, j]
        xmin, xmax = xcol.min(), xcol.max()
        rng = (xmax - xmin) if (xmax > xmin) else 1.0
        X_[:, j] = (xcol - xmin) / rng
        Xi_[:, j] = (Xi[:, j] - xmin) / rng

    # Normalize Y
    Ymin, Ymax = Y.min(), Y.max()
    Yrng = (Ymax - Ymin) if (Ymax > Ymin) else 1.0
    Y_ = (Y - Ymin) / Yrng

    # Build polynomial exponents and then design matrices
    C, Nt = multinom_coeffs(n, m, verbose=verbose)
    Mat = np.ones((N, Nt), dtype=float)
    Mati = np.ones((Ni, Nt), dtype=float)
    for i_exp in range(Nt):
        for j_col in range(m):
            exp_j = C[i_exp, j_col]
            if exp_j != 0:
                Mat[:, i_exp] *= X_[:, j_col] ** exp_j
                Mati[:, i_exp] *= Xi_[:, j_col] ** exp_j

    if verbose and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Parallel MLS query loop: distributing {Ni} queries across ranks...")

    # Prepare parallel tasks: each is (i_q, Xi_[i_q,:], X_, Y_, Mat, theta).
    # Precompute weight matrix for all query points
    diff = Xi_[:, None, :] - X_[None, :, :]
    Dis2_all = np.sum(diff * diff, axis=2)
    W_all = np.exp(-theta * Dis2_all)

    # Prepare parallel tasks with precomputed weights
    tasks = [(i_q, W_all[i_q], Y_, Mat) for i_q in range(Ni)]

    # We'll store polynomial coeffs for each query i_q in M[:, i_q].
    M = np.zeros((Nt, Ni), dtype=float)

    # Initialize counters for progress updates.
    completed_count = 0
    rank_counts = {}

    # Use MPIPoolExecutor to parallelize the query loop.
    with MPIPoolExecutor() as executor:
        futures = [executor.submit(worker_multinom_query_point, task) for task in tasks]
        for future in as_completed(futures):
            i_q, alpha, worker_rank = future.result()
            M[:, i_q] = alpha
            completed_count += 1
            # Update count per worker rank
            rank_counts[worker_rank] = rank_counts.get(worker_rank, 0) + 1
            if completed_count % 200 == 0:
                print(
                    f"Status update: {completed_count} tasks completed. "
                    f"Rank breakdown: {rank_counts}"
                )

    # Evaluate polynomials at Xi_ with the local coefficients M.
    Mi = Mati * M.T  # elementwise product, shape (Ni, Nt)
    Yi_ = np.sum(Mi, axis=1)

    # Unscale the predictions.
    Yi = Yi_ * Yrng + Ymin

    return Yi


def optimize_theta_and_degree_cv(
    X, Y, theta_values, degree_values, cv_method="loo", k=5, verbose=False
):
    """
    Optimize both theta and the polynomial degree for multinom_MLS using cross validation,
    but now using the parallel MLS in each CV iteration.
    """
    from sklearn.metrics import r2_score  # or you can import at top-level

    N, m = X.shape
    best_error = 0
    best_theta = None
    best_degree = None
    error_table = []

    # Loop over candidate polynomial degrees and theta values.
    for degree in degree_values:
        for theta in theta_values:
            if cv_method == "loo":
                # Leave-One-Out Cross Validation
                sq_errors = []
                for i in range(N):
                    # Build train/val sets
                    X_train = np.delete(X, i, axis=0)
                    Y_train = np.delete(Y, i, axis=0)
                    X_val = X[i : i + 1, :]  # shape (1, m)
                    Y_val = Y[i]

                    # --- PARALLEL MLS PREDICTION ---
                    Y_pred = predict_mls_parallel(
                        X_train, Y_train, X_val, theta, degree
                    )

                    sq_errors.append((Y_pred[0] - Y_val) ** 2)
                mean_error = np.mean(sq_errors)

            elif cv_method == "kfold":
                indices = np.arange(N)
                np.random.shuffle(indices)
                fold_sizes = np.full(k, N // k, dtype=int)
                fold_sizes[: N % k] += 1
                folds = []
                current = 0
                for fold_size in fold_sizes:
                    folds.append(indices[current : current + fold_size])
                    current += fold_size

                fold_r2s = []
                for i in range(k):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                    X_train = X[train_idx, :]
                    Y_train = Y[train_idx]
                    X_val = X[val_idx, :]
                    Y_val = Y[val_idx]

                    # --- PARALLEL MLS PREDICTION ---
                    Y_pred = predict_mls_parallel(
                        X_train, Y_train, X_val, theta, degree
                    )

                    r2_fold = r2_score(Y_val, Y_pred)
                    fold_r2s.append(r2_fold)
                # For MSE, you could do np.mean((Y_val - Y_pred)**2)
                mean_error = np.mean(fold_r2s)

            else:
                raise ValueError("cv_method must be either 'loo' or 'kfold'")

            error_table.append((theta, degree, mean_error))
            if verbose:
                print(
                    f"theta = {theta}, degree = {degree}, CV error = {mean_error:.4f}"
                )

            # Adjust your "best_error" logic if needed.
            # Right now we do "if (mean_error - 1) < best_error:" but that
            # is a little odd. Possibly you want "if mean_error > best_error" for R^2 or
            # "if mean_error < best_error" for MSE.
            # We'll keep your existing pattern, but be sure it’s correct.
            if abs(float(mean_error)) > best_error:
                best_error = mean_error
                best_theta = theta
                best_degree = degree

    return best_theta, best_degree, error_table

    # """
    # Example function to spawn a separate Python process under mpiexec to run
    # the parallel MLS. This mimics your 'run_microscale_simulations' pattern.
    # """
    # command = [
    #     "mpiexec",
    #     "--allow-run-as-root",
    #     "--oversubscribe",
    #     "-np", "12",            # or however many processes you want
    #     "python3", "-u", "-m",
    #     "mpi4py.futures",       # module to enable MPIPoolExecutor
    #     "-m", "coupling.src.run_multinom_MLS",
    #     "--X", x_file,
    #     "--Y", y_file,
    #     "--Xi", xi_file,
    #     "--theta", str(theta),
    #     "--degree", str(n),
    #     "--output", out_file
    # ]
    # try:
    #     subprocess.run(command, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    # except subprocess.CalledProcessError as e:
    #     logging.error(f"Parallel MLS failed (code {e.returncode}).")
    #     raise
    # # The parallel script is expected to save results to 'out_file', or a set of files
    # # e.g. np.save(...) calls. Then you can read them back here if you want.
    # results = np.load(out_file)
    # return results
