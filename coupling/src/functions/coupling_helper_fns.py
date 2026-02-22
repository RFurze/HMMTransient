import logging
import os
import tempfile
import subprocess
import numpy as np


def build_task_list(xi):
    """
    xi is a 9-element list:
      xi[0] = h
      xi[1] = p
      xi[2] = lmbx
      xi[3] = lmby
      xi[4] = lmbz
      xi[5] = gradPx
      xi[6] = gradPy
      xi[7] = gradPz
      xi[8] = gradHx
      xi[9] = gradHy
      xi[10] = gradHz
    We want each micro task to include this new "f" as well.
    """
    logging.info("Building task list from xi...")

    h_base = xi[0]
    p_base = xi[1]
    lmbx_rotated = xi[2]
    lmby_rotated = xi[3]
    lmbz_rotated = xi[4]
    gradPx_rotated = xi[5]
    gradPy_rotated = xi[6]
    gradPz_rotated = xi[7]
    gradHx_rotated = xi[8]
    gradHy_rotated = xi[9]
    gradHz_rotated = xi[10]

    N = len(h_base)  # same as len(p_base), etc.

    tasks = []
    for task_id in range(N):
        # Now we build an 11-tuple so the micro-sim can also see "f_cav"
        # The final item is f_cav[task_id].
        # You can reorder as you prefer.
        task = (
            task_id,
            task_id,
            h_base[task_id],
            p_base[task_id],
            lmbx_rotated[task_id],
            lmby_rotated[task_id],
            lmbz_rotated[task_id],
            gradPx_rotated[task_id],
            gradPy_rotated[task_id],
            gradPz_rotated[task_id],
            gradHx_rotated[task_id],
            gradHy_rotated[task_id],
            gradHz_rotated[task_id],
        )
        tasks.append(task)

    logging.info(f"Built a task list with {len(tasks)} tasks, each with f included.")
    return tasks


def build_task_list_transient(xi):
    """
    xi is a 10-element list:
      xi[0] = h
      xi[1] = p
      xi[2] = lmbx
      xi[3] = lmby
      xi[4] = lmbz
      xi[5] = gradPx
      xi[6] = gradPy
      xi[7] = gradPz
      xi[8] = gradHx
      xi[9] = gradHy
      xi[10] = gradHz
      xi[11] = hdot
      xi[12] = pdot
    """
    logging.info("Building task list from xi...")

    h_base = xi[0]
    p_base = xi[1]
    lmbx_rotated = xi[2]
    lmby_rotated = xi[3]
    lmbz_rotated = xi[4]
    gradPx_rotated = xi[5]
    gradPy_rotated = xi[6]
    gradPz_rotated = xi[7]
    gradHx_rotated = xi[8]
    gradHy_rotated = xi[9]
    gradHz_rotated = xi[10]
    h_dot = xi[11]  # This is the time derivative of h
    p_dot = xi[12]  # This is the time derivative of p

    N = len(h_base)  # same as len(p_base), etc.

    tasks = []
    for task_id in range(N):
        # Now we build an 11-tuple so the micro-sim can also see "f_cav"
        # The final item is f_cav[task_id].
        # You can reorder as you prefer.
        task = (
            task_id,
            task_id,
            h_base[task_id],
            p_base[task_id],
            lmbx_rotated[task_id],
            lmby_rotated[task_id],
            lmbz_rotated[task_id],
            gradPx_rotated[task_id],
            gradPy_rotated[task_id],
            gradPz_rotated[task_id],
            gradHx_rotated[task_id],
            gradHy_rotated[task_id],
            gradHz_rotated[task_id],
            h_dot[task_id],  # Time derivative of h
            p_dot[task_id],  # Time derivative of p
        )
        tasks.append(task)

    logging.info(f"Built a task list with {len(tasks)} tasks, each with f included.")
    return tasks


def run_multinom_mls_simulations(x_file, y_file, xi_file, theta, n, out_file):
    """
    Example function to spawn a separate Python process under mpiexec to run
    the parallel MLS. This mimics your 'run_microscale_simulations' pattern.
    """
    command = [
        "mpiexec",
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        "12",  # or however many processes you want
        "python3",
        "-u",
        "-m",
        "mpi4py.futures",  # module to enable MPIPoolExecutor
        "-m",
        "coupling.src.run_multinom_MLS",
        "--X",
        x_file,
        "--Y",
        y_file,
        "--Xi",
        xi_file,
        "--theta",
        str(theta),
        "--degree",
        str(n),
        "--output",
        out_file,
    ]
    try:
        subprocess.run(command, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except subprocess.CalledProcessError as e:
        logging.error(f"Parallel MLS failed (code {e.returncode}).")
        raise
    # The parallel script is expected to save results to 'out_file', or a set of files
    # e.g. np.save(...) calls. Then you can read them back here if you want.
    results = np.load(out_file)
    return results


def fit_predict_multinom_mls_parallel(X, Y, Xi, theta, degree):
    """
    Convenience function that:
      - Writes (X, Y, Xi) to temp files,
      - Calls run_multinom_mls_simulations in parallel,
      - Returns the predicted array.
    """
    # 1. Write to temporary files
    with tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fx, tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fy, tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fxi:
        np.save(fx.name, X)
        np.save(fy.name, Y)
        np.save(fxi.name, Xi)

        # 2. Another temp file for storing the predictions
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fout:
            out_file = fout.name
            # 3. Launch the parallel MLS script
            preds = run_multinom_mls_simulations(
                x_file=fx.name,
                y_file=fy.name,
                xi_file=fxi.name,
                theta=theta,
                n=degree,
                out_file=out_file,
            )

    # 4. 'preds' is the array that was loaded from out_file
    return preds


def predict_mls_parallel(X_train, Y_train, X_val, theta, degree):
    """
    Write X_train, Y_train, X_val to temporary .npy files,
    invoke 'run_multinom_mls_simulations' for parallel MLS,
    and return the predictions as a 1D numpy array of length X_val.shape[0].
    """

    # 1) Write arrays to disk
    with tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fX, tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fY, tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fXi, tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as fOut:

        np.save(fX.name, X_train)
        np.save(fY.name, Y_train)
        np.save(fXi.name, X_val)

        # 2) Call the parallel MLS script
        out_file = fOut.name
        preds = run_multinom_mls_simulations(
            x_file=fX.name,
            y_file=fY.name,
            xi_file=fXi.name,
            theta=theta,
            n=degree,
            out_file=out_file,
        )

    # 'preds' is now the predictions for all rows in X_val
    return preds
