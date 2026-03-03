"""Create the microscale task list used by the micro sims.

This utility builds the array of microscale simulation inputs (``tasks.npy``)
from the rotated state ``xi_rot.npy``.  When run with ``--transient`` the time
options ``--Time`` and ``--DT`` are parsed as well.

**Transient path** — uses the Error-Driven Adaptive Sampler (EDAS):
  * On the first invocation (``init`` mode) a geometric coverage sample is
    selected so that MLS has enough seed data.
  * On subsequent invocations error indicators from the previous MLS
    evaluation (``mls_error_indicators.npy``) drive sample selection.
  * The EDAS state (normaliser, training data, timestamps) is persisted to
    ``edas_state.npy`` between calls.

**Steady path** — unchanged 1:1 mapping (no downsampling).

Key command line options:
    ``--lb_iter`` and ``--c_iter`` - current iteration counters.
    ``--output_dir`` - where ``tasks.npy`` and ``xi_d.npy`` are written.
"""

import os
import numpy as np
from utils.cli import parse_common_args


def _load_pickle(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def main():
    """Build the list of microscale tasks, optionally in transient mode."""
    args = parse_common_args("Build Microscale Task List")

    lb_iter = args.lb_iter
    c_iter = args.c_iter
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    xi_rot = np.load(os.path.join(output_dir, "xi_rot.npy"))

    if args.transient:
        # ----- EDAS transient path ----------------------------------------
        from coupling.src.functions.edas import ErrorDrivenSampler
        from CONFIGPenalty import edas as edas_cfg, MLS_THETA, MLS_DEGREE

        T = args.Time
        state_file = os.path.join(output_dir, "edas_state.npy")
        error_file = os.path.join(output_dir, "mls_error_indicators.npy")
        existing_file = "transient_existing_xi_d.npy"

        # Load or create the EDAS sampler
        saved_state = _load_pickle(state_file)
        if saved_state is not None:
            sampler = ErrorDrivenSampler.from_state(saved_state)
        else:
            sampler = ErrorDrivenSampler(
                batch_size=edas_cfg.batch_size,
                max_budget=edas_cfg.max_budget,
                error_target=edas_cfg.error_target,
                alpha_blend=edas_cfg.alpha_blend,
                delta_min_quantile=edas_cfg.delta_min_quantile,
                lambda_decay=edas_cfg.lambda_decay,
                sigma_spatial=edas_cfg.sigma_spatial,
                relevance_prune_threshold=edas_cfg.relevance_prune_threshold,
                r0_quantile=edas_cfg.r0_quantile,
                coupling_decay=getattr(edas_cfg, "coupling_decay", 0.5),
            )

        init_cond = sampler.existing_xi_d is None or sampler.existing_xi_d.shape[1] == 0

        # Load error indicators from previous MLS evaluation if available
        mls_errors = None
        if not init_cond and os.path.exists(error_file):
            mls_errors = np.load(error_file)

        # Use average theta for LOOCV error estimation
        avg_theta = float(np.mean(MLS_THETA[:3]))
        avg_degree = int(MLS_DEGREE[0])

        tasks, xi_d, selected_indices = sampler.build(
            xi_rot,
            current_time=T,
            init=init_cond,
            mls_errors=mls_errors,
            theta=avg_theta,
            degree=avg_degree,
            coupling_iter=c_iter,
        )

        # Prune stale training data (with coupling-iteration awareness)
        n_pruned = sampler.prune_training_data(
            T, current_coupling_iter=c_iter,
        )
        if n_pruned > 0:
            print(f"EDAS: pruned {n_pruned} stale training points")

        # Persist sampler state
        np.save(state_file, sampler.get_state(), allow_pickle=True)

        # Also save existing_xi_d for compatibility with generate_MLS_tasks
        if sampler.existing_xi_d is not None:
            np.save(
                os.path.join(output_dir, existing_file),
                sampler.existing_xi_d,
            )

        # Save normaliser state for use by MLS
        np.save(
            os.path.join(output_dir, "edas_normaliser_state.npy"),
            sampler.get_normaliser_state(),
            allow_pickle=True,
        )

        # Save timestamps for relevance weighting in generate_MLS_tasks
        if sampler.timestamps is not None:
            np.save(
                os.path.join(output_dir, "edas_timestamps.npy"),
                sampler.timestamps,
            )

        # Save coupling iteration tags for relevance weighting
        if sampler.coupling_iters is not None:
            np.save(
                os.path.join(output_dir, "edas_coupling_iters.npy"),
                sampler.coupling_iters,
            )

        # Save selected indices for diagnostics
        np.save(
            os.path.join(output_dir, "edas_selected_indices.npy"),
            selected_indices,
        )

        print(
            f"EDAS: selected {len(tasks)} tasks "
            f"(init={init_cond}, training_set={sampler.existing_xi_d.shape[1] if sampler.existing_xi_d is not None else 0})"
        )
    else:
        # ----- Steady path: unchanged 1:1 mapping -------------------------
        from coupling.src.functions.coupling_classes import MetaModel3 as MetaModel

        existing_file = "existing_xi_d.npy"
        init_cond = c_iter == 1 and lb_iter == 1

        theta, degree = None, None
        order = None
        metamodel = MetaModel()

        if init_cond:
            tasks, xi_d = metamodel.build(xi_rot, order, init=True, theta=theta)
            existing_xi_d = xi_d.copy()
        else:
            existing_xi_d = np.load(os.path.join(output_dir, existing_file))
            metamodel.existing_xi_d = existing_xi_d
            tasks, xi_d = metamodel.build(xi_rot, None, init=False, theta=theta)
            existing_xi_d = np.concatenate((existing_xi_d, xi_d), axis=1)

        existing_xi_d_file = os.path.join(output_dir, existing_file)
        np.save(existing_xi_d_file, existing_xi_d)

    tasks_file = os.path.join(output_dir, "tasks.npy")
    np.save(tasks_file, np.array(tasks, dtype=object))
    print(f"Saved {len(tasks)} tasks to {tasks_file}")

    xi_d_file = os.path.join(output_dir, "xi_d.npy")
    np.save(xi_d_file, xi_d)

    print("build_task_list.py completed successfully.")

    task_count_file = os.path.join(output_dir, "task_count.txt")
    with open(task_count_file, "a") as f:
        f.write(f"{len(tasks)}\n")


if __name__ == "__main__":
    main()
