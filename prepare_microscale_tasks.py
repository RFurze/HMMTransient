"""Create the microscale task list used by the micro sims.

This utility builds the array of microscale simulation inputs (``tasks.npy``)
from the rotated state ``xi_rot.npy``.  When run with ``--transient`` the time
options ``--Time`` and ``--DT`` are parsed as well.

Key command line options:
    ``--lb_iter`` and ``--c_iter`` - current iteration counters.
    ``--output_dir`` - where ``tasks.npy`` and ``xi_d.npy`` are written.
"""

import os
import numpy as np
from utils.cli import parse_common_args


def main():
    """Build the list of microscale tasks, optionally in transient mode."""
    args = parse_common_args("Build Microscale Task List")

    lb_iter = args.lb_iter
    c_iter = args.c_iter
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.transient:
        from coupling.src.functions.transient_coupling_classes import (
            MetaModel3 as MetaModel,
        )

        T = args.Time
        DT = args.DT
        step_idx = int(round(T / DT)) if DT != 0 else int(T)
        prefix = f"transient_step{step_idx}_"
        existing_file = "transient_existing_xi_d.npy"
        init_cond = lb_iter == 0 and c_iter == 1
        if not os.path.exists(os.path.join(output_dir, existing_file)):
            legacy_file = os.path.join(output_dir, f"{prefix}existing_xi_d.npy")
            init_cond = not os.path.exists(legacy_file)
        print(
            f"Starting build_task_list.py for T = {T}, lb_iter={lb_iter}, c_iter={c_iter}"
        )
    else:
        from coupling.src.functions.coupling_classes import MetaModel3 as MetaModel

        existing_file = "existing_xi_d.npy"
        init_cond = c_iter == 1 and lb_iter == 1
        print(f"Starting build_task_list.py for lb_iter={lb_iter}, c_iter={c_iter}")

    theta, degree = None, None
    order = None
    metamodel = MetaModel()
    xi_rot = np.load(os.path.join(output_dir, "xi_rot.npy"))

    if init_cond:
        tasks, xi_d = metamodel.build(xi_rot, order, init=True, theta=theta)
        existing_xi_d = xi_d.copy()
    else:
        existing_path = os.path.join(output_dir, existing_file)
        if not os.path.exists(existing_path) and args.transient:
            legacy_path = os.path.join(output_dir, f"{prefix}existing_xi_d.npy")
            existing_path = legacy_path if os.path.exists(legacy_path) else existing_path
        existing_xi_d = np.load(existing_path)
        metamodel.existing_xi_d = existing_xi_d
        tasks, xi_d = metamodel.build(xi_rot, None, init=False, theta=theta)
        existing_xi_d = np.concatenate((existing_xi_d, xi_d), axis=1)

    existing_xi_d_file = os.path.join(output_dir, existing_file)
    np.save(existing_xi_d_file, existing_xi_d)
    print(f"Saved existing_xi_d to {existing_xi_d_file}")

    tasks_file = os.path.join(output_dir, "tasks.npy")
    np.save(tasks_file, np.array(tasks, dtype=object))
    print(f"Saved {len(tasks)} tasks to {tasks_file}")

    xi_d_file = os.path.join(output_dir, "xi_d.npy")
    np.save(xi_d_file, xi_d)
    print(f"Saved xi_d to {xi_d_file}")

    print("build_task_list.py completed successfully.")

    task_count_file = os.path.join(output_dir, "task_count.txt")
    with open(task_count_file, "a") as f:
        f.write(f"{len(tasks)}\n")


if __name__ == "__main__":
    main()
