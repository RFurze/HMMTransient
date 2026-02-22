import argparse


def parse_common_args(description: str, with_time: bool = False, MLS: bool = False):
    """Parse commonly used command line arguments.

    Parameters
    ----------
    description : str
        Description passed to :class:`argparse.ArgumentParser`.
    with_time : bool, optional
        Force inclusion of time related arguments.  If ``True`` or if
        ``--transient`` is supplied on the command line, ``--Time`` and
        ``--DT`` will be parsed.  Default is ``False``.
    MLS : bool, optional
        Include arguments specific to the MLS routines.
    """

    # First parse the ``--transient`` flag so that we know whether to
    # include the time related arguments below.
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--transient", action="store_true", help="Use the transient workflow"
    )
    known, _ = base.parse_known_args()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--transient", action="store_true", help="Use the transient workflow"
    )
    parser.add_argument(
        "--lb_iter", type=int, default=1, help="Current load-balance iteration index."
    )
    parser.add_argument(
        "--c_iter", type=int, default=1, help="Current coupling iteration index."
    )
    if with_time or known.transient:
        parser.add_argument(
            "--Time", type=float, default=0.0, help="Current simulation time."
        )
        parser.add_argument("--DT", type=float, default=1.0, help="Time-step size.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output/hmm_job",
        help="Directory to store output results.",
    )
    if MLS:
        parser.add_argument(
            "--k_neighbors",
            type=int,
            default=500,
            help="Number of nearest neighbors for MLS.",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=64,
            help="How many query points a worker solves at once",
        )
    return parser.parse_args()
