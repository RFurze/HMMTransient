"""Error-Driven Adaptive Sampling (EDAS) for transient multiscale coupling.

This module replaces the geometric-coverage downsampling in
:class:`transient_coupling_classes.MetaModel3` with an approach that
unifies downsampling and upsampling through MLS prediction error.

Key ideas
---------
1.  A **SharedNormaliser** ensures that feature scaling is identical in
    the sampler and in MLS, and is stable across coupling iterations.
    For the dynamic features (Hdot, Pdot) that change between coupling
    iterations, the normaliser is reset per time step to avoid stale
    range dilution.
2.  An **ErrorDrivenSampler** uses LOOCV error estimates and coverage
    distance to decide *where* new microscale simulations are needed.
3.  Training data is relevance-weighted using **coupling-iteration decay**
    (not just time-decay) so that data with stale Hdot/Pdot values from
    earlier coupling iterations fades quickly.
4.  A refinement loop in the shell driver repeats sampling/MLS passes
    until the maximum pointwise error indicator drops below a target, or
    a per-iteration budget is exhausted.
"""

import numpy as np
from scipy.spatial import cKDTree

from coupling.src.functions.coupling_helper_fns import build_task_list_transient


# ---- Feature indices in the full 13-component xi vector ------------------
_FEATURE_XI_INDICES = [0, 1, 5, 6, 11, 12]
_FEATURE_NAMES = ["H", "P", "dPdx", "dPdy", "Hdot", "Pdot"]
_N_XI_COMPONENTS = 13

# Static features (don't change within a time step's coupling iterations)
_STATIC_FEATURE_POSITIONS = [0, 1, 2, 3]   # H, P, dPdx, dPdy in the 6-feature vector
# Dynamic features (change every coupling iteration)
_DYNAMIC_FEATURE_POSITIONS = [4, 5]         # Hdot, Pdot in the 6-feature vector


# ---------------------------------------------------------------------------
# SharedNormaliser
# ---------------------------------------------------------------------------

class SharedNormaliser:
    """Monotonically-widening min/max normaliser.

    Bounds only expand (never shrink), so data normalised in an earlier
    iteration is still consistently scaled when new data arrives that
    extends the range.

    The state can be persisted to disk via :meth:`get_state` /
    :meth:`from_state` and is shared between the sampler and MLS.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.running_min: np.ndarray | None = None
        self.running_max: np.ndarray | None = None

    # -- persistence --------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "running_min": self.running_min.copy() if self.running_min is not None else None,
            "running_max": self.running_max.copy() if self.running_max is not None else None,
            "n_features": self.n_features,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SharedNormaliser":
        obj = cls(state["n_features"])
        obj.running_min = state["running_min"]
        obj.running_max = state["running_max"]
        return obj

    # -- core API -----------------------------------------------------------

    def update(self, X: np.ndarray) -> None:
        """Widen bounds to encompass *X* (n_samples, n_features)."""
        if X.size == 0:
            return
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        if self.running_min is None:
            self.running_min = xmin.copy()
            self.running_max = xmax.copy()
        else:
            self.running_min = np.minimum(self.running_min, xmin)
            self.running_max = np.maximum(self.running_max, xmax)

    def reset_dynamic_features(
        self,
        X_current: np.ndarray,
        dynamic_positions: list[int] | None = None,
    ) -> None:
        """Reset normaliser bounds for dynamic features to the current data.

        This prevents stale ranges from previous coupling iterations from
        diluting the normalised space for features that change between
        iterations (Hdot, Pdot).

        Parameters
        ----------
        X_current : (N, D)
            Current feature data to set dynamic bounds from.
        dynamic_positions : list of int
            Column indices in X_current that are dynamic.  Defaults to
            the module-level ``_DYNAMIC_FEATURE_POSITIONS``.
        """
        if dynamic_positions is None:
            dynamic_positions = _DYNAMIC_FEATURE_POSITIONS
        if self.running_min is None or X_current.size == 0:
            return
        xmin = X_current.min(axis=0)
        xmax = X_current.max(axis=0)
        for j in dynamic_positions:
            self.running_min[j] = xmin[j]
            self.running_max[j] = xmax[j]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale *X* to [0, 1] using current bounds."""
        if self.running_min is None:
            raise RuntimeError("SharedNormaliser has not been fitted yet.")
        rng = self.running_max - self.running_min
        rng[rng < 1e-15] = 1.0
        return (X - self.running_min) / rng

    def update_and_transform(self, X: np.ndarray) -> np.ndarray:
        self.update(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# Relevance weighting
# ---------------------------------------------------------------------------

def compute_relevance_weights(
    training_timestamps: np.ndarray,
    current_time: float,
    training_xi_norm: np.ndarray,
    query_xi_norm: np.ndarray,
    lambda_decay: float = 2.0,
    sigma_spatial: float = 0.3,
    coupling_iters: np.ndarray | None = None,
    current_coupling_iter: int | None = None,
    coupling_decay: float = 0.5,
) -> np.ndarray:
    """Weight each training point by recency and proximity to the current
    query distribution.

    Parameters
    ----------
    training_timestamps : (N_train,)
        The simulation time at which each training point was created.
    current_time : float
        Current simulation time *T*.
    training_xi_norm : (N_train, D)
        Normalised feature coordinates of training points.
    query_xi_norm : (N_query, D)
        Normalised feature coordinates of all macroscale nodes.
    lambda_decay : float
        Exponential time-decay rate.
    sigma_spatial : float
        Lengthscale for spatial relevance kernel.
    coupling_iters : (N_train,) or None
        The coupling iteration at which each training point was created.
        When provided, enables coupling-iteration-aware decay which is
        critical because Hdot/Pdot change every coupling iteration.
    current_coupling_iter : int or None
        Current coupling iteration number.
    coupling_decay : float
        Per-coupling-iteration decay factor.  A point from *k* iterations
        ago gets weight ``coupling_decay ** k``.

    Returns
    -------
    weights : (N_train,)
        Composite relevance weight in (0, 1].
    """
    # Time decay
    age = np.abs(current_time - training_timestamps)
    w_time = np.exp(-lambda_decay * age)

    # Coupling-iteration decay (critical for within-timestep convergence)
    if coupling_iters is not None and current_coupling_iter is not None:
        iter_age = np.abs(current_coupling_iter - coupling_iters)
        w_coupling = np.power(coupling_decay, iter_age)
    else:
        w_coupling = np.ones_like(w_time)

    # Spatial relevance: distance to nearest query point
    if query_xi_norm.shape[0] > 0 and training_xi_norm.shape[0] > 0:
        tree = cKDTree(query_xi_norm)
        d_nearest, _ = tree.query(training_xi_norm, k=1)
        w_spatial = np.exp(-d_nearest ** 2 / (2.0 * sigma_spatial ** 2))
    else:
        w_spatial = np.ones(training_xi_norm.shape[0])

    return w_time * w_coupling * w_spatial


# ---------------------------------------------------------------------------
# Error indicators
# ---------------------------------------------------------------------------

def compute_error_indicators(
    X_train_norm: np.ndarray,
    Y_train: np.ndarray,
    X_query_norm: np.ndarray,
    theta: float,
    degree: int,
    k_loocv: int = 30,
    w_thresh: float = 1e-3,
    alpha_blend: float = 0.5,
) -> np.ndarray:
    """Compute a per-query-point error indicator combining coverage gap
    and estimated LOOCV prediction error.

    Parameters
    ----------
    X_train_norm : (N_train, D)
        Normalised training features.
    Y_train : (N_train,) or (N_train, n_outputs)
        Training responses (one column per output).  If 2-D the error is
        the max across outputs at each point.
    X_query_norm : (N_query, D)
        Normalised query features.
    theta : float
        MLS Gaussian bandwidth parameter.
    degree : int
        MLS polynomial degree.
    k_loocv : int
        Neighbours used for local LOOCV estimates.
    w_thresh : float
        Weight threshold for effective neighbours.
    alpha_blend : float
        Blending weight: ``alpha * coverage + (1-alpha) * prediction``.

    Returns
    -------
    epsilon : (N_query,)
        Error indicator (higher = more uncertain).
    """
    N_train = X_train_norm.shape[0]
    N_query = X_query_norm.shape[0]

    if N_train == 0:
        return np.ones(N_query)

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]
    n_out = Y_train.shape[1]

    # ----- 1. Coverage component: distance to nearest training point ------
    train_tree = cKDTree(X_train_norm)
    d_nearest, _ = train_tree.query(X_query_norm, k=1)
    # Normalise by median inter-training distance for a relative measure
    if N_train >= 2:
        d_train, _ = train_tree.query(X_train_norm, k=2)
        median_spacing = float(np.median(d_train[:, 1]))
        if median_spacing < 1e-15:
            median_spacing = 1.0
    else:
        median_spacing = 1.0
    eps_coverage = d_nearest / median_spacing  # dimensionless

    # ----- 2. Prediction component: local LOOCV error estimate -----------
    # Compute LOO residual at every training point using local MLS.
    k_use = min(k_loocv, N_train - 1) if N_train > 1 else 0
    loo_errors = np.zeros(N_train)

    if k_use >= 2:
        from coupling.src.functions.multinom_MLS_par import multinom_coeffs
        D = X_train_norm.shape[1]
        C, Nt = multinom_coeffs(degree, D)

        # Build polynomial design matrix for all training points
        Mat_all = np.ones((N_train, Nt), dtype=float)
        for i_exp in range(Nt):
            for j_col in range(D):
                exp_j = C[i_exp, j_col]
                if exp_j != 0:
                    Mat_all[:, i_exp] *= X_train_norm[:, j_col] ** exp_j

        # k+1 neighbours because first hit is the point itself
        k_query = min(k_use + 1, N_train)
        d_nn, idx_nn = train_tree.query(X_train_norm, k=k_query)

        for i in range(N_train):
            # Neighbours excluding the point itself
            nbrs = idx_nn[i, 1:]
            dists = d_nn[i, 1:]
            if len(nbrs) == 0:
                continue

            wght = np.exp(-theta * dists ** 2)
            w_max = wght.max()
            if w_max < 1e-15:
                loo_errors[i] = 0.0
                continue

            mask = wght >= w_thresh * w_max
            if mask.sum() < Nt:
                mask = np.ones(len(nbrs), dtype=bool)

            sel = nbrs[mask]
            w_sel = wght[mask]
            Mat_sel = Mat_all[sel]
            Matw = Mat_sel * w_sel[:, None]

            # Per-output LOO error
            max_err = 0.0
            for c in range(n_out):
                Y_sel = Y_train[sel, c]
                Pw = Y_sel * w_sel
                try:
                    alpha, *_ = np.linalg.lstsq(Matw, Pw, rcond=None)
                except np.linalg.LinAlgError:
                    max_err = max(max_err, 1.0)
                    continue
                # Predict at held-out point
                mat_i = Mat_all[i]
                y_pred = mat_i @ alpha
                y_range = Y_train[:, c].ptp()
                if y_range < 1e-15:
                    y_range = 1.0
                err = abs(y_pred - Y_train[i, c]) / y_range
                max_err = max(max_err, err)

            loo_errors[i] = max_err

    # Interpolate LOO errors to query points using nearest-neighbour
    # weighting from training points.
    if N_train > 0 and N_query > 0 and k_use >= 2:
        k_interp = min(5, N_train)
        d_interp, idx_interp = train_tree.query(X_query_norm, k=k_interp)
        # Inverse-distance weighted average of LOO errors at nearby training points
        eps_prediction = np.zeros(N_query)
        for i in range(N_query):
            dists = d_interp[i]
            idxs = idx_interp[i]
            if np.ndim(dists) == 0:
                dists = np.array([dists])
                idxs = np.array([idxs])
            # Use inverse distance weights (with a floor to avoid div-by-zero)
            inv_d = 1.0 / (dists + 1e-12)
            weights = inv_d / inv_d.sum()
            eps_prediction[i] = np.dot(weights, loo_errors[idxs])
    else:
        eps_prediction = np.ones(N_query)

    # ----- 3. Composite indicator ----------------------------------------
    epsilon = alpha_blend * eps_coverage + (1.0 - alpha_blend) * eps_prediction
    return epsilon


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(
    epsilon: np.ndarray,
    X_query_norm: np.ndarray,
    X_train_norm: np.ndarray | None,
    batch_size: int,
    delta_min: float = 0.0,
) -> np.ndarray:
    """Select the *batch_size* query points with the highest error indicator,
    subject to a minimum-spacing constraint *delta_min* in normalised space.

    Parameters
    ----------
    epsilon : (N_query,)
    X_query_norm : (N_query, D)
    X_train_norm : (N_train, D) or None
        Existing training points.  New samples must be at least *delta_min*
        away from these *and* from each other.
    batch_size : int
    delta_min : float

    Returns
    -------
    selected : 1-D integer array of indices into the query arrays.
    """
    order = np.argsort(-epsilon)  # descending error

    selected = []
    selected_norm = []

    # Build the training-set tree ONCE (previously rebuilt every candidate)
    train_tree = None
    if delta_min > 0 and X_train_norm is not None and X_train_norm.shape[0] > 0:
        train_tree = cKDTree(X_train_norm)

    for idx in order:
        if len(selected) >= batch_size:
            break

        candidate = X_query_norm[idx]

        # Spacing check against existing training set
        if train_tree is not None:
            d, _ = train_tree.query(candidate.reshape(1, -1), k=1)
            if d[0] < delta_min:
                continue

        # Spacing check against already-selected points in this batch
        if delta_min > 0 and selected_norm:
            dists = np.linalg.norm(
                np.asarray(selected_norm) - candidate, axis=1
            )
            if np.any(dists < delta_min):
                continue

        selected.append(int(idx))
        selected_norm.append(candidate)

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Initial coverage sampling (cold-start)
# ---------------------------------------------------------------------------

def initial_coverage_sample(
    X_query_norm: np.ndarray,
    target_fraction: float = 0.05,
    r0_quantile: float = 0.25,
) -> np.ndarray:
    """Greedy geometric coverage for the first iteration when no training
    data exists yet.

    This is essentially the same algorithm as the old MetaModel3 ``init``
    path, but with deterministic ordering (sorted by L2 norm from centroid)
    and a configurable target fraction.

    Returns
    -------
    selected : 1-D integer array of selected query indices.
    """
    N = X_query_norm.shape[0]
    if N == 0:
        return np.array([], dtype=int)

    # Compute r0 from the 1-NN distance distribution
    tree = cKDTree(X_query_norm)
    if N >= 2:
        d_nn, _ = tree.query(X_query_norm, k=2)
        r0 = float(np.quantile(d_nn[:, 1], r0_quantile))
    else:
        return np.array([0], dtype=int)

    # Deterministic ordering: sort by distance to centroid so that
    # sampling starts near the centre and spreads outward
    centroid = X_query_norm.mean(axis=0)
    dist_to_c = np.linalg.norm(X_query_norm - centroid, axis=1)
    order = np.argsort(dist_to_c)

    selected = []
    remaining = set(range(N))

    for idx in order:
        if idx not in remaining:
            continue
        selected.append(idx)
        # Remove all points within r0 of the accepted point
        center = X_query_norm[idx]
        to_remove = [
            j for j in remaining
            if np.linalg.norm(X_query_norm[j] - center) <= r0
        ]
        for j in to_remove:
            remaining.discard(j)

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# ErrorDrivenSampler - main class
# ---------------------------------------------------------------------------

class ErrorDrivenSampler:
    """Transient EDAS sampler that replaces MetaModel3 for the transient path.

    Manages:
    - A :class:`SharedNormaliser` for consistent feature scaling.
    - Accumulated training data (``existing_xi_d``) with timestamps
      and coupling-iteration tags.
    - Error-indicator-driven sample selection.
    - Relevance weighting for training data pruning using both
      time-decay and coupling-iteration decay.
    """

    def __init__(
        self,
        batch_size: int = 200,
        max_budget: int = 1000,
        error_target: float = 0.05,
        alpha_blend: float = 0.5,
        delta_min_quantile: float = 0.1,
        lambda_decay: float = 2.0,
        sigma_spatial: float = 0.3,
        relevance_prune_threshold: float = 0.01,
        r0_quantile: float = 0.25,
        coupling_decay: float = 0.5,
    ):
        self.batch_size = batch_size
        self.max_budget = max_budget
        self.error_target = error_target
        self.alpha_blend = alpha_blend
        self.delta_min_quantile = delta_min_quantile
        self.lambda_decay = lambda_decay
        self.sigma_spatial = sigma_spatial
        self.relevance_prune_threshold = relevance_prune_threshold
        self.r0_quantile = r0_quantile
        self.coupling_decay = coupling_decay

        self.normaliser = SharedNormaliser(len(_FEATURE_XI_INDICES))

        # Accumulated training data: (n_xi_components, N_train)
        self.existing_xi_d: np.ndarray | None = None
        # Timestamp per training point
        self.timestamps: np.ndarray | None = None
        # Coupling iteration per training point
        self.coupling_iters: np.ndarray | None = None

    # -- persistence --------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "normaliser": self.normaliser.get_state(),
            "existing_xi_d": self.existing_xi_d,
            "timestamps": self.timestamps,
            "coupling_iters": self.coupling_iters,
            "batch_size": self.batch_size,
            "max_budget": self.max_budget,
            "error_target": self.error_target,
            "alpha_blend": self.alpha_blend,
            "delta_min_quantile": self.delta_min_quantile,
            "lambda_decay": self.lambda_decay,
            "sigma_spatial": self.sigma_spatial,
            "relevance_prune_threshold": self.relevance_prune_threshold,
            "r0_quantile": self.r0_quantile,
            "coupling_decay": self.coupling_decay,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ErrorDrivenSampler":
        obj = cls(
            batch_size=state.get("batch_size", 200),
            max_budget=state.get("max_budget", 1000),
            error_target=state.get("error_target", 0.05),
            alpha_blend=state.get("alpha_blend", 0.5),
            delta_min_quantile=state.get("delta_min_quantile", 0.1),
            lambda_decay=state.get("lambda_decay", 2.0),
            sigma_spatial=state.get("sigma_spatial", 0.3),
            relevance_prune_threshold=state.get("relevance_prune_threshold", 0.01),
            r0_quantile=state.get("r0_quantile", 0.25),
            coupling_decay=state.get("coupling_decay", 0.5),
        )
        obj.normaliser = SharedNormaliser.from_state(state["normaliser"])
        obj.existing_xi_d = state.get("existing_xi_d")
        obj.timestamps = state.get("timestamps")
        obj.coupling_iters = state.get("coupling_iters")
        return obj

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_features(xi: np.ndarray) -> np.ndarray:
        """Extract the 6 sampling features from a (13, N) xi array.

        Returns shape (N, 6).
        """
        return np.column_stack([xi[i] for i in _FEATURE_XI_INDICES])

    def _get_train_features_norm(self) -> np.ndarray:
        """Return normalised training features (N_train, 6)."""
        if self.existing_xi_d is None or self.existing_xi_d.shape[1] == 0:
            return np.zeros((0, len(_FEATURE_XI_INDICES)))
        raw = self._extract_features(self.existing_xi_d)
        return self.normaliser.transform(raw)

    def _compute_delta_min(self, X_norm: np.ndarray) -> float:
        """Derive a minimum spacing from the query point distribution."""
        if X_norm.shape[0] < 2:
            return 0.0
        tree = cKDTree(X_norm)
        d_nn, _ = tree.query(X_norm, k=2)
        return float(np.quantile(d_nn[:, 1], self.delta_min_quantile))

    # -- core API -----------------------------------------------------------

    def build(
        self,
        xi: list | np.ndarray,
        current_time: float,
        init: bool = False,
        mls_errors: np.ndarray | None = None,
        theta: float = 5000.0,
        degree: int = 2,
        coupling_iter: int = 1,
    ):
        """Select microscale simulation points from the current macroscale state.

        Parameters
        ----------
        xi : array-like
            13-element list of arrays, each of length N (macroscale nodes).
        current_time : float
            Current simulation time *T*.
        init : bool
            If True, use geometric coverage (cold-start).
        mls_errors : (N,) or None
            Per-node error indicators from a previous MLS evaluation.
            If None and not init, falls back to coverage-only selection.
        theta : float
            MLS theta for LOOCV error estimation.
        degree : int
            MLS polynomial degree for LOOCV error estimation.
        coupling_iter : int
            Current coupling iteration number (for coupling-iteration
            decay tagging).

        Returns
        -------
        tasks : list of tuples
            Microscale task list for the selected points.
        xi_d : list of arrays
            xi components for the selected points only.
        selected_indices : ndarray
            Indices into the macroscale mesh of the selected points.
        """
        xi = [np.asarray(comp) for comp in xi]
        N = len(xi[0])

        # Extract and normalise query features
        X_query_raw = self._extract_features(xi)

        # Reset dynamic feature normaliser bounds to current data to
        # prevent stale Hdot/Pdot ranges from diluting the space.
        self.normaliser.update(X_query_raw)
        self.normaliser.reset_dynamic_features(X_query_raw)
        X_query_norm = self.normaliser.transform(X_query_raw)

        if init or self.existing_xi_d is None or self.existing_xi_d.shape[1] == 0:
            # --- Cold-start: geometric coverage ---------------------------
            selected = initial_coverage_sample(
                X_query_norm,
                r0_quantile=self.r0_quantile,
            )
            # Cap to budget
            if len(selected) > self.max_budget:
                selected = selected[: self.max_budget]
        else:
            # --- Error-driven selection -----------------------------------
            X_train_norm = self._get_train_features_norm()

            if mls_errors is not None:
                # Use pre-computed error indicators from run_MLS.py
                epsilon = mls_errors
            else:
                # Compute error indicators from LOOCV on training data
                # We need training Y; use coverage-only as fallback
                # (prediction component will be all-ones)
                epsilon = compute_error_indicators(
                    X_train_norm,
                    np.zeros(X_train_norm.shape[0]),  # dummy Y
                    X_query_norm,
                    theta=theta,
                    degree=degree,
                    alpha_blend=1.0,  # coverage-only
                )

            delta_min = self._compute_delta_min(X_query_norm)

            # Only apply spacing constraint against current-iteration
            # training data — stale data should not block new samples
            # from being placed where they are most needed.
            if self.coupling_iters is not None and self.coupling_iters.size > 0:
                current_mask = self.coupling_iters == coupling_iter
                if current_mask.any():
                    current_train = X_train_norm[current_mask]
                else:
                    current_train = None
            else:
                current_train = X_train_norm

            selected = select_samples(
                epsilon,
                X_query_norm,
                current_train,
                batch_size=self.batch_size,
                delta_min=delta_min,
            )

        # --- Build outputs ------------------------------------------------
        if len(selected) > 0:
            new_points = np.asarray(
                [np.asarray(comp)[selected] for comp in xi]
            )
            new_timestamps = np.full(len(selected), current_time)
            new_coupling_iters = np.full(len(selected), coupling_iter)

            # Append to accumulated training data
            if self.existing_xi_d is None or self.existing_xi_d.shape[1] == 0:
                self.existing_xi_d = new_points
                self.timestamps = new_timestamps
                self.coupling_iters = new_coupling_iters
            else:
                self.existing_xi_d = np.concatenate(
                    (self.existing_xi_d, new_points), axis=1
                )
                self.timestamps = np.concatenate(
                    (self.timestamps, new_timestamps)
                )
                self.coupling_iters = np.concatenate(
                    (self.coupling_iters, new_coupling_iters)
                )

            xi_d = [new_points[i, :] for i in range(new_points.shape[0])]
        else:
            xi_d = [np.array([]) for _ in range(_N_XI_COMPONENTS)]
            selected = np.array([], dtype=int)

        tasks = build_task_list_transient(xi_d)
        return tasks, xi_d, selected

    def prune_training_data(
        self,
        current_time: float,
        query_xi_norm: np.ndarray | None = None,
        current_coupling_iter: int | None = None,
    ) -> int:
        """Remove training points whose relevance weight has dropped below
        the threshold.  Returns the number of points pruned.
        """
        if self.existing_xi_d is None or self.existing_xi_d.shape[1] == 0:
            return 0
        if self.timestamps is None:
            return 0

        train_norm = self._get_train_features_norm()
        if query_xi_norm is None:
            query_xi_norm = train_norm

        weights = compute_relevance_weights(
            self.timestamps,
            current_time,
            train_norm,
            query_xi_norm,
            lambda_decay=self.lambda_decay,
            sigma_spatial=self.sigma_spatial,
            coupling_iters=self.coupling_iters,
            current_coupling_iter=current_coupling_iter,
            coupling_decay=self.coupling_decay,
        )

        keep = weights >= self.relevance_prune_threshold
        n_pruned = int((~keep).sum())

        if n_pruned > 0:
            self.existing_xi_d = self.existing_xi_d[:, keep]
            self.timestamps = self.timestamps[keep]
            if self.coupling_iters is not None:
                self.coupling_iters = self.coupling_iters[keep]

        return n_pruned

    def get_training_matrix(self) -> np.ndarray:
        """Return cumulative training predictors (N_train, 6).

        Columns: [H, P, dPdx, dPdy, Hdot, Pdot].
        """
        if self.existing_xi_d is None or self.existing_xi_d.shape[1] == 0:
            return np.zeros((0, len(_FEATURE_XI_INDICES)))
        return self._extract_features(self.existing_xi_d)

    def get_relevance_weights(
        self,
        current_time: float,
        query_xi_norm: np.ndarray | None = None,
        current_coupling_iter: int | None = None,
    ) -> np.ndarray:
        """Compute relevance weights for all training points."""
        if self.existing_xi_d is None or self.existing_xi_d.shape[1] == 0:
            return np.array([])
        if self.timestamps is None:
            return np.ones(self.existing_xi_d.shape[1])

        train_norm = self._get_train_features_norm()
        if query_xi_norm is None:
            query_xi_norm = train_norm

        return compute_relevance_weights(
            self.timestamps,
            current_time,
            train_norm,
            query_xi_norm,
            lambda_decay=self.lambda_decay,
            sigma_spatial=self.sigma_spatial,
            coupling_iters=self.coupling_iters,
            current_coupling_iter=current_coupling_iter,
            coupling_decay=self.coupling_decay,
        )

    def get_normaliser_state(self) -> dict:
        """Return the normaliser state for use by MLS."""
        return self.normaliser.get_state()
