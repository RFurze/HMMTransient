import numpy as np
from coupling.src.functions.coupling_helper_fns import build_task_list_transient
from scipy.spatial import cKDTree
import math
import logging
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from CONFIGPenalty import ND_FACTOR


class MetaModel3:
    def __init__(self, Nd_factor=ND_FACTOR):
        self.existing_xi_d = (
            None  # Will hold accepted points with shape (9, n_points) now
        )
        self.dQd = None
        self.dPd = None
        self.Fstd = None
        self.Nd_factor = Nd_factor
        self.sampling_state = None

    @staticmethod
    def _min_max_scale(data):
        """Given 1D numpy array 'data', returns (scaled_data, data_min, data_max)."""
        data_min = np.min(data)
        data_max = np.max(data)
        span = data_max - data_min
        if abs(span) < 1e-15:
            span = 1.0
        scaled_data = (data - data_min) / span
        return scaled_data, data_min, data_max

    @staticmethod
    def _coverage_fraction(all_points_norm, subset_points_norm, r0):
        """
        all_points_norm: shape (N, D)
        subset_points_norm: shape (M, D)
        We say a point i in all_points is 'covered' if there's at least
        one subset_points_norm j within radius r0 in D-dimensional space.
        """
        if subset_points_norm.size == 0:
            return 0.0
        N = all_points_norm.shape[0]
        covered_count = 0
        for i in range(N):
            diffs = subset_points_norm - all_points_norm[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            if np.any(dists <= r0):
                covered_count += 1
        return covered_count / N

    @staticmethod
    def _choose_r0(all_points_norm: np.ndarray, q: float = 0.75) -> float:
        """
        Return the *q*-quantile of the 1-NN distance distribution
        (k-D tree query with k=2; the first hit is the point itself).
        If there are fewer than two points, return 0 so every point
        will be accepted.
        """
        if all_points_norm.shape[0] < 2:
            return 0.0
        tree = cKDTree(all_points_norm)
        d, _ = tree.query(all_points_norm, k=2)
        return float(np.quantile(d[:, 1], q))

    def set_sampling_state(self, state):
        """Set persisted normalisation/radius state for downsampling."""
        self.sampling_state = state

    def get_sampling_state(self):
        """Return the current persisted normalisation/radius state."""
        return self.sampling_state

    def _normalise_with_state(self, data, key):
        dmin = self.sampling_state[f"{key}_min"]
        dmax = self.sampling_state[f"{key}_max"]
        span = dmax - dmin
        if abs(span) < 1e-15:
            span = 1.0
        return (data - dmin) / span

    def build(self, xi, order, init, theta=None):
        """Build downsampled transient tasks with persistent sampling state."""
        H, P, U, V, lmbZ, dPdx, dPdy, gradPz, dHdx, dHdy, dHdz, Hdot, Pdot = xi

        # Ignore U, V, dHdx, dHdy in downsampling distance calculations.
        sample_specs = {
            "H": H,
            "P": P,
            "dPdx": dPdx,
            "dPdy": dPdy,
            "Hdot": Hdot,
            "Pdot": Pdot,
        }

        if init or self.sampling_state is None:
            state = {}
            norm_cols = []
            for key, values in sample_specs.items():
                vals_n, vmin, vmax = self._min_max_scale(values)
                state[f"{key}_min"] = float(vmin)
                state[f"{key}_max"] = float(vmax)
                norm_cols.append(vals_n)
            new_data_norm = np.column_stack(norm_cols)
            state["r0"] = self._choose_r0(new_data_norm, q=1.0 - self.Nd_factor)
            self.sampling_state = state
        else:
            new_data_norm = np.column_stack(
                [self._normalise_with_state(v, k) for k, v in sample_specs.items()]
            )

        r0 = self.sampling_state["r0"]
        accepted_indices = []

        if init:
            indices_left = set(range(new_data_norm.shape[0]))
            while indices_left:
                idx = next(iter(indices_left))
                center = new_data_norm[idx]
                accepted_indices.append(idx)
                remove_list = [
                    j
                    for j in indices_left
                    if np.linalg.norm(new_data_norm[j] - center) <= r0
                ]
                for j in remove_list:
                    indices_left.remove(j)
        else:
            if self.existing_xi_d is None:
                raise ValueError(
                    "Called update mode but no existing_xi_d has been set yet."
                )

            ex = self.existing_xi_d
            existing_norm = np.column_stack(
                [
                    self._normalise_with_state(ex[0, :], "H"),
                    self._normalise_with_state(ex[1, :], "P"),
                    self._normalise_with_state(ex[5, :], "dPdx"),
                    self._normalise_with_state(ex[6, :], "dPdy"),
                    self._normalise_with_state(ex[11, :], "Hdot"),
                    self._normalise_with_state(ex[12, :], "Pdot"),
                ]
            )

            accepted_norm_pts = []
            for i in range(new_data_norm.shape[0]):
                candidate = new_data_norm[i]
                dist_existing = np.linalg.norm(existing_norm - candidate, axis=1)
                if accepted_norm_pts:
                    dist_new = np.linalg.norm(
                        np.asarray(accepted_norm_pts) - candidate, axis=1
                    )
                else:
                    dist_new = np.array([np.inf])

                if np.all(dist_existing > r0) and np.all(dist_new > r0):
                    accepted_indices.append(i)
                    accepted_norm_pts.append(candidate)

        if accepted_indices:
            new_points = np.asarray([np.asarray(comp)[accepted_indices] for comp in xi])
            if init or self.existing_xi_d is None:
                self.existing_xi_d = new_points
            else:
                self.existing_xi_d = np.concatenate(
                    (self.existing_xi_d, new_points), axis=1
                )
            xi_d = [new_points[i, :] for i in range(new_points.shape[0])]
        else:
            xi_d = [np.array([]) for _ in range(13)]

        tasks = build_task_list_transient(xi_d)
        return tasks, xi_d

    def update_results(self, dq_results, dP_results):
        """Appends new micro-simulation results to dQd, dPd, Fstd."""
        if self.dQd is None:
            self.dQd = dq_results.copy()
            self.dPd = dP_results.copy()
        else:
            self.dQd = np.concatenate((self.dQd, dq_results), axis=0)
            self.dPd = np.concatenate((self.dPd, dP_results), axis=0)

    def load_results(
        self,
        dq_results,
        dP_results,
        taust_results,
        pmax_results,
        pmin_results,
        hmax_results,
        hmin_results,
    ):
        self.dQd = dq_results.copy()
        self.dPd = dP_results.copy()
        self.taustd = taust_results.copy()
        self.pmaxd = pmax_results.copy()
        self.pmind = pmin_results.copy()
        self.hmaxd = hmax_results.copy()
        self.hmind = hmin_results.copy()

    def get_training_matrix(self):
        """Return cumulative training predictors used by transient MLS.

        Columns: [H, P, dPdx, dPdy, Hdot, Pdot]
        """
        if self.existing_xi_d is None:
            return np.zeros((0, 6))

        ex = self.existing_xi_d  # shape (13, n_points)
        Hvals = ex[0, :]
        Pvals = ex[1, :]
        dPdxv = ex[5, :]
        dPdyv = ex[6, :]
        Hdotvals = ex[11, :]
        Pdotvals = ex[12, :]
        return np.column_stack([Hvals, Pvals, dPdxv, dPdyv, Hdotvals, Pdotvals])


def weighted_distance(a, b, weight_f=1.0):
    # a, b: shape (D,) or (N, D)
    # We assume the last column is F
    diffs = a - b
    # Scale the final dimension
    diffs[..., -1] *= weight_f
    return np.linalg.norm(diffs, axis=-1)