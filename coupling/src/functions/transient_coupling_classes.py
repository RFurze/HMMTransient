import logging
import numpy as np
from CONFIGPenalty import ND_FACTOR
from coupling.src.functions.coupling_helper_fns import build_task_list_transient


FEATURE_ROWS = [0, 1, 5, 6, 11, 12]  # H, P, dPdx, dPdy, Hdot, Pdot


class MetaModel3:
    def __init__(self, Nd_factor=ND_FACTOR):
        self.existing_xi_d = None
        self.dQd = None
        self.dPd = None
        self.Fstd = None
        self.Nd_factor = Nd_factor

    @staticmethod
    def _extract_features(xi_arr: np.ndarray) -> np.ndarray:
        """Return feature matrix (N, 6): H, P, dPdx, dPdy, Hdot, Pdot."""
        return np.column_stack([xi_arr[row, :] for row in FEATURE_ROWS])

    @staticmethod
    def _compute_mahalanobis_metric(points: np.ndarray):
        """Return mean and inverse covariance for stable Mahalanobis distances."""
        mean = points.mean(axis=0)
        centered = points - mean
        if points.shape[0] < 2:
            cov = np.eye(points.shape[1])
        else:
            cov = np.cov(centered, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
        cov += 1e-10 * np.eye(cov.shape[0])
        cov_inv = np.linalg.pinv(cov)
        return mean, cov_inv

    @staticmethod
    def _mahalanobis_distance_batch(candidates: np.ndarray, reference: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
        """Distance from each candidate (N,D) to one reference (D,), shape (N,)."""
        delta = candidates - reference
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, cov_inv, delta))

    def _choose_r0(self, points: np.ndarray, cov_inv: np.ndarray, q: float = 0.75) -> float:
        """Quantile of 1-NN Mahalanobis distances."""
        n = points.shape[0]
        if n < 2:
            return 0.0

        nn = np.empty(n, dtype=float)
        for i in range(n):
            dists = self._mahalanobis_distance_batch(points, points[i], cov_inv)
            dists[i] = np.inf
            nn[i] = np.min(dists)
        return float(np.quantile(nn, q))

    def _coverage_fraction(self, all_points: np.ndarray, subset_points: np.ndarray, r0: float, cov_inv: np.ndarray) -> float:
        if subset_points.size == 0:
            return 0.0
        covered = 0
        for i in range(all_points.shape[0]):
            dists = self._mahalanobis_distance_batch(subset_points, all_points[i], cov_inv)
            if np.any(dists <= r0):
                covered += 1
        return covered / all_points.shape[0]

    def build(self, xi, order, init, theta=None):
        H, P, U, V, lmbZ, dPdx, dPdy, gradPz, dHdx, dHdy, dHdz, Hdot, Pdot = xi

        xi_arr = np.vstack([
            H, P, U, V, lmbZ, dPdx, dPdy, gradPz, dHdx, dHdy, dHdz, Hdot, Pdot
        ])
        batch_features = self._extract_features(xi_arr)

        if init or self.existing_xi_d is None:
            ex = np.zeros((13, 0), dtype=float)
            existing_features = np.zeros((0, len(FEATURE_ROWS)), dtype=float)
        else:
            ex = self.existing_xi_d
            existing_features = self._extract_features(ex)

        all_features = np.vstack([existing_features, batch_features])
        _, cov_inv = self._compute_mahalanobis_metric(all_features)
        r0 = self._choose_r0(all_features, cov_inv, q=1.0 - self.Nd_factor)

        accepted_indices = []
        accepted_features = []

        if init or existing_features.shape[0] == 0:
            indices_left = set(range(batch_features.shape[0]))
            while indices_left:
                idx = next(iter(indices_left))
                center = batch_features[idx]
                accepted_indices.append(idx)
                accepted_features.append(center)
                dists = self._mahalanobis_distance_batch(batch_features, center, cov_inv)
                remove_idx = np.where(dists <= r0)[0]
                for ridx in remove_idx:
                    indices_left.discard(int(ridx))
        else:
            frac_before = self._coverage_fraction(batch_features, existing_features, r0, cov_inv)
            logging.info(f"[UPDATE] Coverage fraction BEFORE = {frac_before:.3f}")

            for i in range(batch_features.shape[0]):
                candidate = batch_features[i]
                d_ex = self._mahalanobis_distance_batch(existing_features, candidate, cov_inv)
                if accepted_features:
                    d_new = self._mahalanobis_distance_batch(np.asarray(accepted_features), candidate, cov_inv)
                else:
                    d_new = np.array([np.inf])

                if np.all(d_ex > r0) and np.all(d_new > r0):
                    accepted_indices.append(i)
                    accepted_features.append(candidate)

        if accepted_indices:
            new_points = xi_arr[:, accepted_indices]
            self.existing_xi_d = np.concatenate((ex, new_points), axis=1)
        else:
            self.existing_xi_d = ex

        if not accepted_indices:
            xi_d = [np.array([]) for _ in range(13)]
            logging.info("No new points accepted this call => xi_d is empty.")
        else:
            xi_d = [self.existing_xi_d[i, -len(accepted_indices):] for i in range(13)]

        updated_features = self._extract_features(self.existing_xi_d) if self.existing_xi_d.shape[1] else np.zeros((0, 6))
        frac_after = self._coverage_fraction(batch_features, updated_features, r0, cov_inv)
        logging.info(f"[BUILD] Accepted {len(accepted_indices)} new points; coverage AFTER = {frac_after:.3f}")

        tasks = build_task_list_transient(xi_d)
        return tasks, xi_d

    def update_results(self, dq_results, dP_results):
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
        if self.existing_xi_d is None:
            return np.zeros((0, len(FEATURE_ROWS)))
        return self._extract_features(self.existing_xi_d)
