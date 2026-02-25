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

    def build(self, xi, order, init, theta=None):
        """
        xi is a list of 11 arrays:
          xi = [
            h, p,
            lmbx, lmby, lmbz,
            gradpx, gradpy, gradpz,
            gradHx, gradHy, gradHz,
          ]

        'order' is an index array for reordering the vertices. We do:
            xi[i] = xi[i][order]
        to reorder them consistently.

        We want to run coverage on 6D space: (H, P, lmbx, dPdx, dPdy, f).

        This function must produce:
           1) self.existing_xi_d with shape (11, n_points).
           2) a new xi_d also with shape (11, n_new_points).
        """
        # 1) Reorder each array in xi
        # xi = [comp[order] for comp in xi] #orders the xi arrays in line with the snaking order separately calculated
        #   xi now = [h[order], p[order],
        #             lmbx[order], lmby[order], lmbz[order],
        #             gradpx[order], gradpy[order], gradpz[order],
        #             f[order]]

        # 2) Unpack them. We'll call lmbx => U in coverage, ignoring lmby,lmbz,gradpz for coverage
        H, P, U, V, lmbZ, dPdx, dPdy, gradPz, dHdx, dHdy, dHdz, Hdot, Pdot = xi

        # 3) Scale the 6 coverage variables
        Hn, H_min, H_max = self._min_max_scale(H)
        Pn, P_min, P_max = self._min_max_scale(P)
        Un, U_min, U_max = self._min_max_scale(U)
        Vn, V_min, V_max = self._min_max_scale(V)
        dPdxn, dPdx_min, dPdx_max = self._min_max_scale(dPdx)
        dPdyn, dPdy_min, dPdy_max = self._min_max_scale(dPdy)
        dHdxn, dHdx_min, dHdx_max = self._min_max_scale(dHdx)
        dHdyn, dHdy_min, dHdy_max = self._min_max_scale(dHdy)
        Hdotn, Hdot_min, Hdot_max = self._min_max_scale(Hdot)
        Pdotn, Pdot_min, Pdot_max = self._min_max_scale(Pdot)

        # 4) Combine into a matrix for coverage checking: shape (#points, 6)
        new_data_norm = np.column_stack(
            [Hn, Pn, Un, Vn, dPdxn, dPdyn, dHdxn, dHdyn, Hdotn, Pdotn]
        )

        # We'll keep track of newly accepted points in "new_unscaled" with shape (#,6).
        new_unscaled = np.zeros((0, 10))

        if init:
            # ==============================================================
            # INIT MODE: A simple "greedy cover" approach
            # ==============================================================
            r0 = self._choose_r0(new_data_norm, q=1.0 - self.Nd_factor)

            # --- optional diagnostic histogram --------------------------
            tree = cKDTree(new_data_norm)
            dists, _ = tree.query(new_data_norm, k=2)
            nn_dists = dists[:, 1]
            plt.figure(figsize=(6, 4))
            plt.hist(nn_dists, bins=50)
            plt.axvline(
                r0, color="red", linestyle="--", linewidth=1.5, label=rf"$r_0={r0:.3f}$"
            )
            plt.xlabel("Distance to nearest neighbour (normalized)")
            plt.ylabel("Frequency")
            plt.title("NN distance distribution (init)")
            plt.tight_layout()
            plt.savefig("T_nn_distance_histogram_init.png")
            plt.close()
            # -------------------------------------------------------------
            indices_left = set(range(len(new_data_norm)))
            chosen_centers = []
            while len(indices_left) > 0:
                idx = next(iter(indices_left))  # pick any index
                center = new_data_norm[idx]
                chosen_centers.append(center)
                # remove all points within r0
                remove_list = []
                for j in indices_left:
                    dist = np.linalg.norm(new_data_norm[j] - center)
                    # dist = weighted_distance(new_data_norm[j], center, weight_f=1.0) #weighted to f
                    if dist <= r0:
                        remove_list.append(j)
                for j in remove_list:
                    indices_left.remove(j)
            chosen_centers = np.array(chosen_centers, dtype=float)  # shape (M,6)

            # Coverage fraction w.r.t. new_data_norm
            frac_init = self._coverage_fraction(new_data_norm, chosen_centers, r0)
            logging.info(
                f"[INIT] Coverage fraction of new data by chosen set = {frac_init:.3f}"
            )

            # "Unscale" the chosen set from [0..1] back to physical units
            def unscale(col_norm, dmin, dmax):
                return col_norm * (dmax - dmin) + dmin

            # chosen_unscaled: shape (M,6)
            M = chosen_centers.shape[0]
            chosen_unscaled = np.zeros_like(chosen_centers)
            chosen_unscaled[:, 0] = unscale(chosen_centers[:, 0], H_min, H_max)
            chosen_unscaled[:, 1] = unscale(chosen_centers[:, 1], P_min, P_max)
            chosen_unscaled[:, 2] = unscale(chosen_centers[:, 2], U_min, U_max)
            chosen_unscaled[:, 3] = unscale(chosen_centers[:, 3], V_min, V_max)
            chosen_unscaled[:, 4] = unscale(chosen_centers[:, 4], dPdx_min, dPdx_max)
            chosen_unscaled[:, 5] = unscale(chosen_centers[:, 5], dPdy_min, dPdy_max)
            chosen_unscaled[:, 6] = unscale(chosen_centers[:, 6], dHdx_min, dHdx_max)
            chosen_unscaled[:, 7] = unscale(chosen_centers[:, 7], dHdy_min, dHdy_max)
            chosen_unscaled[:, 8] = unscale(chosen_centers[:, 8], Hdot_min, Hdot_max)
            chosen_unscaled[:, 9] = unscale(chosen_centers[:, 9], Pdot_min, Pdot_max)

            # Build self.existing_xi_d as shape (9, M)
            # We'll store:
            #   row 0 => H
            #   row 1 => P
            #   row 2 => lmbx
            #   row 3 => lmby  
            #   row 4 => lmbz  ( = 0 if you're ignoring it)
            #   row 5 => dPdx
            #   row 6 => dPdy
            #   row 7 => dPdz ( = 0 if ignoring)
            #   row 8 => dHdx
            #   row 9 => dHdy
            #   row 10 => dHdz ( = 0 if ignoring)
            #   row 11 => Hdot
            #   row 12 => Pdot
            existing_array = np.zeros((13, M))
            existing_array[0, :] = chosen_unscaled[:, 0]  # H
            existing_array[1, :] = chosen_unscaled[:, 1]  # P
            existing_array[2, :] = chosen_unscaled[:, 2]  # lmbx
            existing_array[3, :] = chosen_unscaled[:, 3]  # lmby
            # ignoring lmby,lmbz => store 0
            existing_array[4, :] = 0.0  # lmbz
            existing_array[5, :] = chosen_unscaled[:, 4]  # dPdx
            existing_array[6, :] = chosen_unscaled[:, 5]  # dPdy
            # ignoring gradPz => store 0
            existing_array[7, :] = 0.0  # gradPz
            existing_array[8, :] = chosen_unscaled[:, 6]  # dHdx
            existing_array[9, :] = chosen_unscaled[:, 7]  # dHdy
            existing_array[10, :] = 0.0  # dHdz
            existing_array[11, :] = chosen_unscaled[:, 8]  # Hdot
            existing_array[12, :] = chosen_unscaled[:, 9]  # Pdot

            self.existing_xi_d = existing_array
            new_unscaled = chosen_unscaled  # shape (M,13)

        else:
            # ==============================================================
            # UPDATE MODE: choose new points that are outside coverage of existing
            # ==============================================================
            if self.existing_xi_d is None:
                raise ValueError(
                    "Called update mode but no existing_xi_d has been set yet."
                )

            ex = self.existing_xi_d  # shape (11, old_count)
            # We'll re-scale columns from ex into the same normalized space used above.
            #   ex[0,:] = H   => scale w.r.t. (H_min,H_max)
            #   ex[1,:] = P   => scale w.r.t. (P_min,P_max)
            #   ex[2,:] = lmbx => scale w.r.t. (U_min,U_max)
            #   ex[3,:] = lmby => scale w.r.t. (V_min,V_max)
            #   ex[5,:] = dPdx => scale
            #   ex[6,:] = dPdy => scale
            #   ex[8,:] = dHdx => scale
            #   ex[9,:] = dHdx => scale
            #   ex[11,:] = Hdot => scale
            #   ex[12,:] = Pdot => scale
            # everything else is 0 or ignored.

            def scale_column(col, cmin, cmax):
                denom = cmax - cmin
                if abs(denom) < 1e-15:
                    denom = 1.0
                return (col - cmin) / denom

            # existing_norm shape => (# old, 6)
            Hx = scale_column(ex[0, :], H_min, H_max)
            Px = scale_column(ex[1, :], P_min, P_max)
            Ux = scale_column(ex[2, :], U_min, U_max)
            Vx = scale_column(ex[3, :], V_min, V_max)
            dx = scale_column(ex[5, :], dPdx_min, dPdx_max)
            dy = scale_column(ex[6, :], dPdy_min, dPdy_max)
            dHdx = scale_column(ex[8, :], dHdx_min, dHdx_max)
            dHdy = scale_column(ex[9, :], dHdy_min, dHdy_max)
            Hdotx = scale_column(ex[11, :], Hdot_min, Hdot_max)
            Pdotx = scale_column(ex[12, :], Pdot_min, Pdot_max)
            existing_norm = np.column_stack(
                [Hx, Px, Ux, Vx, dx, dy, dHdx, dHdy, Hdotx, Pdotx]
            )

            all_norm = np.vstack([existing_norm, new_data_norm])
            r0 = self._choose_r0(all_norm, q=1.0 - self.Nd_factor)

            # optional diagnostic plot (use new batch only)
            tree = cKDTree(new_data_norm)
            dists, _ = tree.query(new_data_norm, k=2)
            nn_dists = dists[:, 1]
            plt.figure(figsize=(6, 4))
            plt.hist(nn_dists, bins=50)
            plt.axvline(
                r0, color="red", linestyle="--", linewidth=1.5, label=rf"$r_0={r0:.3f}$"
            )
            plt.xlabel("Distance to nearest neighbour (normalized)")
            plt.ylabel("Frequency")
            plt.title("NN distance distribution (update)")
            plt.tight_layout()
            plt.savefig("T_nn_distance_histogram_update.png")
            plt.close()

            frac_before = self._coverage_fraction(new_data_norm, existing_norm, r0)
            logging.info(f"[UPDATE] Coverage fraction BEFORE = {frac_before:.3f}")

            accepted_indices = []
            accepted_norm_pts = []
            for i in range(new_data_norm.shape[0]):
                # distance to each existing center in 6D
                candidate = new_data_norm[i]
                dist_existing = weighted_distance(
                    existing_norm, candidate, weight_f=1.0
                )

                # distance to previously accepted new points
                if len(accepted_norm_pts) > 0:
                    dist_new = weighted_distance(
                        np.asarray(accepted_norm_pts), candidate, weight_f=1.0
                    )
                else:
                    dist_new = np.array([np.inf])

                if np.all(dist_existing > r0) and np.all(dist_new > r0):
                    accepted_indices.append(i)
                    accepted_norm_pts.append(candidate)
            logging.info(f"[UPDATE] Accepted {len(accepted_indices)} new points.")

            if len(accepted_indices) > 0:
                accepted_norm = new_data_norm[accepted_indices]  # shape (#acc,6)
                # unscale
                new_unscaled = np.zeros_like(accepted_norm)
                new_unscaled[:, 0] = accepted_norm[:, 0] * (H_max - H_min) + H_min
                new_unscaled[:, 1] = accepted_norm[:, 1] * (P_max - P_min) + P_min
                new_unscaled[:, 2] = accepted_norm[:, 2] * (U_max - U_min) + U_min
                new_unscaled[:, 3] = accepted_norm[:, 3] * (V_max - V_min) + V_min
                new_unscaled[:, 4] = (
                    accepted_norm[:, 4] * (dPdx_max - dPdx_min) + dPdx_min
                )
                new_unscaled[:, 5] = (
                    accepted_norm[:, 5] * (dPdy_max - dPdy_min) + dPdy_min
                )
                new_unscaled[:, 6] = (
                    accepted_norm[:, 6] * (dHdx_max - dHdx_min) + dHdx_min
                )
                new_unscaled[:, 7] = (
                    accepted_norm[:, 7] * (dHdy_max - dHdy_min) + dHdy_min
                )
                new_unscaled[:, 8] = (
                    accepted_norm[:, 8] * (Hdot_max - Hdot_min) + Hdot_min
                )
                new_unscaled[:, 9] = (
                    accepted_norm[:, 9] * (Pdot_max - Pdot_min) + Pdot_min
                )

                # Build new_points => shape (9, #acc)
                # fill the same rows
                n_acc = new_unscaled.shape[0]
                new_points = np.zeros((13, n_acc))
                new_points[0, :] = new_unscaled[:, 0]
                new_points[1, :] = new_unscaled[:, 1]
                new_points[2, :] = new_unscaled[:, 2]
                new_points[3, :] = new_unscaled[:, 3]
                # row 4 => 0 W
                new_points[5, :] = new_unscaled[:, 4]
                new_points[6, :] = new_unscaled[:, 5]
                # row 7 => 0 dpdz
                new_points[8, :] = new_unscaled[:, 6]
                new_points[9, :] = new_unscaled[:, 7]
                # row 10 => 0 dHdz
                new_points[11, :] = new_unscaled[:, 8] # Hdot
                new_points[12, :] = new_unscaled[:, 9]  # Pdot

                # concat
                self.existing_xi_d = np.concatenate((ex, new_points), axis=1)
                logging.info(
                    f"[UPDATE] Total points now = {self.existing_xi_d.shape[1]}"
                )

            # coverage fraction after
            ex_updated = self.existing_xi_d
            # rescale ex_updated
            Hx = scale_column(ex_updated[0, :], H_min, H_max)
            Px = scale_column(ex_updated[1, :], P_min, P_max)
            Ux = scale_column(ex_updated[2, :], U_min, U_max)
            Vx = scale_column(ex_updated[3, :], V_min, V_max)
            dx = scale_column(ex_updated[5, :], dPdx_min, dPdx_max)
            dy = scale_column(ex_updated[6, :], dPdy_min, dPdy_max)
            dHdx = scale_column(ex_updated[8, :], dHdx_min, dHdx_max)
            dHdy = scale_column(ex_updated[9, :], dHdy_min, dHdy_max)
            Hdotx = scale_column(ex_updated[11, :], Hdot_min, Hdot_max)
            Pdotx = scale_column(ex_updated[12, :], Pdot_min, Pdot_max)
            existing_norm_updated = np.column_stack(
                [Hx, Px, Ux, Vx, dx, dy, dHdx, dHdy, Hdotx, Pdotx]
            )
            frac_after = self._coverage_fraction(
                new_data_norm, existing_norm_updated, r0
            )
            logging.info(f"[UPDATE] Coverage fraction AFTER = {frac_after:.3f}")

        # ~~~~~ Build the return array xi_d  ~~~~~
        # new_unscaled has shape (# newly accepted, 6)
        # We want 9 arrays for final xi_d:
        #   [H, P, lmbx, 0, 0, dPdx, dPdy, 0, f]
        if new_unscaled.shape[0] == 0:
            logging.info("No new points accepted this call => xi_d is empty.")
            xi_d = [np.array([]) for _ in range(13)]
        else:
            Hd = new_unscaled[:, 0]
            Pd = new_unscaled[:, 1]
            Ud = new_unscaled[:, 2]
            Vd = new_unscaled[:, 3]
            dxd = new_unscaled[:, 4]
            dyd = new_unscaled[:, 5]
            dHdxd = new_unscaled[:, 6]
            dHdyd = new_unscaled[:, 7]
            Hdotd = new_unscaled[:, 8]
            Pdotd = new_unscaled[:, 9]

            Z = np.zeros_like(Hd)
            # Build a 12-list
            xi_d = [
                Hd,  # row 0
                Pd,  # row 1
                Ud,  # row 2
                Vd,  # row 3 => ignoring lmby
                Z,  # row 4 => ignoring lmbz
                dxd,  # row 5 => dPdx
                dyd,  # row 6 => dPdy
                Z,  # row 7 => ignoring gradPz
                dHdxd,  # row 8 => dHdx
                dHdyd,  # row 9 => dHdy
                Z,  # row 10 => ignoring dHdz
                Hdotd,  # row 11 => Hdot
                Pdotd,  # row 12 => Pdot
            ]

        # # 6) Build tasks from all newly accepted points
        tasks = build_task_list_transient(xi_d)
        return tasks, xi_d

    # def build(self, xi, order, init, theta=None):
    #     """
    #     Version without downsampling (1:1)
    #     """

    #     xi_d = xi
    #     tasks = build_task_list_transient(xi_d)
    #     return tasks, xi_d

    # ---------------------------------
    # other methods: update_results(), load_results(), etc.
    # ---------------------------------

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
        """
        Return the entire cumulative set of chosen points as shape (#points, 6).
        For example, columns: [H, P, U, dPdx, dPdy, f].
        """
        if self.existing_xi_d is None:
            return np.zeros((0, 6))

        ex = self.existing_xi_d  # shape (10, n_points)
        # Extract the 10 coverage variables
        Hvals = ex[0, :]
        Pvals = ex[1, :]
        Uvals = ex[2, :]
        Vvals = ex[3, :]
        dPdxv = ex[5, :]
        dPdyv = ex[6, :]
        dHdxv = ex[8, :]
        dHdyv = ex[9, :]
        Hdotvals = ex[11, :]
        Pdotvals = ex[12, :]
        return np.column_stack(
            [
                Hvals,
                Pvals,
                Uvals,
                Vvals,
                dPdxv,
                dPdyv,
                dHdxv,
                dHdyv,
                Hdotvals,
                Pdotvals,
            ]
        )


def weighted_distance(a, b, weight_f=1.0):
    # a, b: shape (D,) or (N, D)
    # We assume the last column is F
    diffs = a - b
    # Scale the final dimension
    diffs[..., -1] *= weight_f
    return np.linalg.norm(diffs, axis=-1)
