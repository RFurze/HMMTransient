# EDAS Convergence Analysis and Improvement Plan

## 1. Summary of Findings

The EDAS scheme was aborted during its second refinement pass on the first transient time step (T=0.05). The MLS max error at abort was **0.613** against a target of **0.05** — a 12× gap. Coupling convergence showed oscillatory behaviour with three divergence spikes (iterations 7, 8, and 14), though it eventually recovered. Load balance converged well.

The root causes fall into three categories: normaliser instability from changing Hdot/Pdot, an over-constrained sample selector that wastes budget, and an accumulating training set that is far too large for the cKDTree operations in the inner loop.

---

## 2. Root Cause Analysis

### 2.1 The SharedNormaliser and Hdot/Pdot Drift

**This is the most critical issue.**

The xi vector is `[H, P, U, V, dpdx, dpdy, hdot, pdot]` (user description) which maps to the 13-component rotated vector. The EDAS feature space uses 6 dimensions: `[H, P, dPdx, dPdy, Hdot, Pdot]`.

Within a single time step, H, P, U, V, dpdx, dpdy are fixed (from the start of the time step). But **Hdot and Pdot change on every coupling iteration** because they come from the time derivative computed by the macroscale solver after applying corrections. This means:

1. Training data accumulated from coupling iteration 1 has Hdot/Pdot values computed with zero corrections.
2. Coupling iteration 2 has different Hdot/Pdot because the macroscale was re-solved with dP, dQ corrections.
3. The `SharedNormaliser` monotonically widens its bounds to accommodate all of these, so the normalised feature space becomes increasingly diluted.

**Consequence:** The MLS metamodel is being trained on data where two of the six feature dimensions are inconsistent across the training set. Points that were "close" in feature space in iteration 1 may be "far" in iteration 2, or vice versa. This directly undermines the MLS interpolation quality.

**Evidence from logs:** The relevance weights are 0.92–1.0, meaning nearly nothing is pruned despite the data staleness. With lambda_decay=2.0 and T=0.05, the time decay is exp(−2.0 × 0.05) ≈ 0.905 — far too weak to prune iteration-to-iteration drift.

### 2.2 The `select_samples` Performance Bug

In `edas.py` line 342, a `cKDTree` is **rebuilt from the entire training set on every candidate evaluation**:

```python
for idx in order:
    ...
    if delta_min > 0 and X_train_norm is not None and X_train_norm.shape[0] > 0:
        tree = cKDTree(X_train_norm)   # <-- rebuilt on every iteration!
        d, _ = tree.query(candidate.reshape(1, -1), k=1)
```

With 31,000+ training points and up to 9,918 candidates to evaluate, this is O(N_candidates × N_train × log(N_train)) tree constructions. This is where the run was killed (KeyboardInterrupt at exactly this line).

**Fix:** Build the tree once before the loop. This is a pure performance bug, not a tuning issue.

### 2.3 Over-Constrained delta_min Spacing

The `delta_min_quantile=0.1` derives a minimum spacing from the query point distribution. This prevents new samples being placed near existing training data. But when the high-error region is precisely where existing (stale) training data sits — which is the case with Hdot/Pdot drift — the spacing constraint prevents the sampler from placing new, *correct* samples where they are most needed.

**Evidence:** The `select_samples` efficiency analysis shows the selected points are not the highest-error ones — the top-k overlap is well below 1.0, meaning many high-error candidates are being rejected by the spacing constraint.

### 2.4 Training Set Accumulation Without Effective Pruning

After the first load balance iteration's EDAS passes, the training set reached 31,868 points. With relevance weights essentially all above 0.9, the pruning mechanism removes almost nothing. This has two consequences:

1. The cKDTree operations scale with N_train, making each iteration increasingly slow.
2. Old data with stale Hdot/Pdot values dominates the MLS fit, biasing predictions.

### 2.5 MLS Theta Too High for 6D Space

`MLS_THETA` is set to 10,000 for the primary variables (dQx, dQy, dP). In 6 dimensions with normalised features in [0,1], the Gaussian weight `exp(-θ × d²)` falls to `exp(-10000 × d²)`. At distance d=0.01 (1% of range), the weight is exp(−1) ≈ 0.37. At d=0.03, it is exp(−9) ≈ 0.0001. This means MLS is essentially a very local interpolant — it only "sees" training points within about 1% of the normalised range.

For a 6D unit hypercube, covering the space to within 1% requires roughly (1/0.01)^6 ≈ 10^12 points. Even with the batch_size=500 and max_budget=6000, we are sampling a negligible fraction of the space. The MLS is therefore heavily dependent on having training data very close to each query point.

### 2.6 Composite Error Indicator Weighting

The error indicator in `run_MLS.py` uses a 40% coverage + 40% LOOCV + 20% fallback blend. Because the LOOCV component requires accumulated training outputs (transient_existing_dp.npy, transient_existing_dq.npy), and these accumulate stale data, the LOOCV errors may be misleading — they reflect the fit quality to a mixed dataset rather than the current time derivative regime.

---

## 3. Improvement Plan

### Phase 1: Performance Fix (Immediate)

**3.1 Pre-build cKDTree in `select_samples`**

Build the tree once before the candidate loop. This is a one-line change that eliminates the O(N²) tree construction.

```python
def select_samples(epsilon, X_query_norm, X_train_norm, batch_size, delta_min=0.0):
    order = np.argsort(-epsilon)
    selected = []
    selected_norm = []

    # Build tree ONCE
    train_tree = None
    if delta_min > 0 and X_train_norm is not None and X_train_norm.shape[0] > 0:
        train_tree = cKDTree(X_train_norm)

    for idx in order:
        if len(selected) >= batch_size:
            break
        candidate = X_query_norm[idx]

        if train_tree is not None:
            d, _ = train_tree.query(candidate.reshape(1, -1), k=1)
            if d[0] < delta_min:
                continue
        # ... rest unchanged
```

**Impact:** Reduces selection from minutes/hours to seconds.

### Phase 2: Normaliser and Training Data Stability

**3.2 Separate normaliser bounds for static vs dynamic features**

Partition the 6 features into:
- **Static features** (per time step): H, P, dPdx, dPdy — these don't change within a time step's coupling iterations
- **Dynamic features**: Hdot, Pdot — these change every coupling iteration

For the static features, use fixed normaliser bounds (set once at the start of the time step). For the dynamic features, use a **per-coupling-iteration** normaliser that resets each iteration, or use the current iteration's range only.

This prevents stale Hdot/Pdot from diluting the normalised space.

**3.3 Coupling-iteration-aware training data management**

Instead of accumulating all training data indefinitely, implement a two-tier strategy:

- **Tier 1 (current coupling iteration):** Full weight. These points have correct Hdot/Pdot for the current macroscale state.
- **Tier 2 (previous coupling iterations, same time step):** Reduced weight. The H,P,dPdx,dPdy are still correct, but Hdot/Pdot are stale. Apply a coupling-iteration decay factor (e.g., 0.5 per iteration) on top of the time decay.
- **Tier 3 (previous time steps):** Time-decayed as currently implemented, but with a stronger lambda_decay.

**3.4 Increase lambda_decay for meaningful pruning**

With DT=0.05 and lambda_decay=2.0, the per-step decay is only 9.5%. At lambda_decay=10.0 it would be 39%, and at lambda_decay=20.0 it would be 63%. Given that the physics changes significantly between time steps (sinusoidal load), a higher decay rate is justified.

**Recommended sweep values:** lambda_decay ∈ {5.0, 10.0, 15.0, 20.0}

Run the parameter sensitivity tool (`diagnostics/edas_parameter_sensitivity.py`) to determine the best value for your case.

### Phase 3: Sampling Strategy Refinements

**3.5 Reduce or adapt delta_min_quantile**

The spacing constraint should be relaxed when targeting regions near existing (stale) training data. Options:

- Reduce `delta_min_quantile` from 0.1 to 0.02 or lower.
- Apply the spacing constraint only against training data from the *current* coupling iteration, not against all historical data.
- Remove the spacing constraint against training data entirely and only keep the inter-batch spacing (preventing duplicate selections within a single batch).

**3.6 Reduce batch_size, increase max_refine_passes**

Instead of 500 samples × 4 passes (2000 max), consider 200 samples × 10 passes (2000 max). Smaller batches with more feedback loops allow the error indicator to guide the sampler more precisely, rather than making a large semi-blind batch selection.

**3.7 Consider dimensionality reduction for MLS**

With only 6 features and degree-2 polynomials, the polynomial basis has C(6+2, 2) = 28 terms. But if Hdot and Pdot are highly correlated (both are time derivatives driven by the same solver), a PCA or feature-selection step could reduce the effective dimensionality.

### Phase 4: MLS Robustness

**3.8 Reduce MLS_THETA for primary variables**

The current θ=10,000 creates an extremely localised interpolant. With 31k training points in 6D, a softer kernel (θ=1000–3000) would:
- Allow each query point to "see" more training data
- Reduce sensitivity to local gaps in coverage
- Decrease the fallback rate

**Recommended sweep:** θ ∈ {1000, 2000, 3000, 5000, 10000} using cross-validation on the existing training data.

**3.9 Weight MLS training data by relevance**

The relevance weights computed by EDAS are currently saved but not used in the MLS solve. Pass them through to the MLS worker so that stale training points contribute less to the local polynomial fit.

This requires modifying `generate_MLS_tasks.py` and `run_MLS.py` to incorporate `edas_relevance_weights.npy` as a multiplicative factor in the Gaussian weighting.

### Phase 5: Coupling Stability

**3.10 Aitken diagnostics**

The Aitken relaxation currently uses a single omega derived from dP. Add logging of omega_k, ||delta_k||, and the Aitken denominator to the live monitoring hook. This will reveal whether the divergence spikes at iterations 7, 8, and 14 correspond to omega overshoots.

**3.11 Freeze EDAS during initial coupling iterations**

For the first 2–3 coupling iterations of a new time step, use the existing MLS metamodel *without* EDAS refinement. This allows the coupling to settle before investing budget in refinement. Only trigger EDAS when coupling has partially converged (e.g., coupling_error < 0.01).

This avoids training the MLS on corrections from a macroscale state that is still far from converged.

---

## 4. Monitoring Strategy

### 4.1 Live Monitoring (per EDAS pass)

Insert the `live_monitor_hook.sh` into RUN_PENALTY.sh after each EDAS refinement pass (Step 1.4). This logs:

| Metric | File | What it tells you |
|--------|------|-------------------|
| `mls_max_error` | `mls_max_error.txt` | Whether EDAS refinement is reducing the max indicator |
| `coupling_error` | `d_coupling_errs.txt` | Whether the coupling iteration is converging or oscillating |
| `n_training` | EDAS state | Whether the training set is growing uncontrollably |
| `n_fallback_dP` | `dP_mls_flag.npy` | Whether MLS is degrading (more fallbacks = worse fit) |
| Aitken omega | `coupling_omega.npy` | Whether relaxation is oscillating |

Usage — add these two lines to `RUN_PENALTY.sh` after the `run_MLS.py` step:

```bash
source diagnostics/live_monitor_hook.sh
edas_monitor "$OUTPUT_DIR" "$CASE_ROOT" "$T" "$lb_iter" "$c_iter" "$refine_iter"
```

### 4.2 Post-Run Dashboard

Run after a completed or aborted simulation:

```bash
python diagnostics/convergence_dashboard.py --case BaseCase020326
```

This produces:
- `convergence_summary.json` — machine-readable full analysis
- `convergence_dashboard.png` — 6-panel visual dashboard
- Terminal report with key findings

### 4.3 Parameter Tuning

Run on existing data to evaluate parameter changes without re-running:

```bash
python diagnostics/edas_parameter_sensitivity.py --case BaseCase020326 --current_time 0.05
```

This sweeps alpha_blend, delta_min_quantile, lambda_decay, sigma_spatial, and performs a feature importance ablation study. Use the output to choose parameters for the next run.

### 4.4 History Plotting

After accumulating live monitoring data across multiple iterations:

```bash
python diagnostics/plot_live_history.py --case BaseCase020326
```

---

## 5. Recommended Parameter Changes for Next Run

Based on the analysis of BaseCase020326, here are the suggested starting parameters:

| Parameter | Current | Suggested | Rationale |
|-----------|---------|-----------|-----------|
| `lambda_decay` | 2.0 | 10.0 | Current value barely prunes anything; stale data dominates |
| `delta_min_quantile` | 0.1 | 0.02 | Over-constraining prevents sampling in high-error regions |
| `batch_size` | 500 | 200 | Smaller batches with more passes for better error-guided placement |
| `max_refine_passes` | 4 | 10 | More passes compensate for smaller batch size |
| `MLS_THETA[0:3]` | 10000 | 3000 | Softer kernel to see more training data per query point |
| `sigma_spatial` | 0.3 | 0.5 | Wider spatial relevance to keep more data alive in sparse regions |
| `r0_quantile` | 0.25 | 0.15 | Tighter initial coverage for denser cold-start sampling |

These should be validated with the parameter sensitivity tool before committing to a full run.

---

## 6. Implementation Priority

1. **Performance fix** (§3.1): Pre-build cKDTree. This is blocking — runs cannot complete without it.
2. **Live monitoring** (§4.1): Insert hook into shell script. Zero risk, high information value.
3. **Lambda decay increase** (§3.4): Single parameter change, large expected impact.
4. **delta_min relaxation** (§3.5): Single parameter change, moderate expected impact.
5. **Normaliser partitioning** (§3.2): Moderate code change, addresses the core Hdot/Pdot drift issue.
6. **Training data tiering** (§3.3): Requires new logic in `ErrorDrivenSampler`, high expected impact.
7. **MLS theta reduction** (§3.8): Single parameter change, needs cross-validation first.
8. **Relevance-weighted MLS** (§3.9): Moderate code change, leverages existing infrastructure.
9. **Coupling-EDAS phasing** (§3.11): Moderate shell script change, reduces wasted budget.

---

## 7. Files Created

| File | Purpose |
|------|---------|
| `diagnostics/convergence_dashboard.py` | Post-run analysis: JSON summary, terminal report, PNG dashboard |
| `diagnostics/edas_parameter_sensitivity.py` | Offline parameter sweeps on existing EDAS state |
| `diagnostics/live_monitor_hook.sh` | Drop-in shell hook for per-iteration metric logging |
| `diagnostics/plot_live_history.py` | Time-series plots from live monitoring data |
| `diagnostics/EDAS_CONVERGENCE_PLAN.md` | This document |
