#!/bin/bash
set -euo pipefail

# Reset a case so RUN_PENALTY.sh resumes from the start of the transient stage.
# Usage:
#   ./reset_case_to_transient_start.sh [case_id_or_case_path] [t0]
#
# Arguments:
#   case_id_or_case_path  Optional. Case id under $HMM_CASES_BASE_DIR or an absolute/relative
#                         case directory path. Defaults to ./latest_case.
#   t0                    Optional transient start time written to restart_state. Defaults to 0.0.

CASE_REF="${1:-latest_case}"
T0="${2:-${HMM_T0:-0.05}}"
CASES_BASE_DIR="${HMM_CASES_BASE_DIR:-/data/output/cases}"

if [[ "$CASES_BASE_DIR" == /data/* ]] && ! mountpoint -q /data; then
  CASES_BASE_DIR="$(pwd)/data/output/cases"
  echo "WARNING: /data not mounted; using $CASES_BASE_DIR"
fi

if [[ -d "$CASE_REF" ]]; then
  CASE_ROOT="$(cd "$CASE_REF" && pwd)"
else
  CASE_ROOT="${CASES_BASE_DIR}/${CASE_REF}"
fi

RUN_ROOT="${CASE_ROOT}/run"
EXCHANGE_DIR="${RUN_ROOT}/exchange"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoint"

if [[ ! -d "$EXCHANGE_DIR" ]]; then
  echo "Error: exchange directory not found: $EXCHANGE_DIR"
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

# Remove transient-only rolling state so the next run starts transient fresh.
TRANSIENT_STATE_FILES=(
  "d_coupling_errs.txt"
  "d_load_balance_err.txt"
  "d_eccentricity.txt"
  "d_eccentricity_out.txt"
  "d_friction.txt"
  "d_friction_macro.txt"
  "lb_eccentricities.txt"
  "p0.npy"
  # EDAS sampler / normaliser persistent state
  "edas_state.npy"
  "edas_normaliser_state.npy"
  "edas_timestamps.npy"
  "edas_selected_indices.npy"
  "edas_relevance_weights.npy"
  # EDAS error indicators written by run_MLS.py
  "mls_error_indicators.npy"
  "mls_max_error.txt"
  # Task-count file written by prepare_microscale_tasks.py
  "task_count.txt"
  # Accumulated EDAS training data (transient prefix)
  "transient_existing_xi_d.npy"
  "transient_existing_dp.npy"
  "transient_existing_dq.npy"
  "transient_existing_hmin.npy"
  "transient_existing_hmax.npy"
  "transient_existing_pmax.npy"
  "transient_existing_pmin.npy"
  "transient_existing_taust.npy"
)

for name in "${TRANSIENT_STATE_FILES[@]}"; do
  rm -f "$EXCHANGE_DIR/$name"
done

# Remove transient timestep snapshots (keep T_0_* as transient initial state).
find "$EXCHANGE_DIR" -maxdepth 1 -type f \
  -regextype posix-extended -regex '.*/T_[1-9][0-9]*_(p|h|def|xi)\.npy' \
  -delete

# Force RUN_PENALTY.sh to skip steady and start at transient phase at T0.
printf 'TRANSIENT 0 1 %s\n' "$T0" > "$CHECKPOINT_DIR/restart_state"

echo "Case reset complete."
echo "  case_root:     $CASE_ROOT"
echo "  exchange_dir:  $EXCHANGE_DIR"
echo "  checkpoint:    $CHECKPOINT_DIR/restart_state"
echo "  restart state: TRANSIENT 0 1 $T0"