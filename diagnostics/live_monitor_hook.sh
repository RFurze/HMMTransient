#!/bin/bash
# -----------------------------------------------------------------------
# live_monitor_hook.sh — Drop-in hook for RUN_PENALTY.sh
#
# Call this after each EDAS refinement pass (Step 1.4) to accumulate
# per-iteration diagnostics without modifying any existing code.
#
# Usage (insert into RUN_PENALTY.sh after the MLS evaluation step):
#
#   source diagnostics/live_monitor_hook.sh
#   edas_monitor "$OUTPUT_DIR" "$CASE_ROOT" "$T" "$lb_iter" "$c_iter" "$refine_iter"
#
# This writes a single JSONL line to:
#   <case>/diagnostics/live_history.jsonl
#
# and a summary CSV row to:
#   <case>/diagnostics/live_summary.csv
# -----------------------------------------------------------------------

edas_monitor() {
    local exchange_dir="$1"
    local case_root="$2"
    local T="$3"
    local lb_iter="$4"
    local c_iter="$5"
    local refine_iter="$6"

    local diag_dir="${case_root}/diagnostics"
    mkdir -p "$diag_dir"

    # Read key scalars
    local mls_max_err=""
    if [ -f "${exchange_dir}/mls_max_error.txt" ]; then
        mls_max_err=$(head -n 1 "${exchange_dir}/mls_max_error.txt")
    fi

    local coupling_err=""
    if [ -f "${exchange_dir}/d_coupling_errs.txt" ]; then
        coupling_err=$(tail -n 1 "${exchange_dir}/d_coupling_errs.txt")
    fi

    local lb_err=""
    if [ -f "${exchange_dir}/d_load_balance_err.txt" ]; then
        lb_err=$(tail -n 1 "${exchange_dir}/d_load_balance_err.txt")
    fi

    local n_tasks=""
    if [ -f "${exchange_dir}/task_count.txt" ]; then
        n_tasks=$(tail -n 1 "${exchange_dir}/task_count.txt")
    fi

    # Training set size from EDAS state (quick Python one-liner)
    local n_training=""
    n_training=$(python3 -c "
import numpy as np, os, sys
p = os.path.join('${exchange_dir}', 'edas_state.npy')
if not os.path.exists(p): sys.exit(0)
s = np.load(p, allow_pickle=True).item()
e = s.get('existing_xi_d')
print(e.shape[1] if e is not None else 0)
" 2>/dev/null)

    # MLS fallback counts
    local n_fallback_dP=""
    n_fallback_dP=$(python3 -c "
import numpy as np, os, sys
p = os.path.join('${exchange_dir}', 'dP_mls_flag.npy')
if not os.path.exists(p): sys.exit(0)
f = np.load(p)
print(int(f.sum()))
" 2>/dev/null)

    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Append JSONL record
    python3 -c "
import json, sys
rec = {
    'timestamp': '${ts}',
    'T': float('${T}'),
    'lb_iter': int('${lb_iter}'),
    'c_iter': int('${c_iter}'),
    'refine_iter': int('${refine_iter}'),
    'mls_max_error': float('${mls_max_err}') if '${mls_max_err}' else None,
    'coupling_error': float('${coupling_err}') if '${coupling_err}' else None,
    'lb_error': float('${lb_err}') if '${lb_err}' else None,
    'n_new_tasks': int('${n_tasks}') if '${n_tasks}' else None,
    'n_training': int('${n_training}') if '${n_training}' else None,
    'n_fallback_dP': int('${n_fallback_dP}') if '${n_fallback_dP}' else None,
}
with open('${diag_dir}/live_history.jsonl', 'a') as f:
    f.write(json.dumps(rec) + '\n')
" 2>/dev/null

    # Append CSV row (create header if new file)
    local csv_file="${diag_dir}/live_summary.csv"
    if [ ! -f "$csv_file" ]; then
        echo "timestamp,T,lb_iter,c_iter,refine_iter,mls_max_error,coupling_error,lb_error,n_new_tasks,n_training,n_fallback_dP" > "$csv_file"
    fi
    echo "${ts},${T},${lb_iter},${c_iter},${refine_iter},${mls_max_err},${coupling_err},${lb_err},${n_tasks},${n_training},${n_fallback_dP}" >> "$csv_file"

    echo "      [MONITOR] T=${T} c=${c_iter} r=${refine_iter} mls_max=${mls_max_err} coupling=${coupling_err} train=${n_training}"
}
