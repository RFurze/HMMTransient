#!/bin/bash

# ---------------------------------------------------------------------------
# Function to read configuration values from ``CONFIG.py`` so that the shell
# script and the Python pipeline share the same parameters.
source "$(dirname "${BASH_SOURCE[0]}")/read_config_penalty.sh"
read_configPenalty

# ---------------------------------------------------------------------------
# Case-oriented layout
# ---------------------------------------------------------------------------
CASE_ID="$OUTPUT_DIR"
CASES_BASE_DIR="${HMM_CASES_BASE_DIR:-/data/output/cases}"
CASE_ROOT="${CASES_BASE_DIR}/${CASE_ID}"
RUN_ROOT="${CASE_ROOT}/run"
EXCHANGE_DIR="${RUN_ROOT}/exchange"
LOG_DIR="${RUN_ROOT}/logs"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoint"
RESULTS_DIR="${CASE_ROOT}/results"

mkdir -p "$EXCHANGE_DIR" "$LOG_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

# Preserve variable name used by all downstream scripts.
OUTPUT_DIR="$EXCHANGE_DIR"
export HMM_CASE_ID="$CASE_ID"
export HMM_CASE_ROOT="$CASE_ROOT"
export HMM_RESULTS_DIR="$RESULTS_DIR"

# Convenience symlink for quick access to the latest run.
ln -sfn "$CASE_ROOT" latest_case

LOGFILE="$LOG_DIR/run.log"

# Save the original fds 1 & 2 so that we can still write to the terminal if we want to change the output later:
exec 3>&1 4>&2

# Redirect everything from stdout+stderr into `tee`, which writes to both terminal (via fd3/4) and appends to the logfile
exec > >(tee -a "$LOGFILE" >&3) 2> >(tee -a "$LOGFILE" >&4)

# Helper to run a command and abort on failure
run_step() {
    local desc="$1"
    shift
    { "$@"; } >>"$LOGFILE" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "Error: ${desc} failed with exit code $rc. Aborting."
        exit $rc
    fi
}

# Checkpoint helpers
save_checkpoint() {
    local phase="$1"
    echo "$phase $lb_iter $c_iter $T" > "$CHECKPOINT_DIR/restart_state"
}

load_checkpoint() {
    if [ -f "$CHECKPOINT_DIR/restart_state" ]; then
        read checkpoint_phase lb_iter c_iter T < "$CHECKPOINT_DIR/restart_state"
        echo "Restarting from checkpoint: phase=$checkpoint_phase lb_iter=$lb_iter c_iter=$c_iter T=$T"
    else
        checkpoint_phase=""
    fi
}

# Append progress information
log_progress() {
    echo "$1 $T $lb_iter $c_iter" >> "$LOG_DIR/progress.log"
}

load_checkpoint



#--------------------------------------------------------#
# STEP 0: Initialize Macroscale (serial)
#--------------------------------------------------------#
if [ -z "$checkpoint_phase" ]; then
    echo "[STEP 0.0] Macroscale initialization (serial)"
    run_step "run_steady_macroscale_init.py" python3 run_steady_macroscale_init_penalty.py \
        --lb_iter 0 \
        --output_dir "$OUTPUT_DIR"
fi


#--------------------------------------------------------#
# Outer loop: Load-balance iterations
#--------------------------------------------------------#
if [ -z "$checkpoint_phase" ] || [ "$checkpoint_phase" = "STEADY" ]; then
    if [ "$checkpoint_phase" = "STEADY" ]; then
        start_lb_iter=$lb_iter
    else
        start_lb_iter=1
    fi
    lb_iter=$start_lb_iter

    while [ $lb_iter -le $MAX_LB_ITERS ]; do
    echo "=========================================="
    echo "Load-balance iteration ${lb_iter}"
    echo "=========================================="

    # Reset coupling iteration counter for each load-balance iteration
    if [ "$checkpoint_phase" = "STEADY" ] && [ $lb_iter -eq $start_lb_iter ]; then
        c_iter_start=$c_iter
        checkpoint_phase=""
    else
        c_iter_start=1
    fi
    c_iter=$c_iter_start

    while [ $c_iter -le $MAX_COUPLING_ITERS ]; do
        echo "  -----------------------------"
        echo "   Coupling iteration ${c_iter}"
        echo "  -----------------------------"
        save_checkpoint STEADY
        log_progress STEADY
        
        #--------------------------------------------------------#
        # STEP 1: Downsample and build microscale task list (serial)
        #--------------------------------------------------------#
        echo "[STEP 0.1] Downsample and build microscale task list (serial)"
        run_step "prepare_microscale_tasks.py" python3 prepare_microscale_tasks.py \
            --lb_iter ${lb_iter} \
            --c_iter ${c_iter} \
            --output_dir "$OUTPUT_DIR"

        #--------------------------------------------------------#
        # STEP 0.2: Microscale simulations (parallel)
        # --oversubscribe is used on openmpi (like uni system)
        #--------------------------------------------------------#
        echo "   [STEP 0.2] Run microscale sims (parallel)"
        # run_step "run_microscale.py" mpiexec -np 12 python3 -m mpi4py.futures run_microscale.py \
        run_step "run_microscale.py" python3 run_microscale.py \
            --lb_iter ${lb_iter} \
            --c_iter ${c_iter} \
            --output_dir "$OUTPUT_DIR"

        # #--------------------------------------------------------#
        # # STEP 0.3: Update metamodel (serial)
        # #--------------------------------------------------------#
        # echo "   [STEP 0.3] Update metamodel (serial)"
        # run_step "generate_MLS_tasks.py" python3 generate_MLS_tasks.py \
        #     --lb_iter ${lb_iter} \
        #     --c_iter ${c_iter} \
        #     --output_dir "$OUTPUT_DIR"

        # #--------------------------------------------------------#
        # # STEP 0.4: Run MLS evaluations (parallel)
        # #--------------------------------------------------------#
        # echo "   [STEP 0.4] Run MLS evaluations (parallel)"
        # run_step "run_MLS.py" mpiexec -np 12 python3 -m mpi4py.futures run_MLS.py \
        #     --lb_iter ${lb_iter} \
        #     --c_iter ${c_iter} \
        #     --output_dir "$OUTPUT_DIR"

        #--------------------------------------------------------#
        # STEP 0.5: Load micro predictions into macroscale (serial)
        #--------------------------------------------------------#
        echo "   [STEP 0.5] Macroscale solve w/ micro corrections (serial)"
        run_step "run_steady_macroscale_HMM.py" python3 run_steady_macroscale_HMM_penalty.py \
            --lb_iter ${lb_iter} \
            --c_iter ${c_iter} \
            --output_dir "$OUTPUT_DIR"

        log_progress STEADY

        #--------------------------------------------------------#
        # Check coupling convergence using Python for float comparison
        #--------------------------------------------------------#
        c_err=$(tail -n 1 "${OUTPUT_DIR}/coupling_error.txt")
        if [ -z "$c_err" ]; then
            echo "Warning: No coupling_error.txt found or it's empty; continuing anyway."
        else
            # Use Python inline to compare float values
            if python3 -c "import sys; sys.exit(0) if abs(float('$c_err')) < float('$COUPLING_TOL') else sys.exit(1)"; then
                echo "   Coupling converged after iteration ${c_iter} (error=$c_err)."
                break  # Exit the coupling loop; c_iter will be reset in the next lb_iter
            fi
        fi

        # Increment coupling iteration counter
        c_iter=$((c_iter + 1))
    done  # End inner (coupling) loop

    #--------------------------------------------------------#
    # Check load-balance convergence using Python for float comparison
    #--------------------------------------------------------#
    lb_err=$(tail -n 1 "${OUTPUT_DIR}/load_balance_err.txt")
    if [ -z "$lb_err" ]; then
        echo "Warning: No load_balance_err.txt found or it's empty; continuing anyway."
    else
        echo "   Steady load-balance check: error=${lb_err}, tolerance=${LOAD_BALANCE_TOL}"
        if python3 -c "import sys; sys.exit(0) if abs(float('$lb_err')) < float('$LOAD_BALANCE_TOL') else sys.exit(1)"; then
            echo "Load balance converged after iteration ${lb_iter} (error=$lb_err)."
            break  # Exit the load-balance loop and conclude the script
        else
            echo "   Load balance NOT converged after iteration ${lb_iter} (error=$lb_err, tolerance=$LOAD_BALANCE_TOL)."
        fi
    fi

    # Increment load-balance iteration counter
    lb_iter=$((lb_iter + 1))
    done  # End outer (load-balance) loop

    echo "Steady simulation completed - Initialised for transient run."
else
    if [ "$checkpoint_phase" = "TRANSIENT" ]; then
        echo "Skipping steady stage"
    fi
fi

#------------------------------------------------
#Transient simulation loop
#------------------------------------------------
if [ -z "$checkpoint_phase" ] || [ "$checkpoint_phase" != "TRANSIENT" ]; then
    if awk -v t="$T" 'BEGIN{exit (t == 0.0 ? 0 : 1)}'; then
        echo "Transient start time is 0.0; advancing to first time step at T=DT."
        T=$(awk -v dt="$DT" 'BEGIN{printf "%.8f", dt}')
    fi
fi
while (( $(awk -v t="$T" -v tend="$TEND" 'BEGIN{print (t<=tend)}') )); do
    echo "===================================="
    echo "Transient Step = ${T}"
    echo "===================================="

    # LOGIC TO CHANGE DT BASED ON TIME - CURRENTLY SET TO 0.1 FOR ALL TIMES
    if (( $(awk -v t="$T" 'BEGIN{print (t < 20)}') )); then
        DT=0.05
    else
        DT=0.05
    fi

    if [ "$checkpoint_phase" = "TRANSIENT" ]; then
        start_lb_iter=$lb_iter
        cp_c_iter=$c_iter
        checkpoint_phase_handled=true
        checkpoint_phase=""
    else
        start_lb_iter=0
        checkpoint_phase_handled=false
    fi
    lb_iter=$start_lb_iter

    while [ $lb_iter -le $MAX_LB_ITERS ]; do
        echo "  -----------------------------------"
        echo "  Load Balance Iteration = ${lb_iter}"
        echo "  -----------------------------------"

        # Reset coupling iteration counter for each load-balance iteration
        if $checkpoint_phase_handled && [ $lb_iter -eq $start_lb_iter ]; then
            c_iter=$cp_c_iter
            checkpoint_phase_handled=false
        else
            c_iter=1
        fi

        while [ $c_iter -le $MAX_COUPLING_ITERS ]; do
            echo "    ----------------------------------------"
            echo "     Transient Coupling iteration ${c_iter}"
            echo "    ----------------------------------------"
            save_checkpoint TRANSIENT
            log_progress TRANSIENT

            #------------------------------------------------------------#
            # STEP 1.0: Macroscale Transient solve for T (serial)
            #------------------------------------------------------------#
            echo "    [STEP 1.0] Transient macroscale solve (serial)"
            # If MACRO_ONLY, we dont use the HMM corrections - useful for transient initialisation
            if (( $(awk -v t="$T" 'BEGIN{print (t < 0)}') )); then
                MACRO_ONLY=1 run_step "run_transient_macroscale.py" python3 run_transient_macroscale.py \
                    --lb_iter ${lb_iter} \
                    --c_iter ${c_iter} \
                    --Time $T \
                    --DT $DT \
                    --output_dir "$OUTPUT_DIR"
            else
                run_step "run_transient_macroscale.py" python3 run_transient_macroscale.py \
                    --lb_iter ${lb_iter} \
                    --c_iter ${c_iter} \
                    --Time $T \
                    --DT $DT \
                    --output_dir "$OUTPUT_DIR"
            fi

            #--------------------------------------------------------------#
            # STEP 1.05: Check for convergence of load balance and coupling
            #--------------------------------------------------------------#

            c_err=$(tail -n 1 "${OUTPUT_DIR}/d_coupling_errs.txt")
            if [ -z "$c_err" ]; then
                echo "Warning: No d_coupling_errs.txt found or it's empty; continuing anyway."
            else
                if python3 -c "import sys; sys.exit(0) if abs(float('$c_err')) < float('$D_COUPLING_TOL') else sys.exit(1)"; then
                    echo "   Coupling converged after iteration ${c_iter} (error=$c_err)."
                    lb_err=$(tail -n 1 "${OUTPUT_DIR}/d_load_balance_err.txt")
                    if [ -z "$lb_err" ]; then
                        echo "Warning: No d_load_balance_err.txt found or it's empty; continuing anyway."
                    else
                        if python3 -c "import sys; sys.exit(0) if abs(float('$lb_err')) < float('$D_LOAD_BALANCE_TOL') else sys.exit(1)"; then
                            echo "Load balance converged after iteration ${lb_iter} (error=$lb_err)."
                            break 2  # Exit the coupling loop and the load balance loop - T will be incremented and T loop will continue
                        else
                            echo "   Load balance NOT converged after iteration ${lb_iter} (error=$lb_err)."
                            # If load balance has not converged, we break the coupling loop and let lb_iter increment
                            break
                        fi
                    fi
                fi
            fi

            # Increment coupling iteration counter
            echo "   Coupling NOT converged after iteration ${c_iter} (error=$c_err)."

            if (( $(awk -v t="$T" 'BEGIN{print (t >= 0)}') )); then
                #--------------------------------------------------------#
                # STEP 1.1: Downsample and build microscale task list (serial)
                #--------------------------------------------------------#
                echo "[STEP 1.1] Downsample and build microscale task list (serial)"
                run_step "prepare_microscale_tasks.py" python3 prepare_microscale_tasks.py \
                    --transient \
                    --lb_iter ${lb_iter} \
                    --c_iter ${c_iter} \
                    --Time $T \
                    --DT $DT \
                    --output_dir "$OUTPUT_DIR"

                #--------------------------------------------------------#
                # STEP 1.2: Microscale simulations (parallel)
                # --oversubscribe is used on openmpi (like uni system)
                #--------------------------------------------------------#
                echo "   [STEP 1.2] Run microscale sims (parallel)"
                # run_step "run_microscale.py" mpiexec -np 12 python3 -m mpi4py.futures run_microscale.py \
                run_step "run_microscale.py" python3 run_microscale.py \
                    --transient \
                    --lb_iter ${lb_iter} \
                    --c_iter ${c_iter} \
                    --Time $T \
                    --DT $DT \
                    --output_dir "$OUTPUT_DIR"

                #--------------------------------------------------------#
                # STEP 1.3: Update metamodel (serial)
                #--------------------------------------------------------#
                echo "   [STEP 1.3] Update metamodel (serial)"
                run_step "generate_MLS_tasks.py" python3 generate_MLS_tasks.py \
                    --transient \
                    --lb_iter ${lb_iter} \
                    --c_iter ${c_iter} \
                    --Time $T \
                    --DT $DT \
                    --output_dir "$OUTPUT_DIR"

                #--------------------------------------------------------#
                # STEP 1.4: Run MLS evaluations (parallel)
                #--------------------------------------------------------#
                echo "   [STEP 1.4] Run MLS evaluations (parallel)"
                run_step "run_MLS.py" mpiexec -np 12 python3 -m mpi4py.futures run_MLS.py \
                    --transient \
                    --lb_iter ${lb_iter} \
                    --c_iter ${c_iter} \
                    --output_dir "$OUTPUT_DIR"
            fi

            c_iter=$((c_iter + 1))

        done  # End inner (coupling) loop
        
        # Increment load-balance iteration counter
        lb_iter=$((lb_iter + 1))
    done  # End outer (load-balance) loop
    T=$(awk -v t="$T" -v dt="$DT" 'BEGIN{printf "%.8f", t+dt}')
done

#Need a script to combine all the transient vtk files into one pvd

rm -f "$CHECKPOINT_DIR/restart_state"
echo "Transient simulation complete."