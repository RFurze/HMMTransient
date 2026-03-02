#!/bin/bash

# Function to read configuration values from CONFIG.py so that both shell scripts
# and the Python pipeline use the same parameters.
read_configPenalty() {
    OUTPUT_DIR=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.runtime.OUTPUT_DIR)
PY
    )

    MAX_LB_ITERS=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.runtime.MAX_LB_ITERS)
PY
    )

    MAX_COUPLING_ITERS=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.runtime.MAX_COUPLING_ITERS)
PY
    )

    COUPLING_TOL=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.STEADY_COUPLING_TOL)
PY
    )

    LOAD_BALANCE_TOL=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.STEADY_LOAD_BALANCE_TOL)
PY
    )

    D_COUPLING_TOL=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.transient.coupling_tol)
PY
    )

    D_LOAD_BALANCE_TOL=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.transient.load_balance_tol)
PY
    )

    TEND=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.runtime.TEND)
PY
    )

    DT=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.runtime.DT)
PY
    )

    T=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.runtime.T0)
PY
    )

    # EDAS parameters for the transient refinement loop
    EDAS_MAX_REFINE=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.edas.max_refine_passes)
PY
    )

    EDAS_ERROR_TARGET=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.edas.error_target)
PY
    )

    EDAS_MAX_BUDGET=$(python3 - <<'PY'
import CONFIGPenalty as CONFIG
print(CONFIG.edas.max_budget)
PY
    )
}
