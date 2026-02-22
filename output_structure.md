# Case structure

We separate **run state** (files needed to resume/monitor a run) from **results artifacts** (PVD/XDMF/NPY and post-processing outputs), and keeps everything under one case-root directory.

## Target structure

Use one root for each case, for example:

- `/data/output/cases/<case_id>/`

Where `<case_id>` is unique and sortable, e.g.:

- `20260222_1535_penaltydebug9`
- `case_0142`

Within each case:

```text
/data/output/cases/<case_id>/
├── run/
│   ├── config_snapshot/
│   │   ├── CONFIGPenalty.py
│   │   └── runtime_env.txt
│   ├── logs/
│   │   ├── run.log
│   │   └── progress.log
│   ├── checkpoint/
│   │   └── restart_state
│   └── exchange/
│       ├── p_init.npy
│       ├── xi_rot_prev.npy
│       ├── coupling_error.txt
│       ├── load_balance_err.txt
│       ├── d_coupling_errs.txt
│       └── d_load_balance_err.txt
├── results/
│   ├── steady/
│   │   ├── macroscale/
│   │   └── microscale/
│   ├── transient/
│   │   ├── t_0000p00/
│   │   ├── t_0000p05/
│   │   └── ...
│   ├── viz/
│   │   ├── pvd/
│   │   ├── xdmf/
│   │   └── vtu/
│   └── derived/
│       ├── forces.txt
│       ├── friction.txt
│       └── summary.csv
└── meta/
    ├── case_manifest.yaml
    └── tags.txt
```

## Practical rules

1. **Single case root**: all files for one run stay in one tree.
2. **Run-state vs results split**:
   - `run/` = restart + orchestration I/O
   - `results/` = simulation outputs + post-processing
3. **Immutable result artifacts**:
   - avoid overwriting old time-step files
   - append/write new files by timestep or iteration
4. **Config snapshot per run**:
   - copy `CONFIGPenalty.py` (and environment summary) into `run/config_snapshot/`
5. **Optional convenience symlink** in repo root:
   - `latest_case -> /data/output/cases/<case_id>`

## Retention policy (recommended)

- Keep `run/` and `meta/` always.
- Keep full `results/` for important runs.
- For routine runs, prune heavy `results/viz/vtu/` but retain `results/viz/pvd`, `results/viz/xdmf`, and `results/derived`.

## Migration plan

1. Add a `CASE_ROOT` variable (derived once) and use it everywhere.
2. Set:
   - `OUTPUT_DIR="$CASE_ROOT/run/exchange"` for run-time exchange files.
   - writer paths for visualization/results under `"$CASE_ROOT/results/..."`.
3. Keep backward compatibility for one transition period by mirroring key files or maintaining a symlink.
4. Add one small cleanup/archive script to:
   - compress old cases
   - prune transient VTU files by policy.

## Minimal first step (low risk)

Without changing solver internals yet:

- Define `CASE_ROOT=/data/output/cases/<case_id>` in launcher scripts.
- Point existing `OUTPUT_DIR` at `"$CASE_ROOT/run/exchange"`.
- Move `run.log`, `progress.log`, and `restart_state` into `"$CASE_ROOT/run/..."`.
- Keep current solver outputs, but route any explicit output paths into `"$CASE_ROOT/results"` where possible.

This gives immediate directory hygiene while minimizing code churn.

## Implemented wiring in this repository

A first implementation can be done without touching all solver internals at once:

1. In `RUN_PENALTY.sh`, derive and create:
   - `CASE_ROOT=/data/output/cases/<case_id>`
   - `RUN_ROOT=$CASE_ROOT/run`
   - `EXCHANGE_DIR=$RUN_ROOT/exchange`
   - `LOG_DIR=$RUN_ROOT/logs`
   - `CHECKPOINT_DIR=$RUN_ROOT/checkpoint`
   - `RESULTS_DIR=$CASE_ROOT/results`
2. Keep passing `--output_dir "$EXCHANGE_DIR"` to existing Python scripts.
3. Export environment variables for solvers:
   - `HMM_CASE_ID`, `HMM_CASE_ROOT`, `HMM_RESULTS_DIR`
4. In Python, infer the case id from `--output_dir` and send it to the macroscale writer layer.
5. In the macroscale writer layer, resolve all `data/output/...` writes through `HMM_RESULTS_DIR`.

This keeps exchange/restart files in `run/` while moving result files under `results/` with minimal coupling-risk.