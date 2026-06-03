# VBAT Compute-Hardware Workbench — Cleaned 0305 Core

This package is a cleaned, function-preserving version of the 0305 project core. It keeps the active pipeline and removes old/debug-only scripts, compiled caches, experimental OCP variants, and stale analysis artifacts.

## Active pipeline

1. `run_mpc_replay.py` — closed-loop MPC replay/evaluation layer.
2. `run_hardware_search_checkpointed.py` — checkpointed Main4-style hardware search using recursive hu/hf contraction and final subset/permutation search.
3. `run_hardware_search_full.py` — same search in a single full pass.
4. `run_ocp_path_pool_refine.py` — Main5-style path-pool OCP refinement for selected hardware-search candidates.

## Typical commands

```bash
python run_hardware_search_checkpointed.py       --stage all       --output-dir outputs_hardware_search_15hz_s20_s38       --num-scenarios 40       --seed 42       --scenario-ids 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38       --num-workers 6       --anchor-freq-hz 15       --freq-list 15       --final-entry-mode bisection       --final-grid-step 0.001       --reclaim-grid-step 0.001       --min-metric-rel-change 0.001
```

```bash
python run_ocp_path_pool_refine.py       --main4-output-dir outputs_hardware_search_15hz_s20_s38       --output-dir outputs_ocp_path_pool_15hz_s20_s38_perm24       --num-scenarios 40       --num-workers 6       --seed 42       --candidate-tag closest       --freq-list 15       --ocp-floor-source zero       --score-weights tw=1,tdot=1,delta=1,delta_dot=1       --max-permutations 24       --ocp-bisect-iters 6       --ocp-terminal-pool-nodes 1       --ocp-terminal-z-tol-m 0.10       --ocp-terminal-vx-rel-tol 1.0       --scenario-ids 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
```

## Notes

- The old standalone OCP refinement and old OCP sizing scripts are not included.
- `core/hardware.py` now owns the shared `HardwarePoint` dataclass.
- `utils/hardware_search_helpers.py` replaces the old full `main4_utopia_subset_mpc.py` file with only the helper functions needed by the current Main4 search.
- No final analysis/ranking code is included in this cleanup package.
