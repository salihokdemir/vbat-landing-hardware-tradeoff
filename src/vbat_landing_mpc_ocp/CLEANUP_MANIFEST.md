# Cleanup Manifest

## Kept and renamed

- `main3_final_mpc.py` -> `run_mpc_replay.py`
- `main4_20_04_common.py` -> `hardware_search_common.py`
- `main4_20_04_checkpointed.py` -> `run_hardware_search_checkpointed.py`
- `main4_20_04_full.py` -> `run_hardware_search_full.py`
- `main5_25_04_ocp_path_pool_candidates.py` -> `run_ocp_path_pool_refine.py`
- `ocp/ocp_mpc_path_pool_25_04.py` -> `ocp/path_pool_ocp.py`
- `ocp/ocp_msd4_12_04.py` -> reduced to `core/hardware.py` only; the old OCP class was removed from this package.
- `main4_utopia_subset_mpc.py` -> reduced to `utils/hardware_search_helpers.py` only.
- Date-suffixed config/model/controller/guidance/utils files were renamed to neutral names.

## Removed from package

- `__pycache__/` and all `.pyc` files.
- `candidate_replay_25_04_closest.csv` debug artifact.
- `main5_ocp_refine_candidates.py` and `main5_checkpointed_ocp_refine.py` old non-path-pool refinement flow.
- `main5_25_04_ocp_path_pool_checkpointed.py` because the current project commands use the direct path-pool runner and the checkpoint wrapper contains stale fields from the older Main5 config.
- `main6_25_04_freq_sweep_hfm_ho.py` because the current final analysis is not directly based on Main6.
- `utils/utopia_subset_search_12_04.py` old subset-search implementation no longer needed by the active checkpointed Main4 path.
- `README_25_04_OCP_PATH_POOL.txt` replaced by this package README.

## Functional scope preserved

The preserved scope is the active workflow: closed-loop MPC replay, checkpointed/full hu/hf hardware search, and path-pool OCP refinement for selected Main4 candidates.
