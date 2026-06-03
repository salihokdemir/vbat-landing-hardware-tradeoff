# Compute-Hardware Trade-off in Autonomous V-BAT Landing

This repository contains the code snapshot, selected processed outputs, and final analysis files for a graduation project on autonomous V-BAT landing under moving-deck and wind-disturbance conditions.

The project studies how MPC update frequency affects the minimum actuator authority required for successful closed-loop landing. The four hardware metrics considered are:

* thrust-to-weight ratio, `T/W`
* thrust-rate limit, `Tdot`
* thrust-vector angle limit, `delta`
* thrust-vector rate limit, `delta_dot`

The goal is not only to demonstrate a landing controller, but to analyze the trade-off between onboard computation and actuator authority.

---

## Repository scope

This repository is a cleaned reproducibility package for the final report and presentation. It includes:

* the active MPC and OCP code used in the project,
* selected Main4/MPC hardware-search outputs,
* selected Main5/path-pool OCP refinement outputs,
* processed final analysis tables and figures,
* scripts and notes used to reproduce the final summary figures.

Large intermediate solver logs, checkpoint files, replay dumps, and old experimental scripts are not included.

---

## Method overview

The project pipeline has three main stages.

### 1. Closed-loop MPC evaluation

`run_mpc_replay.py`

This stage evaluates whether a given hardware vector can complete the landing task in closed loop. A hardware candidate is accepted only if the repeated MPC solves and nonlinear rollout reach the deck-relative touchdown region.

### 2. MPC hardware search

`run_hardware_search_checkpointed.py`

This stage searches for low-authority feasible hardware candidates. It uses recursive `hu/hf` contraction, final subset search, and permutation-based reclaim to identify boundary-region candidates.

`run_hardware_search_full.py` provides the same search logic in a single full run.

### 3. Path-pool OCP refinement

`run_ocp_path_pool_refine.py`

This stage refines selected MPC hardware candidates using the saved MPC state and control history. The OCP is used as a local path-pool refinement tool, not as a replacement for the finite-preview MPC controller.

---

## Final analysis

The final processed analysis compares selected MPC update frequencies using scenario-relative hardware ratios. The primary engineering result is the component-wise hardware burden across:

* `T/W`
* `Tdot`
* `delta`
* `delta_dot`

A weighted-product score is used only as a compact summary after ratio normalization. The score should be interpreted together with the component-wise ratios.

The final result should be read as a simulation-based design-support analysis, not as a certified aircraft hardware sizing result.

---

## Important interpretation note

The results are scenario-dependent and sensitive near the feasibility boundary. The study does not claim a universal best MPC frequency. Instead, it shows that minimum hardware demand does not vary as a simple monotonic function of MPC update frequency.

The main conclusion is that computation and actuator authority must be considered together when designing a landing controller near hardware limits.

---

## Directory structure

```text
config/
control/
core/
dynamics/
guidance/
ocp/
utils/

run_mpc_replay.py
run_hardware_search_checkpointed.py
run_hardware_search_full.py
run_ocp_path_pool_refine.py
hardware_search_common.py

requirements.txt
README.md
VALIDATION_LOG.txt
```

If this repository is used together with the processed output package, the final analysis tables and figures are stored under:

```text
data/
results/
analysis/
docs/
```

---

## Basic usage

Create a Python environment with the required dependencies:

```bash
pip install -r requirements.txt
```

Example MPC hardware search command:

```bash
python run_hardware_search_checkpointed.py \
  --stage all \
  --output-dir outputs_hardware_search_example \
  --num-scenarios 20 \
  --seed 42 \
  --scenario-ids 0,1,2,3,4 \
  --num-workers 4 \
  --freq-list 20 \
  --final-entry-mode bisection \
  --final-grid-step 0.001 \
  --reclaim-grid-step 0.001
```

Example OCP refinement command:

```bash
python run_ocp_path_pool_refine.py \
  --main4-output-dir outputs_hardware_search_example \
  --output-dir outputs_ocp_refine_example \
  --num-scenarios 20 \
  --seed 42 \
  --scenario-ids 0,1,2,3,4 \
  --candidate-source shortlist \
  --candidate-tag closest \
  --freq-list 20 \
  --ocp-floor-source zero \
  --max-permutations 24
```

The exact final report results were generated from the processed output tables included in the cleaned analysis package, not by re-running the full solver pipeline from scratch.

---

## Reproducibility

The repository is intended to support the final report figures and methodology discussion. Due to computational cost, large raw solver outputs and checkpoint archives are excluded.

The included processed CSV files and figures are sufficient to verify the final comparison tables and presentation plots.

---

## Citation

If this repository is referenced, please cite the final project report and the archived repository release.

```text
Salih Okdemir, Compute-Hardware Trade-off in Autonomous V-BAT Landing,
Graduation Project, Istanbul Technical University, 2026.
```
