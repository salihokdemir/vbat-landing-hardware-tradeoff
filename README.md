# Compute and Hardware Trade-off in Autonomous V-BAT Landing

This repository is the cleaned reproducibility package for the graduation project on MPC update frequency and minimum actuator authority in a simplified V-BAT landing simulation.

The package contains the active code snapshot, selected processed run outputs, final CSV tables, final plots, and report ready text used to support the final analysis.

## Scope of the included comparison

The final selected-frequency comparison in this package uses:

- MPC update frequencies: 5, 10, 15, 20, 40, and 80 Hz
- candidate policy: closest-only MPC hardware search candidate
- hardware metrics: T/W, thrust-rate limit, vector-angle limit, and vector-rate limit
- common scenario logic and scenario-relative component-wise ratios

The repository is intended to support the final report figures and tables. It is not a complete raw solver archive.

## Repository layout

```text
src/vbat_landing_mpc_ocp/            Active code snapshot
analysis/                        Small reproduction script for summary figures
data/raw_selected_outputs/        Selected Main4 and Main5 CSV outputs
data/completion_outputs/          C0 completion check outputs used for sensitivity discussion
results/strict_common_set/        Final selected-frequency tables, plots, and text
results/completion_sensitivity/   Completion and sensitivity addendum outputs
results/rescue_diagnostics/       Search-sensitive scenario diagnostics
figures_reproduced/               Created if the reproduction script is run
```

## Reproducing summary figures

Install the Python dependencies used for the analysis environment. A minimal plotting run only needs pandas and matplotlib:

```bash
pip install pandas matplotlib
python analysis/reproduce_summary_figures.py
```

The full MPC and OCP runs require the project environment in `src/vbat_landing_mpc_ocp/requirements.txt`.

## Important interpretation note

The results should be interpreted as a simulation-based design-support analysis, not certified aircraft hardware sizing. The comparison operates close to hardware feasibility boundaries, so local nonlinear solver behavior, trajectory inheritance, deck motion, wind profile, and scenario coverage can affect the ranking.

The main conclusion is not a universal best frequency. The final result should be read as a multi-metric compute and hardware trade-off map.

## Final report citation

If this repository is used in the final report, cite a fixed release or commit hash, not only the moving `main` branch.

Example:

```text
Code and processed data supporting the final figures are available at:
https://github.com/<username>/vbat-landing-hardware-tradeoff
Release: v1.0-final, commit: <commit-hash>
```
