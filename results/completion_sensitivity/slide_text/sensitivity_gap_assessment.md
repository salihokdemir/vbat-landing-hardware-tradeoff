# Sensitivity gap assessment for `bit1.pdf`

## What is already present in the deck
- The deck already explains the main methodology: Main4 hardware search, path-pool Main5/OCP refinement, OCP-hardness weighting, and common-success comparison.
- The final result section includes a component-wise heatmap, final ranking, scenario winners/top-3 counts, and a short sensitivity message.

## What is missing or too compressed
1. **A compact rank-stability table** for the sensitivity slide. The current slide says the result is robust, but it does not show the evidence.
2. **A weight-method sensitivity table**: scenario median k, scenario mean k, trimmed k, all-permutation median/mean, and equal weights.
3. **An aggregation sensitivity table**: geomean, median, arithmetic mean, trimmed log-geomean, and average rank.
4. **A coupling-lambda check**: lambda = 0, 0.5, 1. This can stay in a backup table but should be available.
5. **A common-set/completion sensitivity table**: raw common-success set versus c0x09 and c0x11 completion variants. This is important because the current deck mentions strict common-success filtering and excluded scenarios.
6. **Candidate-family diagnostic**: closest is the primary policy, but sparse/balanced candidates were introduced earlier. A backup sensitivity table should show whether the conclusion depends strongly on the closest-only choice.

## Recommended placement
- Replace or extend slide 40 with `sensitivity_one_slide_table.png`.
- Keep `common_set_completion_sensitivity.csv` and `candidate_family_sensitivity_summary.csv` as appendix/backup tables.
- Mention completion variants in the report as robustness/diagnostic analysis, not as the primary result.
