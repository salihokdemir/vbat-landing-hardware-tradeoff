# Final analysis polished package

This package contains slide-ready tables, plots, captions, speaker notes, and a polished report paragraph for the cleaned common-success comparison set.

## Comparison set

- Frequencies compared: **5, 10, 15, 20, 40, 80 Hz**
- Candidate policy: **closest-only**
- Main4/MPC: hardware burden values
- Main5/OCP: success filtering and OCP-hardness weight calibration
- Common-success scenarios: **17** scenarios

## Recommended slides to use

1. Final dataset and scope
2. Component ratio heatmap
3. Frequency ranking table
4. Sensitivity summary
5. Interpretation and limitations

## Recommended files

- `plots/final_component_ratio_heatmap.png`
- `plots/final_frequency_ranking_slide.png`
- `plots/final_weighted_score_by_frequency.png`
- `slide_text/final_results_slide_text_short.md`
- `slide_text/sensitivity_summary_one_slide.md`
- `report_text/final_report_results_section_polished.md`

## Do not overclaim

- Do not say 80 Hz is globally optimal.
- Do not say increasing frequency always reduces hardware.
- Do not interpret the scalar score without the component-wise ratios.
- Do not overinterpret the middle-band ordering among close frequencies.

## Main result

Main ranking: **80 Hz > 20 Hz > 15 Hz > 40 Hz > 10 Hz > 5 Hz**

The key message is that 80 Hz gives the lowest scenario-relative hardware burden score in this cleaned comparison set, but the result should be presented as a multi-metric trade-off rather than as a universal frequency law.
