# Completion-scenario diagnostics package

This package analyzes the scenario-frequency slots recovered by the c0 completion runs within the cleaned comparison set.

## Main files

- `tables/rescue_slot_map.csv`
- `tables/scenario_feature_comparison.csv`
- `tables/rescue_scenario_feature_percentiles.csv`
- `tables/rescue_slot_metric_percentiles.csv`
- `tables/rescue_vs_other_hardware_burden.csv`
- `plots/rescue_slot_map_table.png`
- `plots/rescue_feature_zscore_bar.png`
- `plots/rescue_feature_percentile_heatmap.png`
- `plots/rescue_slot_metric_percentiles_c0x09.png`
- `plots/rescue_slot_metric_percentiles_c0x11.png`
- `plots/rescue_vs_other_score_c0x09.png`
- `plots/rescue_vs_other_score_c0x11.png`
- `slide_text/rescue_scenario_backup_slide.md`
- `report_text/rescue_scenario_report_paragraph.md`

## Key interpretation

The completion scenarios are best interpreted as boundary/search-sensitive cases. The completion runs recover the raw-missing slots, so the missing raw cases should not be described as direct physical infeasibility. The feature comparison suggests somewhat stronger deck-motion intensity in the completion scenarios, but the small sample size means this is diagnostic evidence rather than a formal statistical conclusion.
