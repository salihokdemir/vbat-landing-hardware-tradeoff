# Completion-scenario diagnostic summary

This diagnostic uses the cleaned comparison set with frequencies **5, 10, 15, 20, 40, and 80 Hz**. The completion scenarios in this set are **s01, s03, and s19**.

## Completion slot map

| Scenario | Frequency | c0x09 | c0x11 | Interpretation |
|---|---:|---|---|---|
| s01 | 80 Hz | completed, 24/24 permutations | completed, 24/24 permutations | boundary/search-sensitive slot |
| s03 | 40 Hz | completed, 24/24 permutations | completed, 24/24 permutations | boundary/search-sensitive slot |
| s19 | 40 Hz | completed, 24/24 permutations | completed, 24/24 permutations | boundary/search-sensitive slot |

## Main diagnostic observations

- The completion scenarios are not simply random rows; they show higher-than-average deck heave intensity indicators. The largest feature differences are deck peak vertical speed and deck heave amplitude/peak-to-peak motion.
- The completion group has a slightly higher mean scenario score than the other scenarios, but the sample size is small (**n=3**), so this should be reported as diagnostic evidence rather than a statistical claim.
- The completion runs recover all three raw-missing slots under both c0x09 and c0x11. Therefore, the missing raw slots should not be interpreted as direct physical impossibility; they are better described as search/solver-sensitive boundary cases.
- This supports keeping the primary ranking as a cleaned common-success comparison, while using completion results as robustness diagnostics.
