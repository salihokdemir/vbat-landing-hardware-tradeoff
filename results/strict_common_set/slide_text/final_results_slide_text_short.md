# A. Final dataset

- Cleaned raw/common-success comparison set.
- Frequencies compared: **5, 10, 15, 20, 40, 80 Hz**.
- Closest-only Main4 candidates; Main5/OCP used for success filtering and hardness weights.

**Caption:** Final comparison uses a fixed common-success scenario set with 17 scenarios.

**Explanation TR:** Bu slaytta kapsamı netleştiriyorum. Analiz sadece temiz raw kaynaklar ve aynı ortak başarı senaryo seti üzerinde yapılıyor. Böylece her frekans aynı senaryolarda karşılaştırılıyor ve scenario-set kaynaklı bias azaltılıyor.

# B. Component-wise hardware burden

- Ratios are scenario-relative; **1.0 is the baseline**.
- Values below 1.0 mean lower-than-average demand.
- Component-wise ratios are primary; scalar score is only a summary.

**Caption:** Component ratios show the multi-metric trade-off directly.

**Explanation TR:** Bu slaytta asıl kanıt component-wise tablodur. 80 Hz özellikle Tdot, delta ve delta-dot tarafında güçlü görünüyor, fakat bu tek yönlü bir frequency trendi anlamına gelmiyor. Her metrik farklı davranıyor.

# C. Frequency ranking and sensitivity

- Main ranking: **80 Hz > 20 Hz > 15 Hz > 40 Hz > 10 Hz > 5 Hz**.
- 80 Hz remains first under tested weight and aggregation checks.
- Middle-band ordering is close and should not be overinterpreted.

**Caption:** The scalar score summarizes the component-wise burden into one ranking.

**Explanation TR:** Ana skor 80 Hz'i bu cleaned comparison set içinde en düşük burden olarak gösteriyor. Ancak 15, 20, 40 ve 10 Hz arasındaki farklar ve sıralama daha hassas. Bu yüzden orta grup için kesin dominance iddiası kurmuyoruz.

# D. Interpretation and limitations

- Do not interpret this as a universal optimum.
- Do not claim that higher frequency always helps.
- The relationship remains multi-metric and non-monotonic.

**Caption:** The result is a dataset-conditioned trade-off, not a universal frequency law.

**Explanation TR:** Sonuç, bu cleaned common-success set içinde 80 Hz'in en iyi trade-off'u verdiğini gösteriyor. Fakat bu global optimum ya da monoton iyileşme iddiası değildir. Component-wise trade-offlar sonuç yorumunun merkezinde kalmalı.
