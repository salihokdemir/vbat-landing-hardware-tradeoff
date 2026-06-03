# Sensitivity analysis addendum for report

The current presentation already contains the main ranking and a qualitative sensitivity statement. To make this result defensible in the final report, the sensitivity evidence should be made explicit. I recommend adding three compact sensitivity tables.

First, the primary selected-set ranking should be tested against different OCP-hardness weighting definitions and aggregation rules. In the raw selected set, the leading frequency remains 80 Hz for the tested weight methods and aggregation checks, while 5 Hz remains the weakest case. The middle band, however, is not a strict ordering: 20, 15, 10, and 40 Hz change order depending on the summary method. This supports the statement that the hardware-frequency relationship is multi-metric and non-monotonic.

Second, the strict common-set choice should be treated as a sensitivity dimension. The raw selected set contains 17 common-success scenarios. When completion variants are used, the common-success set increases to 20 scenarios for c0x09 and 20 scenarios for c0x11. In these completion variants, the top pair becomes 10 Hz and 80 Hz, with 10 Hz slightly ahead in scalar score. This does not invalidate the primary result; instead, it shows that the top conclusion is sensitive to scenario-set completion and should be reported as a cleaned-set trade-off, not a universal optimum.

Third, because the methodology introduced closest, sparse, and balanced candidate families, a candidate-family backup table should be included. The primary result should remain closest-only for consistency, but candidate-family sensitivity is useful to show whether the conclusion is mainly an artifact of that one selection rule.
