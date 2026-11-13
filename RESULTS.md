# Results

All numbers come from `scripts/run_baselines.py` (`make baselines`) with the default `ExperimentConfig` (window of 30 cycles, RUL target capped at 125, seed 0).

## M1: baselines

Lower is better for every column. Validation RMSE is measured on a held-out 20 percent of the training engines, over every cycle, against the capped target. Test RMSE and the NASA score use the last cycle of each official test engine against the official RUL file, which is not capped.

| Subset | Model | Val RMSE | Test RMSE | Test NASA score |
|---|---|---|---|---|
| FD001 | Linear regression | 15.38 | 16.61 | 466 |
| FD001 | Gradient boosting | 11.82 | 14.53 | 294 |
| FD002 | Linear regression | 16.74 | 28.64 | 11827 |
| FD002 | Gradient boosting | 14.57 | 26.93 | 8287 |
| FD003 | Linear regression | 16.61 | 18.38 | 616 |
| FD003 | Gradient boosting | 13.04 | 15.70 | 450 |
| FD004 | Linear regression | 19.03 | 30.36 | 7729 |
| FD004 | Gradient boosting | 15.98 | 26.74 | 4824 |

FD001 is the headline subset because published results use it. FD002 and FD004 mix six operating conditions.

### The RUL cap sets a floor on test RMSE

Models are trained on a target capped at 125 cycles, so they cannot predict a larger RUL, but the official test RUL is not capped. Many test engines have a true RUL above 125 (57 of 259 in FD002 and 67 of 248 in FD004). A model that predicted `min(true RUL, 125)` perfectly would still score:

| Subset | Lowest possible test RMSE |
|---|---|
| FD001 | 3.74 |
| FD002 | 18.32 |
| FD003 | 4.71 |
| FD004 | 19.53 |

Much of the gap between FD001 and FD002 or FD004 in test RMSE comes from this floor, not only from the operating conditions. Validation RMSE, which uses the capped target, rises less between subsets (11.82 to 14.57 for gradient boosting).

### Effect of per-condition normalisation on FD002

The same pipeline with one global normalisation instead of one per operating condition:

| Subset | Model | Val RMSE | Test RMSE | Test NASA score |
|---|---|---|---|---|
| FD002 | Linear regression | 16.98 | 29.24 | 13687 |
| FD002 | Gradient boosting | 18.82 | 29.70 | 12087 |

Normalising per condition lowers the gradient boosting test RMSE from 29.70 to 26.93 and the NASA score from 12087 to 8287. Linear regression improves less, from 29.24 to 28.64.

## Notes

The test sets hold one score per engine (100, 259, 100 and 248 engines), so differences of one or two RMSE points between models are within the noise of a single seed. The comparison with a global normalisation uses the same seed and the same engine split.
