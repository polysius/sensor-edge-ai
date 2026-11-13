# Results

All numbers come from `make baselines` with the default `ExperimentConfig` (window of 30 cycles, RUL target capped at 125, seed 0).

## M1: baselines

Lower is better for every column. Validation RMSE is measured on a held-out 20 percent of the training engines, over every cycle, against the capped target. Test RMSE and the NASA score use the last cycle of each official test engine against the official RUL file.

| Subset | Model | Val RMSE | Test RMSE | Test NASA score |
|---|---|---|---|---|
| FD001 | Linear regression | 15.38 | 16.61 | 466 |
| FD001 | Gradient boosting | 11.82 | 14.53 | 294 |
| FD002 | Linear regression | 16.77 | 28.72 | 12033 |
| FD002 | Gradient boosting | 14.56 | 26.65 | 7675 |
| FD003 | Linear regression | 16.61 | 18.38 | 616 |
| FD003 | Gradient boosting | 13.04 | 15.70 | 450 |
| FD004 | Linear regression | 19.01 | 30.41 | 7896 |
| FD004 | Gradient boosting | 15.93 | 27.04 | 5220 |

FD001 is the headline subset because published results use it. The other three are included because FD002 and FD004 mix six operating conditions and are much harder.

### Effect of per-condition normalisation on FD002

The same pipeline with one global normalisation instead of one per operating condition:

| Subset | Model | Val RMSE | Test RMSE | Test NASA score |
|---|---|---|---|---|
| FD002 | Linear regression | 16.98 | 29.24 | 13687 |
| FD002 | Gradient boosting | 18.82 | 29.70 | 12087 |

Normalising per condition lowers the gradient boosting test RMSE from 29.70 to 26.65 and the NASA score from 12087 to 7675. Linear regression improves less, from 29.24 to 28.72.

## Notes

The test sets hold one score per engine (100, 259, 100 and 248 engines), so differences of one or two RMSE points between models are within the noise of a single seed. The comparison with a global normalisation uses the same seed and the same engine split.
