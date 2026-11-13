# sensor-edge-ai

Remaining useful life (RUL) prediction for turbofan engines on NASA C-MAPSS, with the aim of adding anomaly detection, uncertainty estimates, INT8 edge inference and a streaming demo.

The dataset is C-MAPSS, the default in the project spec. FD001 is the headline subset. FD002 to FD004 are also run.

## Status

| Milestone | Scope | State |
|---|---|---|
| M1 | Data pipeline and classical baselines | Done |
| M2 | LSTM and 1D-CNN RUL models | Not started |
| M3 | Anomaly detection and Gaussian process uncertainty | Not started |
| M4 | ONNX export, INT8 quantisation, edge benchmark | Not started |
| M5 | Redpanda streaming and Grafana | Not started |

## M1 pipeline

1. Split the training engines into train and validation sets by engine unit, so no engine appears on both sides. The official test file holds different engines.
2. Cluster the three operating settings with k-means (one cluster for FD001 and FD003, six for FD002 and FD004) and standardise every sensor within its cluster. Clusters and statistics are fitted on the training engines only.
3. Drop sensors that are constant after normalisation. The list is derived from the data and differs between subsets.
4. Label each training cycle with the RUL, capped at 125 cycles. Early in an engine's life degradation is not visible, so the target stays flat until the engine nears failure.
5. Build a window of 30 cycles ending at each cycle. Short histories are padded by repeating the first reading, which also lets the shortest test engines be scored.
6. Summarise each window with the mean, standard deviation, slope and last value of every sensor, then fit linear regression and gradient boosting.

Test scoring uses the last window of each official test engine against the official RUL file. The metrics are RMSE and the PHM08 NASA score, which penalises late predictions more than early ones.

## Results

FD001 test RMSE is 16.61 for linear regression and 14.53 for gradient boosting. The full table, including the effect of per-condition normalisation on FD002, is in [RESULTS.md](RESULTS.md).

## Run it

The core package needs only numpy, pandas and scikit-learn.

```
pip install -e .[dev]
make data        # download C-MAPSS and verify the checksums
make test
make lint
make baselines   # writes artifacts/baselines.md
```

`make data` downloads from the PHM data mirror, because the NASA portal link rejects scripts. It checks the SHA-256 of both archives.

## Limitations

Results use one seed and one engine split, so small differences between models are not significant. The validation RMSE is measured over every cycle of the held-out engines, while the test RMSE uses only the last cycle of each test engine, so the two are not directly comparable.
