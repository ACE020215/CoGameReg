# CoGameReg reference results

The table below records the CoGameReg RMSE values reported from the reference
server experiment. Each cell is the mean and population standard deviation over
ten runs with seeds 42--51.

| Dataset | Label ratio 0.025 | Label ratio 0.05 | Label ratio 0.10 |
|---|---:|---:|---:|
| Abalone | 0.0904 +/- 0.0042 | 0.0851 +/- 0.0020 | 0.0815 +/- 0.0022 |
| Bank32nh | 0.1342 +/- 0.0071 | 0.1249 +/- 0.0040 | 0.1166 +/- 0.0036 |
| Elevators | 0.0608 +/- 0.0039 | 0.0576 +/- 0.0015 | 0.0557 +/- 0.0006 |
| Folds5x2_pp | 0.0654 +/- 0.0047 | 0.0614 +/- 0.0018 | 0.0594 +/- 0.0009 |
| Kin8nm | 0.1539 +/- 0.0046 | 0.1467 +/- 0.0025 | 0.1360 +/- 0.0022 |
| Parkinsons | 0.0713 +/- 0.0057 | 0.0676 +/- 0.0053 | 0.0626 +/- 0.0012 |
| Puma8NH | 0.1887 +/- 0.0083 | 0.1770 +/- 0.0048 | 0.1695 +/- 0.0031 |
| Space_ga | 0.1650 +/- 0.0058 | 0.1565 +/- 0.0054 | 0.1509 +/- 0.0044 |
| Wind | 0.0934 +/- 0.0040 | 0.0883 +/- 0.0026 | 0.0851 +/- 0.0017 |
| Wine_quality | 0.1348 +/- 0.0049 | 0.1300 +/- 0.0033 | 0.1250 +/- 0.0031 |

The complete matrix was reproduced locally with deviations of at most 0.0002 in
the displayed four-decimal means or standard deviations. Such small variation
can arise from differences in numerical libraries and the XGBoost execution
environment. The server values above are the archival manuscript values.

## Quick reference check

For Abalone at label ratio 0.025 and seed 42, the locked local environment
produced:

```text
RMSE = 0.0897684359091
```

which is displayed by the benchmark script as `0.0898`.
