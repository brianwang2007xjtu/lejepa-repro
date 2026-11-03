# Run summary — `expB_in100_vits16_128`

- log records: 1036
- last step: 19760

| metric | first | last | min | max |
|---|---|---|---|---|
| prediction loss (`train/pred`) | 0.3905 | 0.1227 | 0.1093 | 0.4085 |
| SIGReg (`train/sigreg`) | 9.8927 | 1.5361 | 1.3127 | 9.8927 |
| LeJEPA total (`train/lejepa`) | 0.5805 | 0.1510 | 0.1401 | 0.5805 |
| online probe loss (`train/probe`) | 4.7251 | 2.7713 | 2.5033 | 4.7251 |
| online probe acc (val) (`eval/online_probe_acc`) | 0.1424 | 0.3810 | 0.1424 | 0.3810 |
| effective rank (`diag/effective_rank`) | 8.2556 | 15.7938 | 8.2556 | 15.7938 |
| min feature std (`diag/feature_std_min`) | 0.4015 | 0.9055 | 0.4015 | 0.9094 |
| ||mean embedding|| (`diag/mean_norm`) | 0.5352 | 0.0597 | 0.0495 | 0.5352 |
| mean |off-diag corr| (`diag/cov_offdiag`) | 0.2397 | 0.0313 | 0.0313 | 0.2397 |

## Online probe accuracy (validation split)

| step | accuracy | n |
|---|---|---|
| 2470 | 14.24% | 5000 |
| 4940 | 18.30% | 5000 |
| 7410 | 21.00% | 5000 |
| 9880 | 25.34% | 5000 |
| 12350 | 29.76% | 5000 |
| 14820 | 32.42% | 5000 |
| 17290 | 35.26% | 5000 |
| 19760 | 38.10% | 5000 |

