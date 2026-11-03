# Run summary — `expC_in100_vits16_8view`

- log records: 1702
- last step: 32950

| metric | first | last | min | max |
|---|---|---|---|---|
| prediction loss (`train/pred`) | 0.2409 | 0.1860 | 0.1732 | 0.5412 |
| SIGReg (`train/sigreg`) | 35.3532 | 2.1679 | 1.9938 | 35.3532 |
| LeJEPA total (`train/lejepa`) | 1.9965 | 0.2851 | 0.2708 | 1.9965 |
| online probe loss (`train/probe`) | 4.7745 | 1.9474 | 1.7020 | 4.7745 |
| online probe acc (val) (`eval/online_probe_acc`) | 0.3332 | 0.6178 | 0.3332 | 0.6178 |
| effective rank (`diag/effective_rank`) | 53.9241 | 49.4422 | 26.7205 | 53.9241 |
| min feature std (`diag/feature_std_min`) | 0.2549 | 0.7086 | 0.2549 | 0.7092 |
| ||mean embedding|| (`diag/mean_norm`) | 3.5554 | 0.2321 | 0.2321 | 3.5554 |
| mean |off-diag corr| (`diag/cov_offdiag`) | 0.1931 | 0.1164 | 0.1162 | 0.1931 |

## Online probe accuracy (validation split)

| step | accuracy | n |
|---|---|---|
| 6590 | 33.32% | 5000 |
| 13180 | 47.58% | 5000 |
| 19770 | 55.64% | 5000 |
| 26360 | 59.98% | 5000 |
| 32950 | 61.78% | 5000 |

