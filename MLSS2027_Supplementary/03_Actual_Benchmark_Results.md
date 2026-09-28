\# Actual Benchmark Results



\## Dataset and Spatial Evaluation



The YCEO UHI Enhanced ML clean daytime dataset contains 9,790 final modeling observations, 9 predictors, and 182 spatial blocks. Evaluation uses 5-fold spatial cross-validation.



| Metric | Mean | SD |

|---|---:|---:|

| Spatial R² | 0.3813 | 0.0820 |

| Spatial RMSE | 0.6501 | 0.0221 |

| Spatial MAE | 0.4770 | 0.0136 |



\## Failure Detection Under Geographic Distribution Shift



Failure-detection performance was compared between an OOD score and predictive uncertainty.



| Method | AUROC | AUPRC |

|---|---:|---:|

| OOD | 0.6178 | 0.2776 |

| Raw Uncertainty | 0.6678 | 0.3316 |

| Calibrated Uncertainty | 0.6679 | 0.3305 |



Raw uncertainty improved AUROC over the OOD benchmark by 0.0500 and AUPRC by 0.0541. The reported permutation tests indicate statistically significant differences, with p = 0.0002 for AUROC and p < 0.001 for AUPRC.



\## AURC and Selective Prediction



| Method | Mean AURC | SD |

|---|---:|---:|

| Random | 0.4770 | 0.0136 |

| OOD | 0.4139 | 0.0225 |

| Raw Uncertainty | 0.3843 | 0.0220 |

| Calibrated Uncertainty | 0.3852 | 0.0227 |

| Oracle | 0.1861 | 0.0064 |



Raw uncertainty achieves a 19.4% lower AURC than random selection, indicating improved risk-coverage behavior when unreliable predictions are selectively rejected.



\## Interpretation



The benchmark results show that predictive uncertainty contains useful information about prediction failure under geographic distribution shift. Compared with the OOD benchmark, uncertainty provides stronger failure discrimination and lower AURC. The results support the use of uncertainty-aware selective prediction as a mechanism for reducing exposure to high-risk urban heat predictions.

