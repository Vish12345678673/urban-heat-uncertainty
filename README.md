# YCEO-UHI: Spatial Machine Learning Analysis of Urban Heat Islands



## Overview



This repository contains the computational workflow for a global urban heat island (UHI) analysis using YCEO urban clusters and geospatial/environmental predictors.



The study evaluates daytime annual surface urban heat island (SUHI) using machine-learning regression with spatially independent validation and model explainability.



The workflow combines spatially blocked machine-learning validation, model comparison, SHAP-based interpretation, prediction-error analysis, and Moran's I spatial autocorrelation testing.



\---



## Dataset



The final modeling dataset contains:



\- Original YCEO clusters: 10,136

\- Final modeling observations: 9,790

\- Predictors: 9

\- Spatial blocks: 182



The nine predictors are:



1\. NDVI

2\. Elevation

3\. Built-up surface

4\. Population

5\. Mean temperature

6\. Mean monthly precipitation

7\. Urban land-cover fraction

8\. Water land-cover fraction

9\. Vegetation land-cover fraction



The processed dataset is provided in:



`data/YCEO\_UHI\_Enhanced\_ML\_Clean\_Daytime.csv`



\---



## Target Variable



The prediction target is:



`Annual\_day`



This represents the annual daytime surface urban heat island metric used in the modeling dataset.



\---



## Spatial Validation



Spatial validation is based on geographic blocks constructed from latitude and longitude.



Latitude and longitude are divided into 10-degree spatial blocks, and observations belonging to the same spatial block are kept together during validation.



This approach is used to reduce geographic leakage between training and testing observations.



The fixed spatial holdout contains:



\- Training observations: 7,160

\- Testing observations: 2,630

\- Spatial blocks: 182



\---



## Primary Model



The primary machine-learning model is XGBoost regression.



The final model configuration is stored in:



`results/metrics/YCEO\_UHI\_Experiment\_Config.json`



The trained model is provided in:



`models/YCEO\_UHI\_Final\_XGBoost.joblib`



\---



## Final Spatial Holdout Performance



The final XGBoost model achieved:



| Metric | Value |

|---|---:|

| R² | 0.382648 |

| RMSE | 0.614984 |

| MAE | 0.450956 |



These results are based on the fixed spatial holdout containing 2,630 testing observations.



\---



## Five-Fold Spatial Cross-Validation



Five-fold spatial cross-validation was performed using the same spatial-blocking principle.



Results:



| Metric | Mean | SD |

|---|---:|---:|

| R² | 0.381290 | 0.082030 |

| RMSE | 0.650130 | 0.022128 |

| MAE | 0.477012 | 0.013569 |



The five spatial folds produced the following R² values:



| Fold | R² | RMSE | MAE |

|---:|---:|---:|---:|

| 1 | 0.362124 | 0.664973 | 0.492602 |

| 2 | 0.487793 | 0.679385 | 0.480859 |

| 3 | 0.260368 | 0.645712 | 0.482328 |

| 4 | 0.394948 | 0.636438 | 0.473119 |

| 5 | 0.401216 | 0.624143 | 0.456150 |



The mean spatial R² of 0.381 is close to the single spatial holdout R² of 0.383.



However, performance varies between geographic folds, indicating heterogeneous geographic generalization.



\---



## Model Comparison



Four regression models were evaluated using the same spatial holdout:



| Model | Spatial R² | Spatial RMSE | Spatial MAE |

|---|---:|---:|---:|

| XGBoost | 0.382648 | 0.614984 | 0.450956 |

| HistGradientBoosting | 0.382597 | 0.615010 | 0.450944 |

| Random Forest | 0.372324 | 0.620105 | 0.453629 |

| Linear Regression | 0.259672 | 0.673456 | 0.511477 |



The nonlinear tree-based models outperform the linear baseline.



XGBoost achieved the highest spatial R², while its performance was effectively tied with HistGradientBoosting.



Therefore, the results do not support a claim that XGBoost substantially outperforms HistGradientBoosting.



\---



## SHAP Explainability



SHAP (SHapley Additive exPlanations) was used to quantify the contribution of predictors to individual model predictions and to assess global feature importance.



The five highest mean absolute SHAP contributions were:



| Predictor | Mean Absolute SHAP |

|---|---:|

| NDVI | 0.156911 |

| Mean monthly precipitation | 0.151921 |

| Mean temperature | 0.121489 |

| Built-up | 0.085228 |

| Urban land-cover fraction | 0.070283 |



The complete SHAP results are available in:



`results/shap/`



SHAP values describe model associations and prediction contributions. They should not be interpreted as causal effects.



\---



## SHAP Direction Analysis



The analysis also examined the relationship between predictor values and SHAP contributions.



The resulting analysis is provided in:



`results/shap/YCEO\_UHI\_SHAP\_Direction\_Analysis.csv`



Selected results include:



| Feature | Spearman r | Mean SHAP |

|---|---:|---:|

| NDVI | 0.7140 | 0.0518 |

| Mean monthly precipitation | 0.6765 | -0.0254 |

| Mean temperature | -0.9179 | 0.0050 |

| Built-up | 0.9211 | -0.0082 |

| Urban land-cover fraction | 0.1192 | 0.0171 |

| Elevation | -0.1969 | 0.0156 |

| Vegetation land-cover fraction | -0.0465 | 0.0045 |

| Water land-cover fraction | -0.3556 | 0.0004 |

| Population | 0.5600 | -0.0105 |



The direction of a SHAP association is not equivalent to a causal relationship.



\---



## Spatial Prediction Error Analysis



Prediction errors were analyzed across geographic regions.



The overall test-set residual statistics were:



\- Mean residual: -0.060454

\- Mean absolute error: 0.451

\- Residual standard deviation: 0.612122

\- Minimum residual: -3.627690

\- Maximum residual: 4.180018



Absolute-error quantiles were:



| Quantile | Absolute Error |

|---:|---:|

| 50% | 0.346674 |

| 75% | 0.618261 |

| 90% | 0.969223 |

| 95% | 1.237917 |

| 99% | 1.967352 |



Detailed regional error results are available in:



`results/spatial/`



\---



## Moran's I Spatial Autocorrelation



Residual spatial autocorrelation was evaluated using Moran's I with permutation testing.



For eight nearest neighbors:



\- Moran's I: 0.207838

\- Expected I: -0.00038

\- Permutation mean: 0.000038

\- Permutation SD: 0.009160

\- Permutation Z: 22.6852

\- Permutation p-value: 0.001



The sensitivity analysis tested three neighborhood definitions:



| Neighbors (k) | Moran's I | Permutation Z | p-value |

|---:|---:|---:|---:|

| 4 | 0.237035 | 18.3262 | 0.001 |

| 8 | 0.207838 | 22.6852 | 0.001 |

| 16 | 0.184724 | 28.1447 | 0.001 |



The positive and statistically significant Moran's I values indicate that residual prediction errors retain spatial structure.



This is treated as an important limitation of the current predictive model.



\---



## Publication Figures



The repository contains the main analysis figures:



### Figure 1

`figures/Figure\_1\_Model\_Comparison.png`



Comparison of model performance under spatial validation.



### Figure 2

`figures/Figure\_2\_Spatial\_CV.png`



Five-fold spatial cross-validation performance.



### Figure 3

`figures/Figure\_3\_SHAP\_Importance.png`



Global SHAP feature importance.



### Figure 4

`figures/Figure\_4\_Observed\_vs\_Predicted.png`



Observed versus predicted annual daytime SUHI.



### Figure 5

`figures/Figure\_5\_Moran\_Sensitivity.png`



Sensitivity of residual Moran's I to the number of nearest neighbors.



The final results table is:



`figures/Table\_1\_Final\_Results.csv`



\---



## Repository Structure



```text

YCEO\_UHI\_GitHub\_Project/

│

├── code/

│   ├── analysis/

│   │   ├── final\_results\_figures.py

│   │   ├── moran\_sensitivity.py

│   │   ├── moran\_test.py

│   │   ├── shap\_analysis.py

│   │   ├── shap\_dependence\_final.py

│   │   ├── shap\_direction.py

│   │   └── spatial\_error\_analysis.py

│   │

│   ├── gee/

│   │

│   ├── modeling/

│   │   ├── final\_experiment.py

│   │   ├── model\_comparison.py

│   │   ├── spatial\_cv.py

│   │   └── tune\_model.py

│   │

│   └── preprocessing/

│       └── create\_clean\_dataset.py

│

├── data/

│   └── YCEO\_UHI\_Enhanced\_ML\_Clean\_Daytime.csv

│

├── figures/

│   ├── Figure\_1\_Model\_Comparison.png

│   ├── Figure\_2\_Spatial\_CV.png

│   ├── Figure\_3\_SHAP\_Importance.png

│   ├── Figure\_4\_Observed\_vs\_Predicted.png

│   ├── Figure\_5\_Moran\_Sensitivity.png

│   └── Table\_1\_Final\_Results.csv

│

├── models/

│   └── YCEO\_UHI\_Final\_XGBoost.joblib

│

├── results/

│   ├── metrics/

│   ├── predictions/

│   ├── shap/

│   └── spatial/

│

├── LICENSE

├── README.md

└── requirements.txt




