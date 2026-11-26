# Water Quality Index Estimation for Indian Rivers
## Machine Learning Project Report

**Student:** Aman Naveen  
**Programme:** B.Sc. Computer Science  
**Institution:** Ramanujan College, University of Delhi  
**Domain:** Water quality monitoring  
**Relevant SDGs:** SDG 6 — Clean Water and Sanitation; SDG 14 — Life Below Water

---

## Abstract

This project studies the estimation of river-water quality using a weighted-arithmetic Water Quality Index (WQI) and supervised machine-learning methods. Seven physicochemical and microbiological measurements are combined into a project-defined WQI. Regression models estimate the continuous index, while classifiers predict its three categories. Two feature sets are compared: all available measurements and a reduced set of four field measurements. A chronological holdout uses observations through 2012 for training and 2013–2014 for testing. The recorded experiment shows near-perfect performance for the full-measurement Gradient Boosting model (RMSE 0.823; R² 0.997), but this is largely a consequence of learning the formula used to construct the target. With field measurements only, the best regression model is Random Forest (RMSE 11.324; R² 0.440), but it is outperformed by a station-mean baseline (RMSE 7.930; R² 0.726). We therefore define the practical target as beating the station-mean baseline; the field-only models do not meet that target. These results demonstrate why baselines, leakage-aware validation, and careful interpretation are essential when evaluating environmental machine-learning systems.

**Keywords:** Water Quality Index, river monitoring, regression, classification, Random Forest, Gradient Boosting, India.

## 1. Introduction

Access to safe water is central to public health, ecological sustainability, and the achievement of SDG 6. River monitoring produces multiple measurements, but interpreting them individually can be difficult. A composite index offers a compact summary of selected measurements.

This project calculates a weighted-arithmetic WQI from Indian river-monitoring data and evaluates whether machine-learning models can estimate the index or its categories. It also examines a practical reduced-feature setting in which only temperature, dissolved oxygen, pH, and conductivity are available.

### 1.1 Motivation and SDG targets

The study is motivated by the monitoring problem behind SDG 6.3 and SDG 14.1. UN-Water reports that, among the 42 countries for which total wastewater data were sufficient in its 2024 update, only 60% of total wastewater was safely treated; the report also notes that these data cover only about 12% of the world's population, so the figure should not be treated as a global estimate. SDG 6.3.1 tracks the proportion of wastewater flows that are safely treated before discharge. [1]

SDG 14.1 explicitly targets the prevention and significant reduction of marine pollution from land-based activities, including nutrient pollution and marine debris; its indicator 14.1.1 covers coastal eutrophication and plastic-debris density. River monitoring is relevant because riverine pollution is one pathway by which land-based contaminants reach coastal and marine systems. [2]

For this study, the operational target is deliberately quantitative: a field-only ML model should beat the station-mean regression baseline (test RMSE 7.930) on the chronological 2013–2014 holdout. The experiment therefore treats failure to beat this baseline as a useful negative result rather than selecting a more favourable baseline or random split.

### 1.1 Objectives

- Clean and document the river-monitoring dataset.
- Calculate a weighted-arithmetic WQI from seven measurements.
- Compare regression and classification algorithms.
- Compare full-measurement and field-only feature sets.
- Evaluate models against simple baselines.
- Identify limitations affecting generalisation and operational use.

### 1.2 Research questions

1. How accurately can models reproduce the calculated WQI?
2. How much performance is lost when laboratory measurements are excluded?
3. Do machine-learning models improve on mean and station-based baselines?
4. What do the results imply about deploying such models for water-quality monitoring?

## 2. Data

### 2.1 Source and coverage

The project uses the Indian Water Quality Data dataset, distributed through Kaggle and included in the repository as `DOC-20261117-WA0000.csv`. The dataset was accessed on 17 October 2026. The Kaggle dataset page does not expose a clearly verifiable dataset licence in the retrieved metadata; therefore this report does not invent a licence. The repository retains source attribution and the applicable Kaggle terms. The source file contains 1,991 records and 12 fields, covering 2003–2014.

The fields include station code, location, state, temperature, dissolved oxygen, pH, conductivity, biochemical oxygen demand, nitrate/nitrite nitrogen, faecal coliform, total coliform, and year. Field definitions and units are documented in `DATA_DICTIONARY.md`.

### 2.2 Cleaning

Measurements are converted to numeric values; non-numeric tokens are treated as missing. Duplicate records are removed. pH outside the physically valid range 0–14 is marked missing. Records missing any of the seven WQI inputs are excluded from WQI calculation. Missing predictive features are imputed within model pipelines, so imputation statistics are learned from training data only.

The reproduced run retained 1,572 complete WQI observations. Of these, 1,111 observations through 2012 formed the training set and 461 observations from 2013–2014 formed the test set.

## 3. Methodology

### 3.1 Weighted-arithmetic WQI

For each parameter (i), the quality rating is calculated as:

`Qi = 100 × (Vi − Videal,i) / (Si − Videal,i)`

The relative unit weight is:

`Wi = K / Si,  K = 1 / Σ(1 / Si)`

The aggregate index is:

`WQI = Σ(Wi × Qi)`

The implementation uses ideal pH 7 and ideal dissolved oxygen 14.6 mg/L. Parameter standards are pH 8.5, DO 5 mg/L, BOD 3 mg/L, conductivity 2,250 µmhos/cm, nitrate/nitrite nitrogen 10 mg/L, and 500 MPN/100 mL for each coliform measure. Quality ratings are clipped to 0–100.

The nitrate and faecal-coliform limits are project assumptions and should not be represented as official CPCB designated-use limits. The resulting index is a coursework research measure, not a regulatory compliance score.

The project maps WQI ≤ 50 to Good / Excellent, WQI > 50 and ≤ 75 to Poor, and WQI > 75 to Very Poor.

### 3.2 Feature sets

- **Set A:** temperature, DO, pH, conductivity, BOD, nitrate, faecal coliform, total coliform.
- **Set B:** temperature, DO, pH, conductivity.

Set A includes the variables used in the WQI formula. Consequently, high predictive performance on Set A is expected to reflect reconstruction of the target formula. Set B tests the more constrained field-measurement use case.

### 3.3 Models and baselines

Regression models: Linear Regression, Ridge Regression, k-Nearest Neighbours, RBF Support Vector Regression, Decision Tree, Random Forest, and Gradient Boosting.

Classification models: Logistic Regression, k-Nearest Neighbours, Decision Tree, Random Forest, and Gradient Boosting.

Baselines: training-mean and station-mean regression; majority-class and station-majority classification.

### 3.4 Validation design

The selected hyperparameters are recorded directly by the experiment script in outputs/selected_hyperparameters.csv. This file, rather than an independently typed table, is the source for the final manuscript's hyperparameter table. The script also records the fold-level CV-RMSE mean and standard deviation.

Observations are split chronologically: training through 2012 and testing on 2013–2014. Hyperparameters are selected with expanding year-blocked validation folds within the training period. Preprocessing is embedded in scikit-learn pipelines to prevent imputation and scaling from using test-fold statistics.

Regression is evaluated using MAE, RMSE, and R². Classification is evaluated using accuracy and macro-F1, with confusion matrix and per-class report generated for Set A. Permutation importance is computed on the held-out test set.

## 4. Results

### 4.1 Regression

| Model | Set A RMSE | Set A R² | Set B RMSE | Set B R² |
|---|---:|---:|---:|---:|
| Mean baseline | 15.187 | -0.006 | 15.187 | -0.006 |
| Station-mean baseline | 7.930 | 0.726 | 7.930 | 0.726 |
| Linear Regression | 8.509 | 0.684 | 13.406 | 0.216 |
| Ridge Regression | 8.502 | 0.685 | 13.393 | 0.217 |
| k-NN | 6.37 | 0.823 | 11.397 | 0.433 |
| RBF SVR | 2.515 | 0.972 | 11.775 | 0.395 |
| Decision Tree | 3.012 | 0.960 | 12.640 | 0.303 |
| Random Forest | 1.89 | 0.985 | 11.324 | 0.440 |
| Gradient Boosting | 0.823 | 0.997 | 11.34 | 0.439 |

Set A Gradient Boosting produces the lowest recorded RMSE (0.823) and highest R² (0.997). Since all WQI input variables are also supplied to the model, this result primarily indicates that the model can approximate the index calculation.

For Set B, Random Forest has RMSE 11.324 and R² 0.440. The station-mean baseline has RMSE 7.930, so the field-only machine-learning regressors do not outperform this simple benchmark in the recorded experiment.

### 4.1.1 Cross-validation uncertainty and bootstrap intervals

The final GitHub Actions run (workflow run 53, executed 26 November 2026) regenerated the regression outputs. The CV values below are the mean RMSE ± standard deviation across the forward-chaining folds; the test-RMSE intervals are percentile bootstrap 95% intervals from the held-out 2013–2014 predictions.

| Model | Set A CV RMSE ± SD | Set A test RMSE | Set A bootstrap 95% CI | Set B CV RMSE ± SD | Set B test RMSE | Set B bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 9.058 ± 0.575 | 8.509 | 7.728–9.355 | 12.988 ± 1.062 | 13.406 | 12.442–14.485 |
| Ridge Regression | 9.045 ± 0.572 | 8.502 | 7.728–9.338 | 12.983 ± 1.063 | 13.393 | 12.434–14.463 |
| k-NN | 6.099 ± 0.822 | 6.371 | 5.775–7.013 | 10.736 ± 0.712 | 11.397 | 10.573–12.289 |
| RBF SVR | 3.402 ± 0.530 | 2.515 | 2.114–2.987 | 10.988 ± 0.815 | 11.775 | 10.928–12.631 |
| Decision Tree | 3.377 ± 0.518 | 3.012 | 2.631–3.419 | 11.707 ± 0.379 | 12.640 | 11.769–13.494 |
| Random Forest | 2.417 ± 0.446 | 1.851 | 1.493–2.266 | 10.661 ± 0.422 | 11.324 | 10.553–12.116 |
| Gradient Boosting | 1.159 ± 0.287 | 0.823 | 0.729–0.921 | 10.601 ± 0.510 | 11.342 | 10.550–12.088 |

The corresponding machine-readable files are `outputs/regression_A.csv` and `outputs/regression_B.csv`. The selected hyperparameters are recorded in `outputs/selected_hyperparameters.csv`.

### 4.2 Classification

| Model | Set A accuracy | Set A macro-F1 | Set B accuracy | Set B macro-F1 |
|---|---:|---:|---:|---:|
| Majority class | 0.362 | 0.177 | 0.362 | 0.177 |
| Station-majority | 0.731 | 0.740 | 0.731 | 0.740 |
| Logistic Regression | 0.820 | 0.802 | 0.564 | 0.575 |
| k-NN | 0.768 | 0.777 | 0.642 | 0.657 |
| Decision Tree | 0.816 | 0.809 | 0.599 | 0.608 |
| Random Forest | 0.913 | 0.905 | 0.616 | 0.627 |
| Gradient Boosting | 0.933 | 0.930 | 0.657 | 0.661 |

The Set A Gradient Boosting classifier records 93.3% accuracy and 0.930 macro-F1. For Set B, Gradient Boosting is the strongest classifier in the regenerated run at 65.7% accuracy and 0.661 macro-F1, but it remains below the station-majority baseline (73.1% accuracy and 0.740 macro-F1). The earlier report contains 93.9% and 0.937; the difference is retained transparently because the current pipeline uses year-blocked hyperparameter validation.

### 4.3 Uncertainty and error analysis

The final experiment script reports the mean and standard deviation of RMSE across the forward-chaining validation folds and computes a percentile bootstrap 95% interval for test RMSE for each fitted regression model. These values were regenerated successfully in GitHub Actions on 26 November 2026. The machine-readable outputs are outputs/regression_A.csv and outputs/regression_B.csv.

For the practical Set B model, the script additionally writes outputs/error_by_station_B.csv and outputs/error_by_category_B.csv. These tables should be used to report where the Random Forest model fails most strongly, rather than relying only on the overall RMSE. Category-level errors distinguish whether residuals are concentrated in Good / Excellent, Poor, or Very Poor observations; station-level results show whether persistent local conditions explain part of the error.

The station-mean baseline remains the decisive benchmark: the best Set B Random Forest has RMSE 11.324 versus 7.930 for the station-mean baseline. Thus the uncertainty analysis does not change the central conclusion that field-only ML, as implemented here, does not yet beat the simplest station-history predictor.
### 4.3 Error analysis for the best Set B regression model

The best Set B regression model by chronological test RMSE is Random Forest (11.324). The regenerated category-level error summary is:

| WQI category | n | MAE | RMSE | Mean observed WQI | Mean predicted WQI |
|---|---:|---:|---:|---:|---:|
| Good / Excellent | 167 | 8.636 | 11.742 | 42.240 | 48.941 |
| Poor | 205 | 9.782 | 11.767 | 60.875 | 56.378 |
| Very Poor | 89 | 7.389 | 9.305 | 79.994 | 72.834 |

The largest category-level RMSE is in the Poor group (11.767). The model overpredicts the Good / Excellent group and underpredicts the Very Poor group, indicating regression toward the centre of the WQI range. Station-level errors are supplied in `outputs/error_by_station_B.csv`; the highest-RMSE stations have only two test observations each, so station-specific estimates should be interpreted cautiously.

### 4.4 Summary of findings

1. Full-measurement models can closely reproduce the constructed WQI.
2. Field-only prediction is substantially more difficult.
3. The station-mean baseline outperforms the recorded Set B regression models.
4. The model's apparent performance depends strongly on the feature set and validation design.

### Relationship to the submitted PDF

The submitted PDF is the fixed coursework document and is not modified by this repository update. Some classification scores differ slightly from the current year-blocked pipeline results. The current code and its regenerated output files should be used when demonstrating execution; the PDF's separate random 70/30 split result is not reproduced by the present script and should not be claimed as a current pipeline output.


### Secondary random-split diagnostic

A separate 70/30 random split (seed 42) is executed for the Set B Random Forest, with shuffled five-fold grid-search validation inside the random training partition. This is included only to reproduce the report's illustrative comparison; the chronological 2013–2014 holdout remains the primary evaluation. Random splitting can place records from the same station and different years in both partitions, making the result more optimistic. The final regenerated diagnostic gives Random Forest RMSE 9.879 and R² 0.624 (train n=1,100; test n=472). The exact value is taken from `outputs/random_split_comparison.csv` and is not used as the primary result.

## 5. Discussion

The contrast between Set A and Set B is the main result. A model supplied with all variables used to construct the target can learn the mathematical relationship between those variables and the WQI. This is useful as a computational approximation, but it does not demonstrate that the model can independently assess water quality.

The field-only experiment is more relevant to a low-cost monitoring scenario. Its performance is weaker, and the station-mean baseline performs better on the recorded holdout. This suggests that location-specific historical averages contain substantial predictive information and that a more complex model is not automatically better.

The station overlap between training and testing also matters: the experiment does not establish transfer to previously unseen monitoring stations. A future evaluation should hold out entire stations, in addition to holding out future years.

## 6. Limitations

- The target is formula-derived from the same measurements used in Set A.
- Most test stations also appear in training; unseen-station generalisation is untested.
- Set B regression does not beat the station-mean baseline.
- WQI weights and standards are assumptions and affect the index.
- The year-only timestamp does not support seasonal analysis.
- The dataset is secondary-source data and may contain measurement and metadata inconsistencies.
- Feature importance is predictive, not causal.
- The index should not replace official laboratory testing or regulatory assessment.

## 7. Conclusion

The project demonstrates a reproducible workflow for calculating a weighted-arithmetic WQI and comparing supervised-learning models. Full-feature models achieve very high scores, but those results are largely explained by the target's mathematical construction. The reduced field-measurement task remains challenging, and the station-mean baseline is stronger than the tested Set B regression models. These findings motivate stricter generalisation tests, more representative monitoring data, and transparent reporting of assumptions and baselines.

## 8. Future work

- Evaluate generalisation by holding out complete monitoring stations.
- Add grouped temporal validation and compare it with the current chronological split.
- Perform sensitivity analysis over WQI standards and category thresholds.
- Investigate uncertainty intervals and calibrated classification probabilities.
- Incorporate additional independent predictors such as rainfall, flow, land use, and catchment characteristics where reliable data are available.
- Validate model predictions against laboratory measurements not used to construct the WQI.

## References

1. UN-Water. *Progress on Wastewater Treatment – 2024 Update*. United Nations, 2024. https://www.unwater.org/publications/progress-wastewater-treatment-2024-update
2. United Nations Department of Economic and Social Affairs. *Sustainable Development Goal 14, Target 14.1 and Indicator 14.1.1*. https://sdgs.un.org/goals/goal14
3. Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970). A water quality index—do we dare? *Water & Sewage Works*, 117(10), 339–343.
4. Central Pollution Control Board. *Water Quality Criteria: Primary Water Quality Criteria for Designated Best Use Classes*. Government of India. https://cpcb.nic.in/water-quality-criteria/
5. Khoi, D. N., Quan, N. T., Linh, D. Q., Nhi, P. T. T., et al. (2022). Using machine learning models for predicting the water quality index in the La Buong River, Vietnam. *Water*, 14(10), 1552. https://doi.org/10.3390/w14101552
6. Masood, A., Niazkar, M., Zakwan, M., & Piraei, R. (2023). A machine learning-based framework for water quality index estimation in the Southern Bug River. *Water*, 15(20), 3543. https://doi.org/10.3390/w15203543
7. Singh, S., Das, A., & Sharma, P. (2024). Predictive modeling of water quality index (WQI) classes in Indian rivers: insights from the application of multiple machine learning (ML) models on a decennial dataset. *Stochastic Environmental Research and Risk Assessment*. https://doi.org/10.1007/s00477-024-02741-z
8. Aldrees, A., Khan, M., Taha Bakheit, A. B., & Ali, M. (2024). Evaluation of water quality indexes with novel machine learning and SHapley Additive exPlanation (SHAP) approaches. *Journal of Water Process Engineering*, 58, 104789. https://doi.org/10.1016/j.jwpe.2024.104789
9. Mohseni, U., Pande, C. B., Pal, S. C., & Alshehri, F. (2024). Prediction of weighted arithmetic water quality index for urban water quality using ensemble machine learning model. *Chemosphere*, 352, 141393. https://doi.org/10.1016/j.chemosphere.2024.141393
10. Mo, Y., Xu, J., Liu, C., Wu, J., & Chen, D. (2024). Assessment and prediction of Water Quality Index (WQI) by seasonal key water parameters in a coastal city: application of machine learning models. *Environmental Monitoring and Assessment*, 196(11), 1008. https://doi.org/10.1007/s10661-024-13209-6
11. Yang, S., Liang, R., Chen, J., Wang, Y., & Li, K. (2024). Estimating the water quality index based on interpretable machine learning models. *Water Science & Technology*, 89(5), 1340–1356. https://doi.org/10.2166/wst.2024.068
12. Ejaz, U., Khan, S. M., Jehangir, S., Ahmad, Z., Abdullah, A., Iqbal, M., Khalid, N., Nazir, A., & Svenning, J.-C. (2024). Monitoring the industrial waste polluted stream—integrated analytics and machine learning for water quality index assessment. *Journal of Cleaner Production*, 141877. https://doi.org/10.1016/j.jclepro.2024.141877
13. Qu, J., Ma, J., Luo, Y., & Ma, Y. (2025). Application of simplified machine learning models to assess and predict lake water quality. *Water Research X*, 29, 100445. https://doi.org/10.1016/j.wroa.2025.100445
14. Han, Z., Zhang, S., He, L., & Zhu, B. (2025). Predicting and investigating water quality index by robust machine learning methods. *Journal of Environmental Management*, 381, 125156. https://doi.org/10.1016/j.jenvman.2025.125156
15. Choudhary, R., Kumar, A., C., P., Naik, M. M., Choudhury, M., & Khan, N. A. (2025). Predicting water quality index using stacked ensemble regression and SHAP based explainable artificial intelligence. *Scientific Reports*, 15, 31139. https://doi.org/10.1038/s41598-025-09463-4
16. Hridoy, M. A. A. M., Shawkat, A. I., Bordin, C., Acharjee, M. R., Masood, A., Baki, A. O., & Mamun, M. A. (2025). Advanced machine learning models for accurate water quality classification and WQI prediction: implications for aquatic disease risk management. *Science of the Total Environment*, 1008, 180965. https://doi.org/10.1016/j.scitotenv.2025.180965
17. Rahman, A., Syeed, M. M. M., Karim, M. R., Fatema, K., et al. (2025). An optimized ensemble ML-WQI model for reliable water quality prediction by minimizing the eclipsing and ambiguity issues. *Applied Water Science*, 15, 113. https://doi.org/10.1007/s13201-025-02450-0
18. Mahmoudian Boroujerd, L., Shakerdonyavi, A., Asadgol, Z., Donyavi, T., Tavakoli, A., Abdossalami Asl, Y., Tabatabaee Jabali, S. M., Bahari, N., Shahouri, M., & Safarpour, E. (2025). Machine learning-based prediction of drinking water quality index in Western Tehran using KAN, MLP, and traditional models. *Environmental Monitoring and Assessment*, 197, 1065. https://doi.org/10.1007/s10661-025-14500-w
19. Mengistu, T. D., Chung, I.-M., & Chang, S. W. (2026). Machine learning for water quality prediction and uncertainty assessment. *Physics and Chemistry of the Earth, Parts A/B/C*, 143, 104319. https://doi.org/10.1016/j.pce.2026.104319
20. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
21. Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
22. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
23. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192–213. https://doi.org/10.1016/j.ins.2011.12.028
24. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

## AI-use statement

An AI assistant was used for literature discovery, language editing, review of the manuscript against the assignment feedback, and software-review suggestions. The experiment code was executed separately; reported numerical results are to be copied from generated machine-readable outputs. The author reviewed the methodology, assumptions, source material, code changes, and final manuscript and takes responsibility for the submitted content.

---

**Reproducibility note:** The final submission must be generated from a fresh clone after the updated experiment completes. Every numerical value in the manuscript, including CV mean ± SD, bootstrap intervals, random-split diagnostics, selected hyperparameters, and Set B error-analysis summaries, must be copied from the generated outputs rather than typed independently.
