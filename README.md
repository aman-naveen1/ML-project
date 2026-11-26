# Water Quality Index Estimation for Indian Rivers

**Machine Learning course project | SDG 6: Clean Water and Sanitation | SDG 14: Life Below Water**

A reproducible study of river-water quality in India. The project calculates a weighted-arithmetic Water Quality Index (WQI), compares regression and classification models, tests whether a reduced set of field measurements can estimate WQI, and documents where the models do and do not generalise.

> This repository is being organised as a research-style submission: documented data provenance, explicit assumptions, reproducible experiments, validation tests, machine-readable results, figures, and a report. Model scores are not presented as proof of real-world pollution measurement.

## 1. Research questions

1. How does the weighted-arithmetic WQI vary across the available Indian river-monitoring observations?
2. How accurately can machine-learning models reproduce the calculated WQI and its project-defined categories?
3. How much predictive information is lost when only quick field measurements are available?
4. Do the models outperform simple mean and station-based baselines?
5. What limitations arise from formula-derived targets, repeated monitoring stations, and the available time resolution?

## 2. Dataset and provenance

- **Dataset:** Indian Water Quality Data, originally published on Kaggle.
- **Repository copy:** `DOC-20261117-WA0000.csv`.
- **Raw size:** 1,991 records and 12 columns.
- **Coverage:** 2003–2014.
- **Encoding:** Latin-1.
- **Source fields:** station code, location, state, temperature, dissolved oxygen, pH, conductivity, BOD, nitrate/nitrite nitrogen, faecal coliform, total coliform, and year.

The analysis assigns stable project column names after reading the source CSV. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for field mapping, units, use in the models, and cleaning rules.

The dataset is a secondary compilation. Units and source definitions are retained as documented; they should be checked against the original monitoring authority before operational use.

See [DATA_SOURCES.md](DATA_SOURCES.md) for source URLs, retrieval date, attribution boundaries, and the provenance caveats for the additional annual station-level dataset.

## 3. WQI definition and assumptions

The index uses seven measurements: pH, dissolved oxygen (DO), biochemical oxygen demand (BOD), conductivity, nitrate/nitrite nitrogen, faecal coliform, and total coliform.

For parameter (i):

`Qi = 100 × (Vi − Videal,i) / (Si − Videal,i)`

`Wi = K / Si,  where K = 1 / Σ(1 / Si)`

`WQI = Σ(Wi × Qi)`

The implementation uses ideal pH 7, ideal DO 14.6 mg/L, and the project standards documented in the methodology. Quality ratings are clipped to 0–100 before aggregation. These choices materially affect the index.

| Parameter | Project standard |
|---|---:|
| pH | 8.5 |
| DO | 5 mg/L |
| BOD | 3 mg/L |
| Conductivity | 2,250 µmhos/cm |
| Nitrate + nitrite nitrogen | 10 mg/L |
| Faecal coliform | 500 MPN/100 mL |
| Total coliform | 500 MPN/100 mL |

**Important:** nitrate and faecal-coliform limits are project assumptions, not claims that these exact values are CPCB designated-use standards. The computed WQI is an educational/research index, not an official regulatory classification.

| WQI | Project category |
|---:|---|
| ≤ 50 | Good / Excellent |
| > 50 to 75 | Poor |
| > 75 | Very Poor |

## 4. Data preparation

The reproducible pipeline:

1. Reads the checked-in CSV using Latin-1 encoding.
2. Assigns the 12 documented project column names.
3. Converts measurement fields to numeric values; non-numeric markers become missing.
4. Removes duplicate rows.
5. Treats pH values outside 0–14 as invalid.
6. Excludes rows missing any WQI input from WQI calculation.
7. Sorts observations by year.
8. Fits imputers and scalers inside model pipelines, using training folds only.

The experiment retained 1,572 complete WQI observations in the previously recorded run. The exact counts are regenerated on each run and written to `outputs/run_summary.txt`.

## 5. Experimental design

### Chronological holdout

- **Training:** observations through 2012.
- **Testing:** observations from 2013–2014.

Hyperparameter selection uses expanding, year-blocked validation folds within the training period. The held-out 2013–2014 observations are not used for model selection.

### Feature sets

- **Set A — all measurements:** temperature, DO, pH, conductivity, BOD, nitrate, faecal coliform, total coliform.
- **Set B — field-only measurements:** temperature, DO, pH, conductivity.

Set A includes measurements that are themselves inputs to the WQI formula. Its high scores therefore primarily show how well a model can approximate a constructed mathematical index; they do not establish independent water-quality sensing.

### Models

**Regression:** Linear Regression, Ridge Regression, k-NN, RBF Support Vector Regression, Decision Tree, Random Forest, Gradient Boosting.

**Classification:** Logistic Regression, k-NN, Decision Tree, Random Forest, Gradient Boosting.

**Baselines:** training-mean and station-mean for regression; majority-class and station-majority for classification.

### Evaluation

- Regression: MAE, RMSE, R².
- Classification: accuracy, macro-F1, confusion matrix, per-class precision/recall/F1.
- Model interpretation: permutation feature importance on the held-out test set.
- Reproducibility: fixed random seeds, explicit preprocessing pipelines, automated unit tests, and saved outputs.

## 6. Verified results

The following values are from the verified GitHub Actions run (workflow run 53, 26 November 2026) using the current `run_experiments.py` pipeline. These are the repository's current reproducible results.

### Dataset split

| Quantity | Count |
|---|---:|
| Raw records | 1,991 |
| Complete WQI records | 1,572 |
| Training records (through 2012) | 1,111 |
| Test records (2013–2014) | 461 |

### Regression

| Model | Set A RMSE | Set A R² | Set B RMSE | Set B R² |
|---|---:|---:|---:|---:|
| Mean baseline | 15.187 | -0.006 | 15.187 | -0.006 |
| Station-mean baseline | 7.930 | 0.726 | 7.930 | 0.726 |
| Linear Regression | 8.509 | 0.684 | 13.406 | 0.216 |
| Ridge Regression | 8.502 | 0.685 | 13.393 | 0.217 |
| k-NN | 6.371 | 0.823 | 11.397 | 0.433 |
| RBF SVR | 2.515 | 0.972 | 11.775 | 0.395 |
| Decision Tree | 3.012 | 0.960 | 12.640 | 0.303 |
| Random Forest | 1.851 | 0.985 | 11.324 | 0.440 |
| Gradient Boosting | 0.823 | 0.997 | 11.342 | 0.439 |

### CV uncertainty and bootstrap intervals

| Model | Set A CV RMSE ± SD | Set A test RMSE | Set A bootstrap 95% CI | Set B CV RMSE ± SD | Set B test RMSE | Set B bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 9.058 ± 0.575 | 8.509 | 7.728–9.355 | 12.988 ± 1.062 | 13.406 | 12.442–14.485 |
| Ridge Regression | 9.045 ± 0.572 | 8.502 | 7.728–9.338 | 12.983 ± 1.063 | 13.393 | 12.434–14.463 |
| k-NN | 6.099 ± 0.822 | 6.371 | 5.775–7.013 | 10.736 ± 0.712 | 11.397 | 10.573–12.289 |
| RBF SVR | 3.402 ± 0.530 | 2.515 | 2.114–2.987 | 10.988 ± 0.815 | 11.775 | 10.928–12.631 |
| Decision Tree | 3.377 ± 0.518 | 3.012 | 2.631–3.419 | 11.707 ± 0.379 | 12.640 | 11.769–13.494 |
| Random Forest | 2.417 ± 0.446 | 1.851 | 1.493–2.266 | 10.661 ± 0.422 | 11.324 | 10.553–12.116 |
| Gradient Boosting | 1.159 ± 0.287 | 0.823 | 0.729–0.921 | 10.601 ± 0.510 | 11.342 | 10.550–12.088 |

These values are generated from the current experiment code. The SD is the fold-to-fold standard deviation of forward-chaining CV RMSE; the CI is the percentile bootstrap 95% interval for held-out test RMSE.

### Classification

| Model | Set A accuracy | Set A macro-F1 | Set B accuracy | Set B macro-F1 |
|---|---:|---:|---:|---:|
| Majority class | 0.362 | 0.177 | 0.362 | 0.177 |
| Station-majority | 0.731 | 0.740 | 0.731 | 0.740 |
| Logistic Regression | 0.820 | 0.802 | 0.564 | 0.575 |
| k-NN | 0.768 | 0.777 | 0.642 | 0.657 |
| Decision Tree | 0.816 | 0.809 | 0.599 | 0.608 |
| Random Forest | 0.913 | 0.905 | 0.616 | 0.627 |
| Gradient Boosting | 0.933 | 0.930 | 0.657 | 0.661 |

**Interpretation:** Set A Gradient Boosting reaches RMSE 0.823 and R² 0.997, but this is expected to be optimistic as the target is calculated from the same measurements. Set B is the more realistic reduced-measurement task. Its best regression result (RMSE 11.324) is worse than the station-mean baseline (RMSE 7.930). This baseline comparison is a central finding, not something to hide.

The current pipeline result is the authoritative result for this repository; older PDF/report numbers from earlier evaluation procedures are superseded.

## Submitted PDF and reproducibility note

The current repository pipeline and the revised report are the source of truth for the verified results. The final PDF should be regenerated from this same verified run before submission; do not mix older PDF values with the current repository outputs.


### Selected hyperparameters

The selected settings from `outputs/selected_hyperparameters.csv` are:

| Set | Model | Selected parameters |
|---|---|---|
| A | Ridge | alpha=10 |
| A | k-NN | n_neighbors=5; weights=distance |
| A | RBF SVR | C=100; epsilon=0.5 |
| A | Decision Tree | max_depth=12; min_samples_leaf=5 |
| A | Random Forest | max_depth=None; min_samples_leaf=1 |
| A | Gradient Boosting | learning_rate=0.1; max_depth=2; n_estimators=300 |
| B | Ridge | alpha=10 |
| B | k-NN | n_neighbors=31; weights=distance |
| B | RBF SVR | C=10; epsilon=2 |
| B | Decision Tree | max_depth=5; min_samples_leaf=5 |
| B | Random Forest | max_depth=5; min_samples_leaf=1 |
| B | Gradient Boosting | learning_rate=0.1; max_depth=3; n_estimators=100 |

Linear Regression has no tunable grid-search parameters.

### Set B error analysis

For the best Set B regression model (Random Forest), category-level test errors are: Good / Excellent n=167, MAE 8.636, RMSE 11.742; Poor n=205, MAE 9.782, RMSE 11.767; Very Poor n=89, MAE 7.389, RMSE 9.305. The model overpredicts Good / Excellent observations and underpredicts Very Poor observations. Station-level results are in `outputs/error_by_station_B.csv`.

### Secondary random-split diagnostic

The pipeline also runs a reproducible 70/30 random split on the complete WQI dataset (seed 42) for the Set B Random Forest. The verified diagnostic gives RMSE 9.879 and R² 0.624 (train n=1,100; test n=472). This is diagnostic only: random splitting can place observations from the same stations and years on both sides and is therefore more optimistic than the chronological holdout.

## 7. Limitations and threats to validity

1. **Formula-derived target:** Set A learns a target calculated from the same inputs. Its high performance is not independent validation of water quality.
2. **Station overlap:** most test stations also occur in training. The experiment does not establish performance on entirely unseen stations.
3. **Weak field-only regression:** the station-mean baseline outperforms the best Set B regressor on the recorded holdout.
4. **Standards sensitivity:** WQI weights and ratings depend on chosen standards, including project assumptions for some parameters.
5. **Temporal resolution:** year-level observations do not support seasonal analysis.
6. **Secondary data quality:** source units, missingness, station metadata, and historical measurement practices may vary.
7. **No causal claims:** model feature importance and cross-sectional patterns do not establish that changing a parameter causes a particular change in water quality.

The project is for coursework and exploratory analysis. It should not replace laboratory testing, field validation, or official CPCB assessments.

## 8.1 Submission-revision checklist

The reproducibility pipeline now supports the reviewer-requested uncertainty and error-analysis outputs:

- regression tables include forward-chaining CV RMSE mean and SD;
- regression test RMSE receives a percentile bootstrap 95% interval;
- selected regression and classification hyperparameters are saved to `outputs/selected_hyperparameters.csv`;
- Set B Random Forest errors are summarised by station and WQI category;
- the random 70/30 comparison remains a diagnostic only and is never used as the primary result;
- the manuscript uses **set A** and **set B** consistently;
- the report records the 17 October 2026 access date;
- the report contains an explicit operational target: beat the station-mean baseline.

For submission, the manuscript/PDF must use the same values as the verified generated outputs. Do not reuse values from an older PDF.

## 8. Reproducibility and quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/aman-naveen1/ML-project.git
cd ML-project
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py --data DOC-20261117-WA0000.csv
```

Generated files are written to `outputs/`. The workflow in `.github/workflows/tests.yml` runs the tests and full experiment and uploads the output directory as a GitHub Actions artifact.

## 9. Python scripts: what each file does

There are three Python scripts in the repository, with different roles:

| File | Purpose | When to use it |
|---|---|---|
| [`wqi_estimation1.py`](wqi_estimation1.py) | The original, Colab-derived workflow. It loads and cleans the main CSV, calculates WQI and categories, trains a smaller set of regression and classification models, and displays basic feature-importance and actual-vs-predicted plots. | Use this to follow the original step-by-step project workflow. It is not the full reproducible experiment pipeline. |
| [`run_experiments.py`](run_experiments.py) | The main reproducible experiment pipeline. It calculates WQI, compares the full regression and classification model sets across Set A and Set B, evaluates baselines with chronological validation, and saves metrics, diagnostics, plots, and a run summary. | **Use this for the primary experiments and for regenerating the main project results.** |
| [`analyze_external_dataset.py`](analyze_external_dataset.py) | Analyses the separate annual station-level dataset containing minimum and maximum measurements. It checks data completeness and calculates parameter-bound WQI envelopes and yearly summaries. It reuses the standards from `run_experiments.py`. | Use this only for the additional 2012–2023 external-dataset analysis. It does not train or externally validate the ML models. |

In short: **`wqi_estimation1.py` = original workflow; `run_experiments.py` = main ML experiments; `analyze_external_dataset.py` = external annual-data analysis.**

## 10. Repository layout

```text
ML-project/
├── DOC-20261117-WA0000.csv
├── wqi_estimation1.py
├── run_experiments.py
├── analyze_external_dataset.py
├── data/
│   └── external/
│       └── RiverWaterQualityOverYears.csv
├── tests/
│   ├── test_wqi.py
│   └── test_external_dataset.py
├── .github/workflows/
│   └── tests.yml
├── DATA_DICTIONARY.md
├── results.txt
├── requirements.txt
├── README.md
└── outputs/                 # generated experiment artefacts
```

## 11. References

1. Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970). A water quality index—do we dare? *Water & Sewage Works*, 117(10), 339–343.
2. Central Pollution Control Board. [Water quality criteria](https://cpcb.nic.in/water-quality-criteria/).
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
5. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

## Academic integrity

This repository documents the data, assumptions, methods, outputs, and limitations used in the project. Any submitted report should be reviewed by the student, accurately reflect the final executed code, and follow the institution's citation and AI-use rules.


## External recent-period dataset (2012–2023)

A second, independent repository copy is included at
`data/external/RiverWaterQualityOverYears.csv`. It is an external annual station-level dataset whose accompanying documentation
identifies the Central Pollution Control Board (CPCB) National Water Quality
Monitoring Programme (NWMP) as the underlying monitoring source. See the
[CPCB NWMP data portal](https://cpcb.nic.in/nwmp-data-2023/) for programme
reference. The checked-in file contains annual station-level minimum/maximum
values rather than individual sample measurements.

This source has 10,006 data records, 22 columns and observations in 2012–2014
and 2017–2023 (no 2015 or 2016 records in this file). It reports annual
minimum/maximum values rather than individual sample-level observations.
Accordingly, `analyze_external_dataset.py` computes a parameter-bound WQI
envelope and yearly coverage summaries. It imports the same WQI standards and
weights used by `run_experiments.py`, rather than maintaining a second set of
thresholds. Invalid/reversed bounds are excluded and completeness is reported
by parameter and year; missing values are not imputed. Incomplete rows retain
missing WQI bounds and are excluded from WQI summaries. The envelope is not a
confidence interval or a sample-level WQI label: annual extrema for different
parameters may have occurred on different dates.

Run:

```bash
python analyze_external_dataset.py
```

Outputs are written to `outputs/external_dataset/`. The analysis deliberately
does not fill missing source measurements with imputed values and does not
train or claim to externally validate the original row-level ML models.
Overlapping years (2012–2014) are useful for checking broad consistency, but
the different annual min/max aggregation means they are not directly
comparable to the original dataset's individual records. The external source's derived labels are not used as independent ground truth.
