# Water Quality Index Estimation for Indian Rivers

A Machine Learning course project that calculates a weighted-arithmetic Water Quality Index (WQI) from Indian river-monitoring measurements, then compares regression and classification models.

**Author:** Aman Naveen  
**Department:** Computer Science, Ramanujan College, University of Delhi  
**Supervisor:** Mr Nikhil Kumar Rajput  
**Course:** Machine Learning

## Project objectives

- Clean river-monitoring measurements and handle invalid pH values.
- Calculate a WQI from seven water-quality parameters.
- Classify observations as Good / Excellent, Poor, or Very Poor.
- Compare regression and classification algorithms.
- Test whether quick field measurements can estimate WQI when laboratory measurements are unavailable.
- Identify the parameters most influential to model predictions.

## Dataset

The project uses the Kaggle dataset [Indian Water Quality Data](https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data). The original CSV is included as `DOC-20261117-WA0000.csv` (1,991 records, 2003–2014).

The WQI calculation requires pH, dissolved oxygen, BOD, conductivity, nitrate, fecal coliform and total coliform. Rows missing one of these required measurements are excluded from WQI analysis. Temperature is retained as a model feature but is not part of the WQI formula, so missing temperature values are imputed inside the model pipeline.

## WQI method

The project uses the weighted arithmetic method:

```
Qi = 100 × (Vi - Videal,i) / (Si - Videal,i)
Wi = K / Si, where K = 1 / sum(1 / Si)
WQI = sum(Wi × Qi)
```

The pH rating uses the absolute deviation from the ideal pH of 7. Dissolved oxygen uses an ideal value of 14.6 mg/L. Quality ratings are clipped to 0–100 before aggregation.

| Parameter | Standard used | Weight (approx.) |
|---|---:|---:|
| pH | 8.5 | 15.6% |
| Dissolved oxygen | 5 mg/L | 26.5% |
| BOD | 3 mg/L | 44.1% |
| Conductivity | 2,250 µmhos/cm | 0.06% |
| Nitrate + nitrite-N | 10 mg/L | 13.2% |
| Fecal coliform | 500 MPN/100 mL | 0.26% |
| Total coliform | 500 MPN/100 mL | 0.26% |

The nitrate and fecal-coliform standards are project assumptions because the cited CPCB designated-use criteria do not specify those exact limits. The resulting index is a research calculation, not an official CPCB compliance rating.

| WQI | Project category |
|---:|---|
| ≤ 50 | Good / Excellent |
| > 50 to 75 | Poor |
| > 75 | Very Poor |

## Experimental design

The full experiment is implemented in `run_experiments.py`.

Two feature sets are compared:

- **Set A — all measurements:** temperature, dissolved oxygen, pH, conductivity, BOD, nitrate, fecal coliform and total coliform.
- **Set B — quick field measurements:** temperature, dissolved oxygen, pH and conductivity.

The data is sorted by year and split chronologically: training uses records through 2012; testing uses 2013–2014. Hyperparameters are selected using five-fold forward-chaining cross-validation on the training data. Imputation and standardisation are fitted within model pipelines to avoid using test-set statistics.

### Models

**Regression:** Linear Regression, Ridge Regression, k-NN, RBF Support Vector Regression, Decision Tree, Random Forest and Gradient Boosting.

**Classification:** Logistic Regression, k-NN, Decision Tree, Random Forest and Gradient Boosting.

The script also compares a training-mean baseline and a station-mean baseline. Regression is evaluated with MAE, RMSE and R². Classification is evaluated with accuracy and macro-F1.

## Run the full experiment

Python 3.10 or later is recommended.

```bash
git clone https://github.com/aman-naveen1/ML-project.git
cd ML-project
pip install -r requirements.txt
python run_experiments.py --data DOC-20261117-WA0000.csv
```

The script creates an `outputs/` folder containing:
- WQI-enriched dataset
- Regression and classification metric tables for both feature sets
- Confusion matrix and classification report
- Permutation feature-importance tables
- Observed-versus-predicted and yearly WQI plots
- A run summary with cleaning counts, selected hyperparameters and evaluation results

## Results reported in the current study

The submitted report records 1,572 complete WQI observations: 1,111 training records and 461 test records.

| Experiment | Model | RMSE | R² |
|---|---|---:|---:|
| All parameters (A) | Gradient Boosting | 0.82 | 0.997 |
| Field measurements (B) | Random Forest | 11.36 | 0.44 |
| Station-mean baseline | — | 7.93 | 0.73 |
| Mean baseline | — | 15.19 | -0.006 |

For classification with all parameters, Gradient Boosting is reported at 93.9% accuracy and 0.937 macro-F1.

These figures are the results documented in the accompanying report/results record. Run `run_experiments.py` to regenerate and independently verify the full experiment in the current repository version before treating any metric as reproduced.

## Interpretation and limitations

- The very high Set A score mainly shows that flexible models can learn the formula used to construct WQI; it is not independent proof that the model can measure river pollution.
- Set B is the more practical test, but its best reported model does not outperform the station-mean baseline.
- BOD has the largest WQI weight (about 44.1%) and is the strongest reported feature in Set A.
- The test set largely contains stations already seen during training. The experiment therefore does not establish generalisation to entirely new stations.
- The dataset has year-level rather than month-level dates, so seasonal patterns cannot be studied.
- The chosen WQI standards and weights affect the final index, especially the low contribution of coliform measurements.

The project is intended as a decision-support and educational experiment. It should not replace laboratory testing or official water-quality assessments.

## Repository files

- `DOC-20261117-WA0000.csv` — source dataset
- `run_experiments.py` — full reproducible model comparison
- `results.txt` — previously recorded detailed metrics
- `requirements.txt` — Python dependencies
- `output/` — notebook output notes and figures

## References

1. Brown, R. M., McClelland, N. I., Deininger, R. A. and Tozer, R. G. (1970). “A water quality index—do we dare?” *Water & Sewage Works*, 117(10), 339–343.
2. Central Pollution Control Board. “Water quality criteria (primary water quality criteria for designated-best-use classes).” https://cpcb.nic.in/water-quality-criteria/
3. Pedregosa, F. et al. (2011). “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research*, 12, 2825–2830.
4. Breiman, L. (2001). “Random forests.” *Machine Learning*, 45, 5–32.
5. Friedman, J. H. (2001). “Greedy function approximation: a gradient boosting machine.” *Annals of Statistics*, 29(5), 1189–1232.
