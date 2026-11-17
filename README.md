# Water Quality Index (WQI) Estimation for Indian Rivers

Machine Learning course project, Department of Computer Science, Ramanujan College, University of Delhi.
Author: Aman Naveen. Supervisor: Mr Nikhil Kumar Rajput.

This project computes a weighted arithmetic Water Quality Index (WQI) from seven river-monitoring
parameters (standards based on CPCB criteria) and asks how well classical ML models can estimate it,
using a chronological train/test split (train 2005-2012, test 2013-2014).

## Main findings
- With **all parameters** (feature set A), gradient boosting reproduces the index almost exactly
  (RMSE 0.82, R2 = 0.997). This mostly shows that the models can learn the formula.
- With **field measurements only** (set B: temperature, pH, dissolved oxygen, conductivity), the best model
  (random forest) reaches R2 of about 0.44, which beats the mean baseline but not the
  station-mean baseline (R2 = 0.73).
- BOD carries 44% of the index weight and is by far the most important input.

## Data
Kaggle dataset "Indian Water Quality Data" (anbarivan):
https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data
The file used is `water_dataX.csv` (1,991 rows, 2003-2014), included in this repository.
The file has no WQI column, so the WQI is computed in the code. Read it with `encoding="latin-1"`.

## Repository contents
| File | Purpose |
|---|---|
| `water_dataX.csv` | Raw dataset |
| `run_experiments.py` | Full pipeline: cleaning, WQI, baselines, 7 regressors, 5 classifiers, time-series CV, permutation importance. Regenerates every table and figure in the report |
| `WQI_Estimation1.ipynb` | Colab notebook: short, step-by-step walkthrough of the method |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated tables (`table4_regression.csv`, `table5_classification.csv`), figures (`fig*.png`) and `results.txt` |

Colab notebook: <PASTE YOUR COLAB LINK HERE>

## How to run
```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python run_experiments.py                            # uses water_dataX.csv in the current folder
# or: python run_experiments.py path/to/water_dataX.csv
```
Runtime is a few minutes (grid searches with 5-fold forward-chaining CV). Random seed is 42.
Results were produced with Python 3.12, scikit-learn 1.8.0 (report) and re-checked on scikit-learn 1.9.1;
small differences in the last digit of some tuned models can occur between scikit-learn versions.

## Method summary
1. **Cleaning:** "NAN"/blank values become missing, one duplicate row removed, 89 pH values outside 0-14 set to missing.
2. **WQI:** weighted arithmetic index from pH, DO, BOD, conductivity, nitrate, fecal and total coliform, with the quality
   rating of each parameter limited to 0-100. Only the 1,572 complete records are kept.
   Categories: WQI <= 50 good/excellent, 51-75 poor, above 75 very poor.
3. **Split:** by year (train 2005-2012, test 2013-2014). The test set is used once.
4. **Models:** linear, ridge, k-NN, SVR, decision tree, random forest, gradient boosting. Classifiers: logistic regression,
   k-NN, decision tree, random forest, gradient boosting. Baselines: mean / station-mean (regression), majority /
   station-majority (classification).
5. **Preprocessing inside the pipeline (fitted on training data only):** log(1+x) on skewed variables, median imputation, standardisation.
6. **Evaluation:** MAE, RMSE, R2 (regression); accuracy and macro-F1 (classification); permutation importance on the test set.

## Limitations
The WQI is computed by us from a method with uneven weights (coliforms contribute under 1%), and the nitrate and
fecal-coliform standards are our assumptions. The data have no month, 99.8% of test stations were seen in training,
so results do not show performance on new rivers.

## AI use
An AI assistant (Claude, Anthropic) helped with literature scanning, drafting the report and the code. The author ran
the code, checked the results and takes responsibility for the content.
