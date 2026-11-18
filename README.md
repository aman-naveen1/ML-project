# Water Quality Index (WQI) Estimation for Indian Rivers

Machine Learning course project, Department of Computer Science, Ramanujan College, University of Delhi.
Author: Aman Naveen. Supervisor: Mr Nikhil Kumar Rajput.

Repository: https://github.com/aman-naveen1/ML-project

This project computes a weighted arithmetic Water Quality Index (WQI) from seven river-monitoring
parameters (standards based on CPCB criteria) and asks how well classical ML models can estimate it,
using a chronological train/test split (train 2005-2012, test 2013-2014).

## Main findings (from the report)
- With **all parameters** (feature set A), gradient boosting reproduces the index almost exactly
  (RMSE 0.82, R2 = 0.997). This mostly shows that the models can learn the formula.
- With **field measurements only** (set B: temperature, pH, dissolved oxygen, conductivity), the best model
  (random forest) reaches R2 of about 0.44, which beats the mean baseline but not the
  station-mean baseline (R2 = 0.73).
- BOD carries 44% of the index weight and is by far the most important input.

## Data
Kaggle dataset "Indian Water Quality Data" (anbarivan):
https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data
The data file is included in this repository (1,991 rows, 2003-2014). The notebook reads it as
`water_dataX.csv`, so **rename the file in the repo (`DOC-20261117-WA0000.csv`) to `water_dataX.csv`**.
The file has no WQI column, so the WQI is computed in the code. It is read with `encoding="latin-1"`.

## Code
| File | Purpose |
|---|---|
| `WQI_Estimation1.py` | Main code: step-by-step notebook (cleaning, WQI, chronological split, regression, classification, feature importance, actual vs predicted plot) |
| `water_dataX.csv` | Raw dataset |
| `requirements.txt` | Python dependencies |
| `colablink.md` | Link to the Colab notebook |

Colab notebook: https://colab.research.google.com/drive/1pJZ5Vrkecg026V1rlan2kHZqa10tQvUG?usp=sharing

## How to run
**Colab:** open the Colab link above, run all cells, and upload `water_dataX.csv` when asked.

**Locally:**
```bash
git clone https://github.com/aman-naveen1/ML-project
cd ML-project
pip install -r requirements.txt jupyter
jupyter notebook WQI_Estimation1.ipynb     # keep water_dataX.csv in the same folder
```
The notebook is short and runs in well under a minute. Random seed is 42.

Note: the notebook is a simplified walkthrough (linear regression, decision tree and random forest;
logistic regression, decision tree and random forest; no baselines or hyper-parameter tuning). The
fuller experiments in the report (7 regressors, 5 classifiers, baselines, time-series cross-validation)
were run with `run_experiments.py`.

## Method summary
1. **Cleaning:** "NAN"/blank values become missing, one duplicate row removed, 89 pH values outside 0-14 set to missing.
2. **WQI:** weighted arithmetic index from pH, DO, BOD, conductivity, nitrate, fecal and total coliform, with the quality
   rating of each parameter limited to 0-100. Only the 1,572 complete records are kept.
   Categories: WQI <= 50 good/excellent, 51-75 poor, above 75 very poor.
3. **Split:** by year (train 2005-2012, test 2013-2014).
4. **Preprocessing:** median imputation and standardisation inside a pipeline.
5. **Evaluation:** MAE, RMSE, R2 (regression); accuracy and per-class report (classification); feature importance.

## Limitations
The WQI is computed by us from a method with uneven weights (coliforms contribute under 1%), and the nitrate and
fecal-coliform standards are our assumptions. The data have no month, 99.8% of test stations were seen in training,
so results do not show performance on new rivers.

## AI use
An AI assistant (Claude, Anthropic) helped with literature scanning, drafting the report and the code. The author ran
the code, checked the results and takes responsibility for the content.
