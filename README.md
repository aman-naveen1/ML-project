# Water Quality Index (WQI) Estimation for Indian Rivers

A machine learning project that calculates a Water Quality Index (WQI) from river-monitoring measurements and explores how well machine learning models can estimate water quality.

**Course project:** Machine Learning  
**Department:** Computer Science, Ramanujan College, University of Delhi  
**Author:** Aman Naveen  
**Supervisor:** Mr Nikhil Kumar Rajput

[Open in Google Colab](https://colab.research.google.com/drive/1pJZ5Vrkecg026V1rlan2kHZqa10tQvUG?usp=sharing)

---

## Project overview

Water quality is influenced by several physical and chemical measurements. This project uses monitoring data from Indian rivers to:

- Clean and prepare the raw measurements.
- Calculate a weighted arithmetic Water Quality Index.
- Group observations into water-quality categories.
- Train regression models to estimate numerical WQI values.
- Train classification models to predict water-quality categories.
- Compare model performance and inspect feature importance.

The dataset contains **1,991 records** from 2003–2014. After duplicate removal, validation of pH values, and filtering for complete WQI inputs, **1,572 observations** are used for WQI analysis.

## Dataset

The project uses the Kaggle dataset [Indian Water Quality Data](https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data).

The raw CSV is included in this repository as `DOC-20261117-WA0000.csv`. The script reads this file directly from the same folder as `wqi_estimation1.py`; no renaming or copying is required.

The measurements used include:

| Parameter | Description |
|---|---|
| Temperature | Water temperature |
| DO | Dissolved oxygen |
| pH | Acidity or alkalinity |
| Conductivity | Electrical conductivity |
| BOD | Biochemical oxygen demand |
| Nitrate | Nitrate concentration |
| FC | Fecal coliform |
| TC | Total coliform |

## Methodology

### 1. Data cleaning

- Convert measurement columns to numeric values.
- Remove duplicate records.
- Treat pH values outside the 0–14 range as missing.
- Remove records missing any parameter required for WQI calculation.

### 2. WQI calculation

The project calculates a weighted arithmetic index using pH, dissolved oxygen, BOD, conductivity, nitrate, fecal coliform, and total coliform.

Parameter weights are derived from the selected standards and normalised so that they sum to one. Parameter quality ratings are clipped to the 0–100 range before being combined into the final WQI.

The categories used in the implementation are:

| WQI range | Category |
|---|---|
| 0–50 | Good |
| Above 50–75 | Poor |
| Above 75 | Very Poor |

These categories and parameter standards are assumptions used for this project; they should not be treated as an official regulatory assessment.

### 3. Model training and evaluation

The checked-in script uses a chronological split:

- **Training data:** up to and including 2012
- **Testing data:** 2013 onwards

It trains regression models to predict WQI and classification models to predict the corresponding category. Missing feature values are imputed using the median. Standardisation is applied in the pipelines for models that need it.

Regression is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

Classification is evaluated using accuracy and a per-class classification report.

### 4. Feature importance and visualisation

The script also plots the WQI distribution, feature importance from the random forest regressor, and actual versus predicted WQI values.

## Results and observations

The accompanying `results.txt` contains a broader experiment comparing multiple models and two feature sets. Its reported chronological test results (2013–2014) include:

| Experiment / model | Feature set | RMSE | R² |
|---|---|---:|---:|
| Gradient Boosting | All parameters (A) | 0.822 | 0.997 |
| Random Forest | Field measurements (B) | 11.329 | 0.440 |
| Station-mean baseline | Baseline | 7.930 | 0.726 |
| Mean baseline | Baseline | 15.187 | -0.006 |

**How to interpret this:** the all-parameter experiment can closely reproduce the constructed WQI, in part because the target is calculated from those same inputs. With field measurements alone, prediction is more difficult. In the reported comparison, the station-mean baseline also outperforms the best field-measurement model, so the latter result should not be presented as strong predictive performance.

The report identifies BOD as the most influential parameter in the all-parameter feature set. This is consistent with its large contribution to the index formula.

> The figures above are transcribed from the repository's `results.txt`; they are not claimed as output from the simpler checked-in `wqi_estimation1.py` script.

## Repository structure

```text
ML-project/
├── DOC-20261117-WA0000.csv   # Raw water-quality dataset
├── wqi_estimation1.py        # Main Python implementation
├── requirements.txt          # Python dependencies
├── results.txt               # Experiment results and metrics
├── colablink.md              # Google Colab notebook link
└── output/                   # Output directory
```

## Getting started

### Requirements

- Python 3.10 or later
- pip

### Run locally

Clone the repository:

```bash
git clone https://github.com/aman-naveen1/ML-project.git
cd ML-project
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The dataset is already included in the repository. Keep `DOC-20261117-WA0000.csv` in the repository folder and run:



```bash
python wqi_estimation1.py
```

If the CSV is not found, the script offers a Google Colab upload fallback. For the Colab workflow, open the [notebook](https://colab.research.google.com/drive/1pJZ5Vrkecg026V1rlan2kHZqa10tQvUG?usp=sharing) and follow its cells.

## Dependencies

The project uses:

- NumPy
- pandas
- scikit-learn
- Matplotlib

See [requirements.txt](requirements.txt) for the version constraints.

## Limitations

- The WQI is calculated from selected parameter standards and weights; different standards or weighting methods can produce different index values.
- Coliform parameters contribute a relatively small share of the calculated index under the selected weighting scheme.
- The dataset does not include month-level information.
- In the reported experiment, 99.8% of test-station records came from stations also present in the training data. Results therefore do not establish that the models generalise to entirely unseen rivers or stations.
- A model that predicts a formula-derived WQI accurately is not necessarily measuring water quality independently of that formula.

## Acknowledgements

An AI assistant (Claude, Anthropic) was used to support literature scanning, report drafting, and code development. The author ran the code, reviewed the results, and takes responsibility for the project.
