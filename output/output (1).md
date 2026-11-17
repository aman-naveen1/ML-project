# Output: WQI_Estimation1.ipynb

Notebook run on `DOC-20261117-WA0000.csv` (copied to the `water_dataX.csv` name the notebook expects). All outputs below were produced by executing the notebook code unchanged.

# Water Quality Index Estimation

This notebook uses water quality data to calculate the Water Quality Index and predict water quality using machine learning.

## Step 1: Upload the dataset

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists("water_dataX.csv"):
    from google.colab import files
    uploaded = files.upload()
```

## Step 2: Read the data

```python
data = pd.read_csv("water_dataX.csv", encoding="latin-1")
data.head()
```

**Output:**
```text
  ï»¿STATION CODE                                              LOCATIONS        STATE  Temp D.O. (mg/l)   PH CONDUCTIVITY (ç¥hos/cm) B.O.D. (mg/l) NITRATENAN N+ NITRITENANN (mg/l) FECAL COLIFORM (MPN/100ml) TOTAL COLIFORM (MPN/100ml)Mean  year
0            1393                   DAMANGANGA AT D/S OF MADHUBAN, DAMAN  DAMAN & DIU  30.6         6.7  7.5                      203           NAN                              0.1                         11                             27  2014
1            1399  ZUARI AT D/S OF PT. WHERE KUMBARJRIA CANAL JOINS, GOA          GOA  29.8         5.7  7.2                      189             2                              0.2                       4953                           8391  2014
2            1475                                    ZUARI AT PANCHAWADI          GOA  29.5         6.3  6.9                      179           1.7                              0.1                       3243                           5330  2014
3            3181                            RIVER ZUARI AT BORIM BRIDGE          GOA  29.7         5.8  6.9                       64           3.8                              0.5                       5382                           8443  2014
4            3182                           RIVER ZUARI AT MARCAIM JETTY          GOA  29.5         5.8  7.3                       83           1.9                              0.4                       3428                           5500  2014
```

## Step 3: Clean the data

```python
data.columns = ["code", "location", "state", "temp", "do", "ph", "cond", "bod", "nitrate", "fc", "tc", "year"]

columns = ["temp", "do", "ph", "cond", "bod", "nitrate", "fc", "tc"]

for col in columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.drop_duplicates()
data.loc[(data["ph"] < 0) | (data["ph"] > 14), "ph"] = np.nan

print("Rows and columns:", data.shape)
print(data.isnull().sum())
```

**Output:**
```text
Rows and columns: (1990, 12)
code          0
location      0
state         0
temp         91
do           30
ph           96
cond         24
bod          42
nitrate     224
fc          315
tc          131
year          0
dtype: int64
```

## Step 4: Calculate the Water Quality Index

```python
standards = {
    "ph": 8.5,
    "do": 5,
    "bod": 3,
    "cond": 2250,
    "nitrate": 10,
    "fc": 500,
    "tc": 500
}

weights = {}
for col in standards:
    weights[col] = 1 / standards[col]

total = sum(weights.values())

for col in weights:
    weights[col] = weights[col] / total

water = data.dropna(subset=list(standards.keys())).copy()

quality = pd.DataFrame(index=water.index)
quality["ph"] = 100 * abs(water["ph"] - 7) / (8.5 - 7)
quality["do"] = 100 * (14.6 - water["do"]) / (14.6 - 5)

for col in ["bod", "cond", "nitrate", "fc", "tc"]:
    quality[col] = 100 * water[col] / standards[col]

quality = quality.clip(0, 100)

water["wqi"] = 0
for col in standards:
    water["wqi"] = water["wqi"] + quality[col] * weights[col]

water["category"] = np.where(water["wqi"] <= 50, "Good", np.where(water["wqi"] <= 75, "Poor", "Very Poor"))

water[["year", "wqi", "category"]].head()
```

**Output:**
```text
   year        wqi category
1  2014  56.837223     Poor
2  2014  49.599117     Good
3  2014  70.625484     Poor
4  2014  56.390819     Poor
5  2014  51.975859     Poor
```

## Step 5: Explore the WQI values

```python
print(water["wqi"].describe())
print(water["category"].value_counts())

water["wqi"].hist(bins=20)
plt.xlabel("Water Quality Index")
plt.ylabel("Number of records")
plt.title("Distribution of WQI")
plt.show()
```

**Output:**
```text
count    1572.000000
mean       56.959928
std        16.263114
min        21.632069
25%        42.989041
50%        54.301556
75%        72.853448
max        90.628289
Name: wqi, dtype: float64
category
Good         640
Poor         618
Very Poor    314
Name: count, dtype: int64
```

![figures/fig1.png](figures/fig1.png)

## Step 6: Prepare the data for machine learning

```python
features = ["temp", "do", "ph", "cond", "bod", "nitrate", "fc", "tc"]

water = water.sort_values("year")

train = water[water["year"] <= 2012]
test = water[water["year"] >= 2013]

X_train = train[features]
X_test = test[features]
y_train = train["wqi"]
y_test = test["wqi"]

print("Training records:", len(train))
print("Testing records:", len(test))
```

**Output:**
```text
Training records: 1111
Testing records: 461
```

## Step 7: Train regression models

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    "Linear Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(random_state=42)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=100, random_state=42))
}

regression_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    regression_results[name] = [mae, rmse, r2]
    print(name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))
    print()
```

**Output:**
```text
Linear Regression
MAE: 18.45
RMSE: 122.01
R2 Score: -63.95

Decision Tree
MAE: 2.0
RMSE: 2.93
R2 Score: 0.96

Random Forest
MAE: 1.2
RMSE: 1.85
R2 Score: 0.99
```

## Step 8: Compare regression models

```python
results = pd.DataFrame(regression_results, index=["MAE", "RMSE", "R2 Score"]).T
results
```

**Output:**
```text
                         MAE        RMSE   R2 Score
Linear Regression  18.452744  122.005273 -63.952369
Decision Tree       2.002582    2.927466   0.962604
Random Forest       1.197338    1.853535   0.985009
```

## Step 9: Predict water quality category

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

encoder = LabelEncoder()
water["category_number"] = encoder.fit_transform(water["category"])

train = water[water["year"] <= 2012]
test = water[water["year"] >= 2013]

X_train = train[features]
X_test = test[features]
y_train = train["category_number"]
y_test = test["category_number"]

classifiers = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(random_state=42)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=100, random_state=42))
}

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(name)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
    print(classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0))
```

**Output:**
```text
Logistic Regression
Accuracy: 0.74
              precision    recall  f1-score   support

        Good       0.71      0.90      0.80       167
        Poor       0.73      0.66      0.69       205
   Very Poor       0.82      0.62      0.71        89

    accuracy                           0.74       461
   macro avg       0.76      0.72      0.73       461
weighted avg       0.74      0.74      0.73       461

Decision Tree
Accuracy: 0.87
              precision    recall  f1-score   support

        Good       0.92      0.89      0.90       167
        Poor       0.84      0.88      0.86       205
   Very Poor       0.87      0.82      0.84        89

    accuracy                           0.87       461
   macro avg       0.88      0.86      0.87       461
weighted avg       0.87      0.87      0.87       461

Random Forest
Accuracy: 0.9
              precision    recall  f1-score   support

        Good       0.93      0.93      0.93       167
        Poor       0.86      0.93      0.89       205
   Very Poor       0.95      0.78      0.85        89

    accuracy                           0.90       461
   macro avg       0.91      0.88      0.89       461
weighted avg       0.90      0.90      0.90       461
```

## Step 10: Check which features are important

```python
forest = models["Random Forest"]
forest_model = forest.named_steps["randomforestregressor"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": forest_model.feature_importances_
})

importance = importance.sort_values("Importance", ascending=False)
print(importance)

plt.bar(importance["Feature"], importance["Importance"])
plt.xticks(rotation=45)
plt.xlabel("Water quality parameters")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
```

**Output:**
```text
   Feature  Importance
4      bod    0.929827
2       ph    0.029998
1       do    0.020516
5  nitrate    0.011714
3     cond    0.003852
7       tc    0.001616
6       fc    0.001340
0     temp    0.001136
```

![figures/fig2.png](figures/fig2.png)

## Step 11: Actual and predicted WQI

```python
best_model = models["Random Forest"]
predictions = best_model.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("Actual WQI")
plt.ylabel("Predicted WQI")
plt.title("Actual vs Predicted WQI")
plt.show()
```

![figures/fig3.png](figures/fig3.png)

## Conclusion

The notebook calculates the Water Quality Index from the available water quality parameters. It then uses regression models to estimate WQI values and classification models to predict the water quality category. The model scores can be used to compare how well the different machine learning models perform.
