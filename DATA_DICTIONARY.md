# Dataset dictionary

Source file: `DOC-20261117-WA0000.csv`  
Source dataset: [Indian Water Quality Data — Kaggle](https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data)

The CSV contains 12 columns. The names below are the names assigned by the project after reading the source file.

| Project column | Source CSV field | Meaning | Unit / format |
|---|---|---|---|
| `station_code` | STATION CODE | Monitoring-station identifier | Code |
| `location` | LOCATIONS | Sampling-station location description | Text |
| `state` | STATE | Indian state / territory recorded in the dataset | Text |
| `temp` | Temp | Water temperature | °C (as represented in the source data) |
| `do` | D.O. (mg/l) | Dissolved oxygen | mg/L |
| `ph` | PH | Acidity / alkalinity | pH units |
| `cond` | CONDUCTIVITY | Electrical conductivity | µmhos/cm in source header |
| `bod` | B.O.D. (mg/l) | Biochemical oxygen demand | mg/L |
| `nitrate` | NITRATENAN N+ NITRITENANN (mg/l) | Nitrate + nitrite nitrogen measurement | mg/L |
| `fc` | FECAL COLIFORM (MPN/100ml) | Faecal coliform count | MPN/100 mL |
| `tc` | TOTAL COLIFORM (MPN/100ml)Mean | Total coliform count (mean field in source header) | MPN/100 mL |
| `year` | year | Observation year | YYYY |

## How the columns are used

- `station_code` is used to construct the station-mean baseline. It is not used as a model feature.
- `location` and `state` are retained as metadata and are not used as model features.
- The WQI calculation uses `ph`, `do`, `bod`, `cond`, `nitrate`, `fc` and `tc`.
- Set A uses all eight measurement features: `temp`, `do`, `ph`, `cond`, `bod`, `nitrate`, `fc` and `tc`.
- Set B uses the four field-measurement features: `temp`, `do`, `ph` and `cond`.
- `year` controls the chronological train/test split and year-blocked validation.

## Cleaning notes

- Measurement columns are converted to numeric values; non-numeric markers such as `NAN` become missing values.
- Duplicate rows are removed.
- pH values outside 0–14 are treated as invalid and set to missing.
- Rows missing any of the seven WQI inputs are excluded from WQI calculation.
- Missing model features such as temperature are imputed using the training-fold median inside the model pipeline.
