# Data sources and provenance

## Primary project dataset

- **File:** `DOC-20261117-WA0000.csv`
- **Upstream source:** [Indian Water Quality Data on Kaggle](https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data)
- **Role in this project:** sample-level historical dataset used by `run_experiments.py`.
- **Caveat:** this is a secondary compilation; see `DATA_DICTIONARY.md` for the field mapping and cleaning rules.

## Additional annual station-level dataset

- **File in this repository:** `data/external/RiverWaterQualityOverYears.csv`
- **Upstream file:** [RiverWaterQualityOverYears.csv](https://github.com/KunalLatkar/RiverWaterQualityDataset/blob/main/RiverWaterQualityOverYears.csv)
- **Upstream repository and documentation:** [KunalLatkar/RiverWaterQualityDataset](https://github.com/KunalLatkar/RiverWaterQualityDataset)
- **Original authority portal identified by the upstream repository:** [CPCB NWMP data](https://cpcb.nic.in/nwmp-data-2023/)
- **Retrieved for this project:** 18 November 2026.
- **Recorded coverage in the checked-in file:** 10,006 records in 2012–2014 and 2017–2023. The file contains annual station-level minima and maxima, not individual sample measurements.

The upstream repository states that this dataset was gathered from the CPCB portal. This project attributes the file to that repository and records the CPCB portal as the upstream authority identified by its author. The checked-in CSV was obtained from the repository's raw annual min/max file; this project has not independently reconciled every row against CPCB's original downloadable annual files. Therefore, describe it as **CPCB-derived via the upstream repository**, not as a directly downloaded or independently verified CPCB extract.

The upstream repository also publishes a separately preprocessed CSV and describes imputation, outlier treatment, and a derived `Contaminated` label. This project uses the raw annual min/max CSV instead. It does not import that derived label or impute missing measurements.

## Appropriate use and limitations

The additional file is used for data-coverage reporting and a parameter-bound WQI envelope. Annual minima and maxima for different parameters may have occurred on different dates. Combining them independently produces an outer parameter-bound envelope, not an observed joint-sample WQI, a confidence interval, or independent row-level validation of the machine-learning models.

The two datasets have different temporal and measurement granularity. Their overlapping years must not be treated as directly matched observations without station/date-level reconciliation.

## Retrieval and reproducibility note

The source URLs and retrieval date above identify the upstream materials used when this repository copy was added. The checked-in CSV preserves the raw annual min/max source file used by the analysis; no automated refresh is performed. If the upstream file changes, re-check its schema, record count, year coverage, and provenance before replacing this copy.
