# Data sources and provenance

## Primary project dataset

- **File:** `DOC-20261117-WA0000.csv`
- **Upstream source:** [Indian Water Quality Data on Kaggle](https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data)
- **Role in this project:** sample-level historical dataset used by `run_experiments.py`.
- **Caveat:** this is a secondary compilation; see `DATA_DICTIONARY.md` for the field mapping and cleaning rules.

## Additional annual station-level dataset

- **File in this repository:** `data/external/RiverWaterQualityOverYears.csv`
- **Source type:** external annual station-level water-quality dataset.
- **CPCB reference:** [Central Pollution Control Board — NWMP data](https://cpcb.nic.in/nwmp-data-2023/), identified in the external dataset's accompanying documentation as the underlying monitoring programme.
- **Retrieved for this project:** 18 November 2026.
- **Recorded coverage in the checked-in file:** 10,006 records in 2012–2014 and 2017–2023. The file contains annual station-level minima and maxima, not individual sample measurements.

The external dataset documentation identifies CPCB's National Water Quality Monitoring Programme (NWMP) as its underlying source. The checked-in CSV is treated as an external source dataset; this project has not independently reconciled every row against CPCB's original downloadable annual files. Accordingly, this repository does not claim that the CSV was downloaded directly from CPCB or independently verified against the original CPCB records.

This project uses the supplied annual min/max data for coverage and interval analysis. It does not use externally derived contamination labels or impute missing measurements.

## Appropriate use and limitations

The additional file is used for data-coverage reporting and a parameter-bound WQI envelope. Annual minima and maxima for different parameters may have occurred on different dates. Combining them independently produces an outer parameter-bound envelope, not an observed joint-sample WQI, a confidence interval, or independent row-level validation of the machine-learning models.

The two datasets have different temporal and measurement granularity. Their overlapping years must not be treated as directly matched observations without station/date-level reconciliation.

## Retrieval and reproducibility note

The CPCB portal link is provided as a reference to the monitoring programme identified by the external dataset documentation. The checked-in CSV is retained as the external source file used by this analysis; no automated refresh is performed. If the file changes, re-check its schema, record count, year coverage, and provenance before replacing this copy.
