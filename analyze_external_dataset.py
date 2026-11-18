"""Analyse the CPCB-derived 2012–2023 station/year min–max dataset.

This is an external temporal/coverage analysis, not row-level ML validation.
The source contains annual minima and maxima, not individual sample records.
The script therefore calculates an interval for WQI rather than inventing a
single observation by averaging the bounds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STANDARDS = {
    "ph": 8.5,
    "do": 5.0,
    "bod": 3.0,
    "cond": 2250.0,
    "nitrate": 10.0,
    "fc": 500.0,
    "tc": 500.0,
}
WEIGHTS = {k: 1.0 / v for k, v in STANDARDS.items()}
WEIGHT_TOTAL = sum(WEIGHTS.values())
WEIGHTS = {k: v / WEIGHT_TOTAL for k, v in WEIGHTS.items()}

COLUMN_MAP = {
    "station code": "station_code",
    "monitoring location": "location",
    "state": "state",
    "temperature (°c) min": "temp_min",
    "temperature (°c) max": "temp_max",
    "dissolved oxygen (mg/l) min": "do_min",
    "dissolved oxygen (mg/l) max": "do_max",
    "ph min": "ph_min",
    "ph max": "ph_max",
    "conductivity (µmho/cm) min": "cond_min",
    "conductivity (µmho/cm) max": "cond_max",
    "bod (mg/l) min": "bod_min",
    "bod (mg/l) max": "bod_max",
    "nitrate n (mg/l) min": "nitrate_min",
    "nitrate n (mg/l) max": "nitrate_max",
    "fecal coliform (mpn/100ml) min": "fc_min",
    "fecal coliform (mpn/100ml) max": "fc_max",
    "total coliform (mpn/100ml) min": "tc_min",
    "total coliform (mpn/100ml) max": "tc_max",
    "fecal streptococci (mpn/100ml) min": "fs_min",
    "fecal streptococci (mpn/100ml) max": "fs_max",
    "year": "year",
}


def load_external_data(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    frame = frame.rename(columns=COLUMN_MAP)
    required = {"station_code", "location", "state", "year"}
    required |= {f"{p}_{bound}" for p in STANDARDS for bound in ("min", "max")}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"External dataset is missing required columns: {missing}")

    numeric = [c for c in required if c not in {"station_code", "location", "state"}]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col].replace("-", np.nan), errors="coerce")
    frame = frame.dropna(subset=["year"]).copy()
    frame["year"] = frame["year"].astype(int)

    invalid_ph = (frame["ph_min"] < 0) | (frame["ph_max"] > 14) | (
        frame["ph_min"] > frame["ph_max"]
    )
    invalid_bounds = pd.Series(False, index=frame.index)
    for parameter in STANDARDS:
        invalid_bounds |= (
            frame[f"{parameter}_min"].notna()
            & frame[f"{parameter}_max"].notna()
            & (frame[f"{parameter}_min"] > frame[f"{parameter}_max"])
        )
    invalid = invalid_ph | invalid_bounds
    stats = {
        "source_rows": int(len(frame)),
        "invalid_bound_rows_excluded": int(invalid.sum()),
        "years": sorted(int(x) for x in frame["year"].unique()),
        "unique_stations": int(frame["station_code"].nunique()),
    }
    frame = frame.loc[~invalid].copy()
    return frame, stats


def q_positive(value: pd.Series, standard: float) -> pd.Series:
    return (100.0 * value / standard).clip(0, 100)


def calculate_wqi_interval(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate conservative WQI bounds from parameter-wise annual bounds."""
    low = pd.Series(0.0, index=frame.index)
    high = pd.Series(0.0, index=frame.index)

    # pH rating is V-shaped around ideal pH 7.
    pmin, pmax = frame["ph_min"], frame["ph_max"]
    ph_low = pd.Series(0.0, index=frame.index)
    ph_low = (100.0 * np.where(
        (pmin <= 7.0) & (pmax >= 7.0), 0.0,
        np.minimum((pmin - 7.0).abs(), (pmax - 7.0).abs()) / 1.5
    )).clip(0, 100)
    ph_high = (100.0 * np.maximum((pmin - 7.0).abs(), (pmax - 7.0).abs()) / 1.5).clip(0, 100)
    low += WEIGHTS["ph"] * ph_low
    high += WEIGHTS["ph"] * ph_high

    # Higher DO lowers its WQI sub-index, so reverse the endpoints.
    do_low = (100.0 * (14.6 - frame["do_max"]) / (14.6 - 5.0)).clip(0, 100)
    do_high = (100.0 * (14.6 - frame["do_min"]) / (14.6 - 5.0)).clip(0, 100)
    low += WEIGHTS["do"] * do_low
    high += WEIGHTS["do"] * do_high

    for parameter in ("bod", "cond", "nitrate", "fc", "tc"):
        low += WEIGHTS[parameter] * q_positive(
            frame[f"{parameter}_min"], STANDARDS[parameter]
        )
        high += WEIGHTS[parameter] * q_positive(
            frame[f"{parameter}_max"], STANDARDS[parameter]
        )

    result = frame[["station_code", "location", "state", "year"]].copy()
    result["wqi_lower_bound"] = low
    result["wqi_upper_bound"] = high
    result["wqi_interval_width"] = high - low
    result["category_lower_bound"] = pd.cut(
        low, [-np.inf, 50, 75, np.inf],
        labels=["Good / Excellent", "Poor", "Very Poor"]
    ).astype(str)
    result["category_upper_bound"] = pd.cut(
        high, [-np.inf, 50, 75, np.inf],
        labels=["Good / Excellent", "Poor", "Very Poor"]
    ).astype(str)
    result["category_is_ambiguous"] = (
        result["category_lower_bound"] != result["category_upper_bound"]
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/external/RiverWaterQualityOverYears.csv"
    )
    parser.add_argument("--output", default="outputs/external_dataset")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data, stats = load_external_data(Path(args.data))
    result = calculate_wqi_interval(data)
    result.to_csv(output / "external_wqi_intervals.csv", index=False)

    yearly = result.groupby("year").agg(
        records=("station_code", "size"),
        stations=("station_code", "nunique"),
        median_wqi_lower=("wqi_lower_bound", "median"),
        median_wqi_upper=("wqi_upper_bound", "median"),
        median_interval_width=("wqi_interval_width", "median"),
        ambiguous_category_share=("category_is_ambiguous", "mean"),
    ).reset_index()
    yearly.to_csv(output / "external_yearly_summary.csv", index=False)

    complete = result.dropna(subset=["wqi_lower_bound", "wqi_upper_bound"])
    summary = [
        "External dataset: CPCB-derived annual station-level min/max records",
        f"Rows in source: {stats['source_rows']}",
        f"Rows excluded for invalid bounds: {stats['invalid_bound_rows_excluded']}",
        f"Rows with a WQI interval: {len(complete)}",
        f"Unique stations in source: {stats['unique_stations']}",
        f"Years in source: {stats['years']}",
        f"Median WQI interval width: {complete['wqi_interval_width'].median():.3f}",
        f"Share with category ambiguity: {complete['category_is_ambiguous'].mean():.3f}",
        "",
        "Caution: the file contains annual minima/maxima, not sample-level records.",
        "WQI bounds are computed from the reported parameter bounds; they are not",
        "a statistical confidence interval and are not directly comparable to",
        "sample-level model predictions from the original dataset.",
    ]
    (output / "external_dataset_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
