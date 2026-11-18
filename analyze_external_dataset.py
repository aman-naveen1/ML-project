"""Analyse annual station-level min/max river-water-quality records.

This is a coverage and parameter-bound WQI analysis, not sample-level ML
validation. The source reports annual extrema; independently combining
parameter extrema gives a conservative envelope, not observed joint samples
or a statistical confidence interval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import WQI_STANDARDS

# Reuse the exact standards used by the primary project WQI implementation.
STANDARDS = WQI_STANDARDS
WEIGHTS = {name: 1.0 / value for name, value in STANDARDS.items()}
_WEIGHT_TOTAL = sum(WEIGHTS.values())
WEIGHTS = {name: value / _WEIGHT_TOTAL for name, value in WEIGHTS.items()}

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
    """Load, normalize and validate source rows without imputing measurements."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.rename(columns=COLUMN_MAP)

    required = {"station_code", "location", "state", "year"}
    required |= {
        f"{parameter}_{bound}"
        for parameter in STANDARDS
        for bound in ("min", "max")
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"External dataset is missing required columns: {missing_columns}"
        )

    source_rows = len(frame)
    numeric = ["year"] + [
        f"{parameter}_{bound}"
        for parameter in STANDARDS
        for bound in ("min", "max")
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(
            frame[column].replace("-", np.nan), errors="coerce"
        )
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    missing_year = frame["year"].isna()
    frame = frame.loc[~missing_year].copy()
    frame["year"] = frame["year"].astype(int)

    invalid = pd.Series(False, index=frame.index)
    # Both endpoints must be physically valid and ordered.
    invalid |= (frame["ph_min"] < 0) | (frame["ph_min"] > 14)
    invalid |= (frame["ph_max"] < 0) | (frame["ph_max"] > 14)
    for parameter in STANDARDS:
        lower, upper = frame[f"{parameter}_min"], frame[f"{parameter}_max"]
        invalid |= lower.notna() & upper.notna() & (lower > upper)
        if parameter != "ph":
            invalid |= (lower < 0) | (upper < 0)

    invalid_rows = int(invalid.sum())
    frame = frame.loc[~invalid].copy()

    parameter_complete = {
        parameter: int(
            frame[[f"{parameter}_min", f"{parameter}_max"]].notna().all(axis=1).sum()
        )
        for parameter in STANDARDS
    }
    complete_mask = frame[
        [f"{parameter}_{bound}" for parameter in STANDARDS
         for bound in ("min", "max")]
    ].notna().all(axis=1)

    stats = {
        "source_rows": int(source_rows),
        "rows_missing_year": int(missing_year.sum()),
        "invalid_bound_rows_excluded": invalid_rows,
        "retained_rows": int(len(frame)),
        "complete_wqi_rows": int(complete_mask.sum()),
        "incomplete_wqi_rows": int((~complete_mask).sum()),
        "parameter_complete_rows": parameter_complete,
        "years": sorted(int(year) for year in frame["year"].unique()),
        "unique_stations": int(frame["station_code"].nunique()),
    }
    return frame, stats


def q_positive(value: pd.Series, standard: float) -> pd.Series:
    """Monotone increasing project sub-index for pollutant parameters."""
    return (100.0 * value / standard).clip(0, 100)


def calculate_wqi_interval(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate conservative lower/upper WQI bounds from parameter extrema.

    The formula and standards match run_experiments.calculate_wqi. Because
    annual minima/maxima may occur on different dates, the result is an
    outer parameter-bound envelope rather than a WQI observed at one time.
    """
    low = pd.Series(0.0, index=frame.index)
    high = pd.Series(0.0, index=frame.index)

    pmin, pmax = frame["ph_min"], frame["ph_max"]
    crosses_ideal = (pmin <= 7.0) & (pmax >= 7.0)
    nearest_distance = pd.concat(
        [(pmin - 7.0).abs(), (pmax - 7.0).abs()], axis=1
    ).min(axis=1)
    farthest_distance = pd.concat(
        [(pmin - 7.0).abs(), (pmax - 7.0).abs()], axis=1
    ).max(axis=1)
    ph_low = (100.0 * nearest_distance / (STANDARDS["ph"] - 7.0)).clip(0, 100)
    ph_low = ph_low.mask(crosses_ideal, 0.0)
    ph_high = (100.0 * farthest_distance / (STANDARDS["ph"] - 7.0)).clip(0, 100)
    low += WEIGHTS["ph"] * ph_low
    high += WEIGHTS["ph"] * ph_high

    # DO sub-index decreases as DO increases: reverse the endpoints.
    do_low = (100.0 * (14.6 - frame["do_max"]) / (14.6 - STANDARDS["do"])).clip(0, 100)
    do_high = (100.0 * (14.6 - frame["do_min"]) / (14.6 - STANDARDS["do"])).clip(0, 100)
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
    bins = [-np.inf, 50, 75, np.inf]
    labels = ["Good / Excellent", "Poor", "Very Poor"]
    result["category_lower_bound"] = pd.cut(
        low, bins=bins, labels=labels, right=True
    ).astype("string")
    result["category_upper_bound"] = pd.cut(
        high, bins=bins, labels=labels, right=True
    ).astype("string")
    result["category_is_ambiguous"] = (
        result["category_lower_bound"].notna()
        & result["category_upper_bound"].notna()
        & (result["category_lower_bound"] != result["category_upper_bound"])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize CPCB-derived annual station-level WQI envelopes."
    )
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

    valid = result.dropna(subset=["wqi_lower_bound", "wqi_upper_bound"])
    yearly = result.assign(
        has_wqi= result["wqi_lower_bound"].notna()
        & result["wqi_upper_bound"].notna()
    ).groupby("year").agg(
        source_records=("station_code", "size"),
        complete_wqi_records=("has_wqi", "sum"),
        stations=("station_code", "nunique"),
        median_wqi_lower=("wqi_lower_bound", "median"),
        median_wqi_upper=("wqi_upper_bound", "median"),
        median_interval_width=("wqi_interval_width", "median"),
        ambiguous_category_share=("category_is_ambiguous", "mean"),
    ).reset_index()
    yearly["incomplete_wqi_records"] = (
        yearly["source_records"] - yearly["complete_wqi_records"]
    )
    yearly.to_csv(output / "external_yearly_summary.csv", index=False)

    lines = [
        "External dataset: CPCB-derived annual station-level min/max records",
        f"Rows in source file: {stats['source_rows']}",
        f"Rows missing year (excluded): {stats['rows_missing_year']}",
        f"Rows excluded for invalid bounds: {stats['invalid_bound_rows_excluded']}",
        f"Rows retained: {stats['retained_rows']}",
        f"Rows with complete WQI inputs: {stats['complete_wqi_rows']}",
        f"Rows with incomplete WQI inputs: {stats['incomplete_wqi_rows']}",
        f"Unique stations in retained rows: {stats['unique_stations']}",
        f"Years in retained rows: {stats['years']}",
        "Complete rows by parameter: " + str(stats["parameter_complete_rows"]),
        f"Median WQI envelope width (complete rows): {valid['wqi_interval_width'].median():.3f}",
        f"Share of complete rows with category ambiguity: {valid['category_is_ambiguous'].mean():.3f}",
        "",
        "Method: same project standards, inverse-standard weights, sub-index",
        "ratings and category cutoffs as run_experiments.py.",
        "Caution: annual parameter extrema may occur on different dates. The",
        "calculated values are an outer parameter-bound envelope, not observed",
        "joint-sample WQI values or a statistical confidence interval.",
        "This dataset is not used as row-level external ML validation.",
        "Missing measurements are not imputed; incomplete rows have missing WQI bounds.",
    ]
    (output / "external_dataset_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
