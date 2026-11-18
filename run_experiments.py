"""
Reproducible experiments for the Indian-rivers Water Quality Index project.

Run:
    python run_experiments.py --data DOC-20261117-WA0000.csv

The script writes metrics, confusion matrices and plots to ./outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_COLUMNS = [
    "station_code", "location", "state", "temp", "do", "ph", "cond",
    "bod", "nitrate", "fc", "tc", "year",
]
WQI_STANDARDS = {
    "ph": 8.5,
    "do": 5.0,
    "bod": 3.0,
    "cond": 2250.0,
    "nitrate": 10.0,
    "fc": 500.0,
    "tc": 500.0,
}
ALL_FEATURES = ["temp", "do", "ph", "cond", "bod", "nitrate", "fc", "tc"]
FIELD_FEATURES = ["temp", "do", "ph", "cond"]
LOG_FEATURES = ["cond", "bod", "nitrate", "fc", "tc"]
CLASS_NAMES = ["Good / Excellent", "Poor", "Very Poor"]


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(path, encoding="latin-1")
    if data.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_COLUMNS)} columns, found {data.shape[1]}. "
            "Check that the original Kaggle CSV is being used."
        )
    data.columns = RAW_COLUMNS
    for col in ALL_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data.drop_duplicates().copy()
    invalid_ph = (data["ph"] < 0) | (data["ph"] > 14)
    invalid_ph_count = int(invalid_ph.sum())
    data.loc[invalid_ph, "ph"] = np.nan
    data = data.dropna(subset=list(WQI_STANDARDS)).copy()
    data = data.dropna(subset=["year"]).sort_values(["year"]).reset_index(drop=True)
    return data, {
        "raw_rows": int(pd.read_csv(path, encoding="latin-1").shape[0]),
        "clean_complete_rows": int(len(data)),
        "invalid_ph_values": invalid_ph_count,
        "year_min": int(data["year"].min()),
        "year_max": int(data["year"].max()),
    }


def calculate_wqi(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    weights = {name: 1.0 / standard for name, standard in WQI_STANDARDS.items()}
    total = sum(weights.values())
    weights = {name: value / total for name, value in weights.items()}

    q = pd.DataFrame(index=data.index)
    q["ph"] = 100 * (data["ph"] - 7.0).abs() / (8.5 - 7.0)
    q["do"] = 100 * (14.6 - data["do"]) / (14.6 - 5.0)
    for name in ["bod", "cond", "nitrate", "fc", "tc"]:
        q[name] = 100 * data[name] / WQI_STANDARDS[name]
    q = q.clip(0, 100)

    result = data.copy()
    result["wqi"] = sum(q[name] * weights[name] for name in WQI_STANDARDS)
    result["category"] = pd.cut(
        result["wqi"],
        bins=[-np.inf, 50, 75, np.inf],
        labels=CLASS_NAMES,
        right=True,
    )
    return result, weights


def transform_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    for col in LOG_FEATURES:
        if col in result.columns:
            result[col] = np.log1p(result[col].clip(lower=0))
    return result


def make_pipeline(model, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def regression_specs():
    return {
        "Linear regression": (
            make_pipeline(LinearRegression(), scale=True), {}
        ),
        "Ridge regression": (
            make_pipeline(Ridge(), scale=True), {"model__alpha": [0.1, 1, 10, 100]}
        ),
        "k-NN": (
            make_pipeline(KNeighborsRegressor(), scale=True),
            {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]},
        ),
        "SVR (RBF)": (
            make_pipeline(SVR(kernel="rbf"), scale=True),
            {"model__C": [1, 10, 100], "model__epsilon": [0.5, 2]},
        ),
        "Decision tree": (
            make_pipeline(DecisionTreeRegressor(random_state=42)),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [5, 10, 20]},
        ),
        "Random forest": (
            make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1)),
            {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 10]},
        ),
        "Gradient boosting": (
            make_pipeline(GradientBoostingRegressor(random_state=42)),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [2, 3]},
        ),
    }


def classifier_specs():
    return {
        "Logistic regression": (
            make_pipeline(LogisticRegression(max_iter=2000), scale=True),
            {"model__C": [0.1, 1, 10]},
        ),
        "k-NN": (
            make_pipeline(KNeighborsClassifier(), scale=True),
            {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]},
        ),
        "Decision tree": (
            make_pipeline(DecisionTreeClassifier(random_state=42)),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [5, 10, 20]},
        ),
        "Random forest": (
            make_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1)),
            {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 10]},
        ),
        "Gradient boosting": (
            make_pipeline(GradientBoostingClassifier(random_state=42)),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [2, 3]},
        ),
    }


def station_baselines(train: pd.DataFrame, test: pd.DataFrame, target: str, classification=False):
    if classification:
        global_value = train[target].mode().iloc[0]
        station_values = train.groupby("station_code", dropna=True)[target].agg(
            lambda x: x.mode().iloc[0]
        )
    else:
        global_value = train[target].mean()
        station_values = train.groupby("station_code", dropna=True)[target].mean()
    return test["station_code"].map(station_values).fillna(global_value).to_numpy()


def make_year_based_splits(train):
    years = np.sort(train["year"].unique())
    validation_years = years[-5:]
    splits = []
    year_values = train["year"].to_numpy()
    for year in validation_years:
        train_indices = np.flatnonzero(year_values < year)
        validation_indices = np.flatnonzero(year_values == year)
        if len(train_indices) and len(validation_indices):
            splits.append((train_indices, validation_indices))
    if len(splits) < 2:
        raise ValueError("Not enough distinct training years for forward-chaining validation.")
    return splits


def run_regression(train, test, features, label, cv, report_lines, output_dir):
    X_train = transform_features(train, features)
    X_test = transform_features(test, features)
    y_train, y_test = train["wqi"], test["wqi"]

    mean_pred = np.full(len(test), y_train.mean())
    station_pred = station_baselines(train, test, "wqi")
    rows = []
    rows.append({"model": "Mean baseline", "feature_set": label,
                 "MAE": mean_absolute_error(y_test, mean_pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, mean_pred)),
                 "R2": r2_score(y_test, mean_pred), "CV_RMSE": np.nan})
    rows.append({"model": "Station-mean baseline", "feature_set": label,
                 "MAE": mean_absolute_error(y_test, station_pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, station_pred)),
                 "R2": r2_score(y_test, station_pred), "CV_RMSE": np.nan})

    fitted = {}
    for name, (estimator, grid) in regression_specs().items():
        search = GridSearchCV(
            estimator, grid, scoring="neg_root_mean_squared_error",
            cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        cv_rmse = -search.best_score_
        rows.append({"model": name, "feature_set": label,
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "R2": r2_score(y_test, pred), "CV_RMSE": cv_rmse})
        fitted[name] = search.best_estimator_
        report_lines.append(
            f"Regression {label} | {name}: best_params={search.best_params_}; "
            f"MAE={rows[-1]['MAE']:.3f}; RMSE={rows[-1]['RMSE']:.3f}; "
            f"R2={rows[-1]['R2']:.3f}; CV_RMSE={cv_rmse:.3f}"
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"regression_{label}.csv", index=False)

    best_name = result[result["model"].isin(fitted)].sort_values("RMSE").iloc[0]["model"]
    best_model = fitted[best_name]
    importance = permutation_importance(
        best_model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=20, random_state=42, n_jobs=-1
    )
    imp = pd.DataFrame({"feature": features, "rmse_increase": importance.importances_mean})
    imp = imp.sort_values("rmse_increase", ascending=False)
    imp.to_csv(output_dir / f"importance_{label}.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.barh(imp["feature"][::-1], imp["rmse_increase"][::-1])
    plt.xlabel("Increase in test RMSE when shuffled")
    plt.title(f"Permutation importance — set {label} ({best_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"permutation_importance_{label}.png", dpi=180)
    plt.close()
    return result, best_name, best_model, X_test, y_test


def run_classification(train, test, features, label, cv, report_lines, output_dir):
    X_train = transform_features(train, features)
    X_test = transform_features(test, features)
    y_train = train["category"].astype(str)
    y_test = test["category"].astype(str)

    majority = y_train.mode().iloc[0]
    majority_pred = np.repeat(majority, len(test))
    station_pred = station_baselines(
        train.assign(category=y_train), test, "category", classification=True
    )
    rows = [
        {"model": "Majority class", "feature_set": label,
         "accuracy": accuracy_score(y_test, majority_pred),
         "macro_f1": f1_score(y_test, majority_pred, average="macro", zero_division=0)},
        {"model": "Station-majority", "feature_set": label,
         "accuracy": accuracy_score(y_test, station_pred),
         "macro_f1": f1_score(y_test, station_pred, average="macro", zero_division=0)},
    ]
    fitted = {}
    for name, (estimator, grid) in classifier_specs().items():
        search = GridSearchCV(
            estimator, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        rows.append({"model": name, "feature_set": label,
                     "accuracy": accuracy_score(y_test, pred),
                     "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0)})
        fitted[name] = search.best_estimator_
        report_lines.append(
            f"Classification {label} | {name}: best_params={search.best_params_}; "
            f"accuracy={rows[-1]['accuracy']:.3f}; macro_F1={rows[-1]['macro_f1']:.3f}"
        )
        if name == "Gradient boosting":
            pd.DataFrame(
                confusion_matrix(y_test, pred, labels=CLASS_NAMES),
                index=CLASS_NAMES, columns=CLASS_NAMES
            ).to_csv(output_dir / f"confusion_matrix_{label}.csv")
            (output_dir / f"classification_report_{label}.txt").write_text(
                classification_report(y_test, pred, labels=CLASS_NAMES, zero_division=0)
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"classification_{label}.csv", index=False)
    return result


def run_random_split_comparison(water: pd.DataFrame, output_dir: Path) -> dict:
    """Secondary random 70/30 Set B comparison; chronological holdout stays primary."""
    train, test = train_test_split(water, test_size=0.30, random_state=42, shuffle=True)
    X_train = transform_features(train, FIELD_FEATURES)
    X_test = transform_features(test, FIELD_FEATURES)
    y_train, y_test = train["wqi"], test["wqi"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    estimator = make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
    grid = {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 10]}
    search = GridSearchCV(estimator, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    prediction = search.predict(X_test)
    mean_prediction = np.full(len(test), y_train.mean())
    station_prediction = station_baselines(train, test, "wqi")
    rows = []
    for name, predicted in [("Random Forest (Set B)", prediction), ("Mean baseline", mean_prediction), ("Station-mean baseline", station_prediction)]:
        rows.append({"model": name, "split": "random 70/30", "random_state": 42, "test_size": 0.30, "train_rows": len(train), "test_rows": len(test), "RMSE": np.sqrt(mean_squared_error(y_test, predicted)), "R2": r2_score(y_test, predicted)})
    pd.DataFrame(rows).to_csv(output_dir / "random_split_comparison.csv", index=False)
    return {"random_rf_rmse": float(rows[0]["RMSE"]), "random_rf_r2": float(rows[0]["R2"]), "random_rf_params": search.best_params_, "random_train_rows": len(train), "random_test_rows": len(test)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="DOC-20261117-WA0000.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(data_path, encoding="latin-1")
    plt.figure(figsize=(8, 4))
    missing_pct = raw_data.isna().mean().mul(100).sort_values(ascending=False)
    missing_pct.plot(kind="bar")
    plt.ylabel("Missing values (%)")
    plt.title("Missing values by source column")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=180)
    plt.close()

    data, cleaning = load_and_clean(data_path)
    water, weights = calculate_wqi(data)
    train = water[water["year"] <= 2012].copy()
    test = water[water["year"] >= 2013].copy()
    if train.empty or test.empty:
        raise ValueError("The chronological split requires training years <=2012 and test years >=2013.")

    cv = make_year_based_splits(train)
    random_result = run_random_split_comparison(water, output_dir)
    lines = [
        f"Raw rows: {cleaning['raw_rows']}",
        f"Complete WQI rows: {cleaning['clean_complete_rows']}",
        f"Invalid pH values set to missing: {cleaning['invalid_ph_values']}",
        f"Years: {cleaning['year_min']}–{cleaning['year_max']}",
        f"Train rows (<=2012): {len(train)}",
        f"Test rows (>=2013): {len(test)}",
        f"WQI mean={water['wqi'].mean():.3f}; SD={water['wqi'].std():.3f}; "
        f"min={water['wqi'].min():.3f}; max={water['wqi'].max():.3f}",
        f"Weights: {weights}",
        f"Category counts: {water['category'].value_counts().to_dict()}",
        "Secondary random 70/30 Set B comparison (seed=42; diagnostic only): "
        f"Random Forest RMSE={random_result['random_rf_rmse']:.3f}; R2={random_result['random_rf_r2']:.3f}; "
        f"train={random_result['random_train_rows']}; test={random_result['random_test_rows']}; "
        f"best_params={random_result['random_rf_params']}",
    ]

    water.to_csv(output_dir / "wqi_dataset.csv", index=False)

    correlation = water[ALL_FEATURES + ["wqi"]].corr(method="spearman")
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation.index)), correlation.index)
    plt.title("Spearman correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "spearman_correlation.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")
    stages = ["Raw CSV", "Cleaning", "WQI calculation", "Year split", "Model pipelines", "Evaluation"]
    for i, stage in enumerate(stages):
        x = 0.02 + i * 0.163
        ax.text(x, 0.5, stage, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
                transform=ax.transAxes, fontsize=8)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 0.145, 0.5), xytext=(x + 0.095, 0.5),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->"))
    plt.tight_layout()
    plt.savefig(output_dir / "methodology_pipeline.png", dpi=180)
    plt.close()

    all_regression = []
    all_classification = []
    for label, features in [("A", ALL_FEATURES), ("B", FIELD_FEATURES)]:
        reg, best_name, best_model, X_test, y_test = run_regression(
            train, test, features, label, cv, lines, output_dir
        )
        all_regression.append(reg)
        cls = run_classification(train, test, features, label, cv, lines, output_dir)
        all_classification.append(cls)

        if label == "A":
            pred = best_model.predict(X_test)
            plt.figure(figsize=(6, 5))
            plt.scatter(y_test, pred, alpha=0.65)
            low, high = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
            plt.plot([low, high], [low, high], linestyle="--")
            plt.xlabel("Observed WQI")
            plt.ylabel("Predicted WQI")
            plt.title(f"Observed vs predicted WQI — set A ({best_name})")
            plt.tight_layout()
            plt.savefig(output_dir / "observed_vs_predicted_A.png", dpi=180)
            plt.close()

    pd.concat(all_regression, ignore_index=True).to_csv(
        output_dir / "regression_summary.csv", index=False
    )
    pd.concat(all_classification, ignore_index=True).to_csv(
        output_dir / "classification_summary.csv", index=False
    )

    plt.figure(figsize=(7, 4))
    water.groupby("year")["wqi"].median().plot(marker="o")
    plt.axhline(50, linestyle="--", linewidth=1)
    plt.axhline(75, linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Median WQI")
    plt.title("Median WQI by year")
    plt.tight_layout()
    plt.savefig(output_dir / "wqi_by_year.png", dpi=180)
    plt.close()

    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nExperiment outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
