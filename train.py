"""
AMR Lens - training and data export (run locally, NOT on Vercel).

What this does
1. Loads the compiled WHO GLASS 2022 dataset (Leclerc 2022, CC BY-NC-SA 3.0 IGO).
2. Evaluates models for the task "estimate a missing surveillance cell":
   given a country, year, specimen, pathogen and antibiotic, estimate % resistant.
   Evaluation holds out WHOLE cells (country x specimen x pathogen x antibiotic,
   all years together) with GroupKFold, so the model never sees the same cell
   in another year while being tested on it.
3. Compares: DummyRegressor (global mean), an additive baseline
   (country effect + bug-drug-specimen effect, via Ridge on one-hot),
   and a RandomForest that can learn interactions.
4. Saves the fitted models to api/model.joblib and the dashboard data
   to public/data.json.

Usage:  python train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_URL = ("https://raw.githubusercontent.com/qleclerc/GLASS2022/"
            "master/compiled_WHO_GLASS_2022.csv")
LOCAL_CSV = ROOT / "compiled_WHO_GLASS_2022.csv"
SEED = 42


# ---------------------------------------------------------------- data
def load() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_CSV if LOCAL_CSV.exists() else DATA_URL)
    df = df.rename(columns={
        "Iso3": "iso3", "CountryTerritoryArea": "country",
        "WHORegionName": "region", "Year": "year", "Specimen": "specimen",
        "PathogenName": "pathogen", "AbTargets": "antibiotic",
        "InterpretableAST": "n_tested", "Resistant": "n_resistant",
        "PercentResistant": "pct_resistant",
    })
    # Note: TotalSpecimenIsolates is aggregated across pathogens/antibiotics,
    # so it is NOT the sample size for a bug-drug pair. We drop it.
    df = df.drop(columns=["TotalSpecimenIsolates"])
    df["specimen"] = df["specimen"].str.title()
    df["year"] = df["year"].astype(int)
    # The "combo" = which bug-drug test this row is about.
    df["combo"] = df["pathogen"] + " | " + df["antibiotic"] + " | " + df["specimen"]
    # A cell = one combo in one country (all years).
    df["cell"] = df["iso3"] + " | " + df["combo"]
    # Sample weight: log of isolates tested. Raw counts would let a few
    # very large national systems (up to ~800k isolates) dominate the fit;
    # equal weights would let 10-isolate rows count as much as 10,000.
    df["weight"] = np.log1p(df["n_tested"])
    return df


# --------------------------------------------------------------- models
# n_resistant and n_tested are deliberately excluded: the target is
# computed from them (leakage).
def make_models():
    def ohe():
        return OneHotEncoder(handle_unknown="ignore")

    additive = Pipeline([
        ("prep", ColumnTransformer([("cat", ohe(), ["iso3", "region", "combo", "year"])])),
        ("reg", Ridge(alpha=1.0)),
    ])
    forest = Pipeline([
        ("prep", ColumnTransformer([("cat", ohe(),
                                     ["iso3", "region", "year", "specimen",
                                      "pathogen", "antibiotic"])])),
        ("reg", RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                      max_features=0.5, n_jobs=-1,
                                      random_state=SEED)),
    ])
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "additive": additive,
        "forest": forest,
    }


FEATURES = ["iso3", "region", "year", "specimen", "pathogen", "antibiotic", "combo"]


def fit(model, X, y, w):
    if isinstance(model, Pipeline):
        return model.fit(X, y, reg__sample_weight=w)
    return model.fit(X, y, sample_weight=w)


def evaluate(df: pd.DataFrame) -> dict:
    X, y, w = df[FEATURES], df["pct_resistant"].values, df["weight"].values
    gkf = GroupKFold(n_splits=5)
    preds = {name: np.zeros(len(df)) for name in make_models()}
    for train_idx, test_idx in gkf.split(X, y, groups=df["cell"]):
        for name, model in make_models().items():
            fit(model, X.iloc[train_idx], y[train_idx], w[train_idx])
            preds[name][test_idx] = np.clip(model.predict(X.iloc[test_idx]), 0, 100)

    big = df["n_tested"].values >= 100
    results = {}
    for name, p in preds.items():
        results[name] = {
            "mae": round(float(mean_absolute_error(y, p)), 2),
            "mae_weighted": round(float(mean_absolute_error(y, p, sample_weight=w)), 2),
            "mae_n100": round(float(mean_absolute_error(y[big], p[big])), 2),
            "r2": round(float(r2_score(y, p)), 3),
        }
    return results, preds


def importance(df: pd.DataFrame, forest) -> dict:
    """Grouped permutation importance: shuffle one input column at a time
    and see how much in-sample MAE worsens. Coarse but readable."""
    rng = np.random.default_rng(SEED)
    X, y = df[FEATURES].copy(), df["pct_resistant"].values
    base = mean_absolute_error(y, forest.predict(X))
    out = {}
    for col in ["iso3", "region", "year", "specimen", "pathogen", "antibiotic"]:
        deltas = []
        for _ in range(3):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            deltas.append(mean_absolute_error(y, forest.predict(Xp)) - base)
        out[col] = round(float(np.mean(deltas)), 2)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------- export
def main():
    df = load()
    print(f"{len(df)} rows, {df['iso3'].nunique()} countries, "
          f"{df['combo'].nunique()} bug-drug-specimen combos")
    print(df.groupby("year")["iso3"].nunique().rename("countries per year"))

    results, _ = evaluate(df)
    print("\nGrouped 5-fold CV (whole cells held out):")
    for k, v in results.items():
        print(f"  {k:9s} {v}")

    # Refit on everything for deployment
    models = make_models()
    X, y, w = df[FEATURES], df["pct_resistant"].values, df["weight"].values
    for m in models.values():
        fit(m, X, y, w)
    imp = importance(df, models["forest"])
    print("\nPermutation importance (MAE increase, forest):", imp)

    countries = (df[["iso3", "country", "region"]].drop_duplicates()
                 .sort_values("country"))
    meta = {
        "countries": countries.to_dict(orient="records"),
        "combos": sorted(df["combo"].unique().tolist()),
        "years": sorted(df["year"].unique().tolist()),
    }
    (ROOT / "api").mkdir(exist_ok=True)
    joblib.dump({"additive": models["additive"], "forest": models["forest"],
                 "meta": meta, "features": FEATURES},
                ROOT / "api" / "model.joblib", compress=3)

    years_per_country = df.groupby("iso3")["year"].nunique()
    panel = sorted(years_per_country[years_per_country == 4].index.tolist())
    rows = df[["iso3", "year", "specimen", "pathogen", "antibiotic",
               "n_tested", "n_resistant"]].values.tolist()
    data = {
        "columns": ["iso3", "year", "specimen", "pathogen", "antibiotic",
                    "n_tested", "n_resistant"],
        "rows": rows,
        "countries": meta["countries"],
        "panel_countries": panel,
        "metrics": results,
        "importance": imp,
        "source": "Leclerc, Q.J. (2022). Compiled data from WHO GLASS 2022 report "
                  "(v2.0). doi:10.5281/zenodo.7486150. Licence CC BY-NC-SA 3.0 IGO.",
    }
    (ROOT / "public").mkdir(exist_ok=True)
    with open(ROOT / "public" / "data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("\nSaved api/model.joblib and public/data.json")


if __name__ == "__main__":
    main()
