# Poland Apartment Price Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-Model-success?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/CatBoost-Model-yellow" alt="CatBoost">
  <img src="https://img.shields.io/badge/Scikit--learn-Pipeline-blue?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Working-green" alt="Status">
</p>

<p align="center">
  <strong>Predicting residential apartment sale prices across 15 Polish cities, using 11 months of listing snapshots</strong><br>
  From raw monthly listing exports through leakage-safe cleaning, exploratory analysis, feature engineering, and gradient-boosted ensemble modeling.
</p>

---

## 📑 Table of Contents

- [About This Project](#about-this-project)
- [Problem Statement](#problem-statement)
- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Wrangling](#1-data-wrangling)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Modeling](#4-modeling)
  - [5. Model Evaluation](#5-model-evaluation)
- [Key Findings](#key-findings)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [How to Run](#how-to-run)
- [Author](#author)

---

## About This Project

This project is a rebuild of an earlier single-city ("Flats in Cracow") pricing model, using a richer, publicly available, multi-city dataset instead of a custom scraper. A Kaggle dataset of Polish apartment listings — 15 cities, 11 monthly snapshots from August 2023 to June 2024 — was cleaned, explored, and used to train gradient-boosted regression models to predict sale price.

The project showcases:
- **Leakage-safe pipeline** — outlier filtering and imputation fit on the training split only, not the full dataset
- **Time-based evaluation** — the model is tested on months it never trained on, not a random split, to reflect how it would actually be used
- **Temporal price tracking** — repeated listings across months are kept (not deduplicated away), preserving real price movement over time
- **Modern gradient boosting** — LightGBM and CatBoost in place of scikit-learn's slower `GradientBoostingRegressor`, plus a learned Stacking ensemble

---

## Problem Statement

The goal is to predict apartment sale prices across major Polish cities using property characteristics and location features, and to evaluate whether the model generalizes to **future** listings rather than just held-out listings from the same time period.

Key questions addressed:
1. Which property and location features most strongly predict sale price?
2. How much does city alone explain, versus property-level characteristics?
3. Does the model hold up when evaluated on months after its training window?

---

## Data Source

**Kaggle: [`krzysztofjamroz/apartment-prices-in-poland`](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland)**

11 monthly CSV snapshots of sale listings (`apartments_pl_YYYY_MM.csv`), August 2023 – June 2024, covering 15 Polish cities. Each row is one listing at one point in time; the same physical apartment (`id`) can appear in multiple months if it stayed listed.

| Feature Type | Examples |
|---|---|
| **Numeric** | price, squareMeters, rooms, floor, floorCount, buildYear |
| **Location** | latitude, longitude, centreDistance, poiCount, and distances to school/clinic/post office/kindergarten/restaurant/college/pharmacy |
| **Boolean** | hasParkingSpace, hasBalcony, hasElevator, hasSecurity, hasStorageRoom |
| **Categorical** | city, type, ownership, buildingMaterial, condition |

**Data quality notes:**
- `id` reliably tracks the same physical unit across months (only ~1,200 of ~93,000 unique listings showed inconsistent `squareMeters` across snapshots — dropped as likely `id` collisions)
- `condition` (76% missing), `buildingMaterial` (39%), and `type` (21%) are filled with an explicit `"unknown"` category rather than dropped
- Outlier bounds (price: 2.5–97.5 percentile, area: 1–99 percentile) and all imputation are fit on the **training split only**

---

## Project Structure

```
poland-apartment-pricing/
├── 00_Data_Wrangling.ipynb          # Combine monthly files, clean, tag with snapshot month
├── 01_Exploratory_Analysis.ipynb    # EDA: price by city, numeric/categorical/temporal analysis
├── 02_Model.ipynb                   # Feature engineering, time-based split, modeling, evaluation
├── requirements.txt
└── img/
    ├── price_distribution.png
    ├── price_by_city.png
    ├── price_per_sqm_by_city.png
    ├── numeric_vs_price.png
    ├── correlation_matrix.png
    ├── binary_features_price.png
    ├── categorical_features_price.png
    ├── price_trend_over_time.png
    ├── price_trend_by_city.png
    └── feature_importance.png
```

---

## Methodology

### 1. Data Wrangling

The cleaning pipeline (`00_Data_Wrangling.ipynb`):

1. **Combine** all 11 monthly sale files, tagging each row with a `snapshot_month` column derived from the filename.
2. **Keep all snapshots** — the same `id` can appear in multiple months. This is intentional, not duplication: ~45,300 listings appear in more than one month, and price genuinely changed for ~12,900 of them (median 4.3% change), which is the temporal signal this project is built to capture.
3. **Drop inconsistent listings** — ~1,200 `id`s where `squareMeters` changed across snapshots (likely `id` collisions rather than real history) are removed.
4. **Fill sparse categoricals** (`condition`, `buildingMaterial`, `type`) with `"unknown"` instead of dropping rows.
5. **Convert boolean amenity columns** (`yes`/`no` strings) to real booleans.
6. **No outlier filtering or imputation here** — both are deferred to the modeling notebook and fit on the training split only.
7. **Output:** `cleaned_data_sale.csv` (190,858 rows after cleaning, from 195,568 raw rows).

### 2. Exploratory Data Analysis

`01_Exploratory_Analysis.ipynb` examines:
- Price distribution (raw and log), overall and by city
- Price per m² by city (normalizes for size differences)
- Numeric features vs. price, and their correlation matrix
- Binary amenities and categorical features vs. price
- **Temporal price trend** across the 11-month window, overall and for the top 5 cities by listing volume

### 3. Feature Engineering

Adapted from the original project's ratio/count approach:

| Feature | Formula | Rationale |
|---|---|---|
| `Log Area` | `log(squareMeters)` | Reduce right-skew |
| `Bool Sum` | count of True amenity flags | Amenity count |
| `Area to Bool Sum` | `squareMeters / (Bool Sum + 1)` | Space per amenity |
| `Rooms to Bool Sum` | `rooms / (Bool Sum + 1)` | Rooms relative to amenities |
| `Area to Rooms` | `squareMeters / rooms` | Average room size |
| `month_idx` | months since first snapshot (ordinal) | Lets the model extrapolate to future months, unlike a categorical encoding |

### 4. Modeling

**Split: time-based, not random.**
- Train: August 2023 – April 2024 (140,401 rows after outlier filtering)
- Test: May – June 2024 (38,405 rows after outlier filtering)
- This tests forward generalization — the realistic deployment scenario — rather than letting the model see future market conditions during training, which a random split would allow.
- ~39% of test-period listings were also seen (at a different price) during training. This is expected, not leakage: rows are never shuffled across the time boundary, and a production model would routinely re-encounter previously-listed properties.

**Preprocessing:**
- `OneHotEncoder(handle_unknown='ignore')` for categorical features
- Median imputation (`SimpleImputer`) for `floor`, `floorCount`, `buildYear`, and the POI-distance columns — fit on train only. (The original project used `KNNImputer`; that doesn't scale to this dataset's row count, since KNN imputation is O(n²).)
- `ColumnTransformer` with `remainder='passthrough'`

**Models:**

| Model | Notes |
|---|---|
| `DummyRegressor` | Mean-predictor baseline |
| `LGBMRegressor` | Fixed hyperparameters (n_estimators=300, num_leaves=63, learning_rate=0.08) |
| `CatBoostRegressor` | Fixed hyperparameters (iterations=400, depth=8, learning_rate=0.08) |
| `StackingRegressor` | LightGBM + CatBoost, blended via a `Ridge` meta-learner (cv=3) |

`GradientBoostingRegressor` (scikit-learn) is included as a commented-out, optional cell — it's sequential and far slower than LightGBM/CatBoost at this row count, without a corresponding accuracy advantage.

### 5. Model Evaluation

Evaluated on the held-out May–June 2024 test set:

| Model | RMSE | MAE | MSLE | RMSE improvement vs. baseline |
|---|---|---|---|---|
| Dummy (Baseline) | 323,717 | 244,845 | 0.167 | — |
| LightGBM | 101,049 | 70,826 | 0.014 | 68.8% |
| CatBoost | 107,190 | 74,900 | 0.015 | 66.9% |
| **Stacking (LGBM+CBR)** | **99,963** | **70,802** | **0.014** | **69.1%** |

> The Stacking ensemble edges out LightGBM alone, echoing the original project's finding that blending outperforms any single model — this time with real numbers and a genuinely out-of-time test set behind the claim.

**Prediction interface:** `get_pred()` accepts raw property characteristics (city, type, size, rooms, location features, amenities, snapshot month) and returns a predicted price rounded to the nearest 1,000 PLN.

---

## Key Findings

### Model Performance
- The **Stacking ensemble** (LightGBM + CatBoost, Ridge meta-learner) achieved the best RMSE, MAE, and MSLE on genuinely future data.
- LightGBM alone comes within ~1% of the ensemble's RMSE — most of the gain comes from LightGBM, not the ensembling itself.

### Feature Insights
- **`squareMeters`** is the strongest single numeric predictor (r ≈ 0.64), consistent with the original Cracow-only project.
- **City** is a dominant driver: median sale price ranges from ~320K PLN (Częstochowa) to ~885K PLN (Warszawa) — nearly a 3x spread.
- Distance-to-amenity features (school, clinic, pharmacy, etc.) show weak individual correlation with price but may still contribute through interactions captured by the tree-based models.

### Business / Temporal Insights
- Median sale price rose **~16%** from August 2023 to mid-2024 across the dataset — a trend the original single-snapshot project could not observe.
- Property size and city remain the dominant price drivers, matching the "size and location dominate" conclusion from the original Cracow project — now validated across a 15-city dataset instead of one.

---

## Design Decisions

A few explicit departures from the original single-city project, made along the way:

- **Kept all monthly snapshots** instead of deduplicating to the latest listing per property, to preserve real price movement over time (Option B from the original project-scoping discussion).
- **Time-based train/test split** instead of a random split, to evaluate genuine forward generalization rather than same-period accuracy.
- **Median imputation instead of KNN imputation** — `KNNImputer` doesn't scale to ~150K training rows.
- **`month_idx` as an ordinal feature** rather than one-hot encoding `snapshot_month`, so the model can extrapolate to months outside its training window instead of treating each month as an unrelated category.
- **LightGBM/CatBoost in place of `GradientBoostingRegressor`** as the primary models — much faster at this scale with comparable or better accuracy; GBR is kept as an optional, commented-out cell for reference.

---

## Limitations

- **Fixed hyperparameters, not grid-searched** — LightGBM, CatBoost, and the excluded GBR option all use a single reasonable configuration rather than `GridSearchCV`, to keep runtime predictable. Widen this if you have more compute time.
- **~39% of test-period listings were also seen during training** (same apartment, earlier snapshot). This reflects realistic deployment rather than classic leakage, but it does mean part of the test set isn't fully "unseen."
- **`month_idx` assumes a roughly monotonic trend** — if the market reverses direction beyond the training window, this feature could mislead the model rather than help it.
- **Rent listings are excluded** — the dataset includes separate monthly rent files (`apartments_rent_pl_*.csv`, Nov 2023–Jun 2024) that aren't used here; rent and sale prices are different economic quantities and would need a separate model.
- **No district-level analysis** — unlike the original Cracow project, this dataset has no district field, only city + coordinates/derived distances, so within-city geographic variation isn't modeled directly.

---

## Next Steps

- **Hyperparameter tuning** via `GridSearchCV` or `RandomizedSearchCV` on LightGBM/CatBoost now that the pipeline is verified working.
- **District-level features** by reverse-geocoding `latitude`/`longitude` into neighborhood boundaries.
- **Rent price model** using the parallel `apartments_rent_pl_*.csv` files.
- **Grouped or blocked time-series cross-validation** (rolling-origin) instead of a single train/test cutoff, for a more robust estimate of forward performance.
- **SHAP analysis** to explain individual predictions beyond aggregate feature importance.
- **Deploy `get_pred()`** as a small Streamlit or FastAPI app for interactive price estimates.

---

## Technical Specifications

| Specification | Detail |
|---|---|
| **Language** | Python 3 |
| **Target variable** | `price` (PLN) — apartment sale price |
| **Cities covered** | 15 |
| **Raw rows** | 195,568 (11 monthly files) |
| **Cleaned rows** | 190,858 |
| **Train rows (post-filtering)** | 140,401 (Aug 2023 – Apr 2024) |
| **Test rows (post-filtering)** | 38,405 (May – Jun 2024) |
| **Best model** | Stacking (LightGBM + CatBoost) |
| **Best RMSE** | 99,963 PLN |
| **Random seed** | 123 |

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data setup

Download the dataset from Kaggle and place all `apartments_pl_YYYY_MM.csv` files in one folder, e.g. `../flats-data/`. Update the `data_dir` variable at the top of each notebook to match your actual folder path.

### Notebook Execution

1. **Run `00_Data_Wrangling.ipynb`** — combines and cleans the monthly files, produces `cleaned_data_sale.csv`.
2. **Run `01_Exploratory_Analysis.ipynb`** — generates exploration plots (optional before modeling, but recommended).
3. **Run `02_Model.ipynb`** — feature engineering, time-based split, model training, evaluation, and `get_pred()`.

> Notebooks are sequential — each depends on the file the previous one wrote.

---

## Author

**Sarvesh Kumar Sharma**

- GitHub: [@shsarv](https://github.com/shsarv)
- LinkedIn: [in/shsarv](https://linkedin.com/in/shsarv)

*Rebuilt as a multi-city project from the original single-city "Flats in Cracow" version.*

---

<p align="center">
  <a href="../README.md">← Back to repository root</a>
</p>
