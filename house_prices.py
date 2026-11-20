"""House price model built on the preprocessing in the notebook.

Notebook steps, reproduced as a scikit-learn transformer (fitted on training data only):
  1. drop columns with more than 15% missing values
  2. numeric gaps: mean if |median - mean| < 40, otherwise median
  3. categorical gaps: most frequent value
Then one-hot encoding of categorical columns and a regressor trained on log(SalePrice)
(the notebook plots the log of SalePrice; the Kaggle metric is RMSE of the log price).

Run ``python house_prices.py`` to compare the regressors imported in the notebook with
5-fold cross-validation and write ``models/metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "House Prices - Advanced Regression Techniques.csv"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
TARGET = "SalePrice"
RANDOM_STATE = 42


class NotebookImputer(BaseEstimator, TransformerMixin):
    """Missing-value handling from the notebook, learned on the training data."""

    def __init__(self, max_missing: float = 15.0, gap: float = 40.0):
        self.max_missing = max_missing
        self.gap = gap

    def fit(self, X: pd.DataFrame, y=None):
        perc = X.isnull().mean() * 100
        self.dropped_ = sorted(perc[perc > self.max_missing].index)
        kept = X.drop(columns=self.dropped_)
        self.fill_ = {}
        for col in kept.columns:
            s = kept[col]
            if pd.api.types.is_numeric_dtype(s):
                mean, median = s.mean(), s.median()
                self.fill_[col] = mean if abs(median - mean) < self.gap else median
            else:
                self.fill_[col] = s.mode().iloc[0]
        self.columns_ = list(kept.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.reindex(columns=self.columns_).copy()
        # assignment, not inplace=True: column-level inplace fillna is a no-op under pandas copy-on-write
        return out.fillna(value=self.fill_)


def make_regressor(name: str):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    return {
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=50000),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=50000),
        "SVR": SVR(C=10, epsilon=0.01),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=3, subsample=0.8,
                                colsample_bytree=0.7, random_state=RANDOM_STATE, n_jobs=-1),
    }[name]


MODELS = ["Ridge", "Lasso", "ElasticNet", "SVR", "RandomForest", "XGBoost"]


def build_pipeline(name: str) -> TransformedTargetRegressor:
    encode = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), make_column_selector(dtype_include=object)),
    ])
    pipe = Pipeline([("impute", NotebookImputer()), ("encode", encode), ("model", make_regressor(name))])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def load_data(path: str | Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} column not found in {path}")
    X = df.drop(columns=[TARGET, "Id"])
    obj = X.select_dtypes(exclude=np.number).columns
    X[obj] = X[obj].astype(object)  # plain object dtype for the encoder selector
    return X, df[TARGET]


def compare_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold CV RMSE of log price (the Kaggle metric) and MAE in dollars."""
    from sklearn.model_selection import KFold, cross_validate

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name in MODELS:
        log_rmse, mae = [], []
        for tr, te in cv.split(X):
            m = build_pipeline(name).fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            log_rmse.append(float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(y.iloc[te])) ** 2))))
            mae.append(float(np.mean(np.abs(pred - y.iloc[te]))))
        results[name] = {"log_rmse": float(np.mean(log_rmse)), "log_rmse_std": float(np.std(log_rmse)),
                         "mae": float(np.mean(mae))}
        print(f"{name:12} log-RMSE {results[name]['log_rmse']:.4f} ± {results[name]['log_rmse_std']:.4f}   MAE ${results[name]['mae']:,.0f}")
    return results


def best_model(results: dict) -> str:
    return min(results, key=lambda k: results[k]["log_rmse"])


if __name__ == "__main__":
    X, y = load_data()
    res = compare_models(X, y)
    best = best_model(res)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"cv": "5-fold KFold(shuffle=True, random_state=42)",
                                        "results": res, "best": best}, indent=2))
    print("best:", best, "->", METRICS_PATH)
