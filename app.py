"""Streamlit demo: Ames house price prediction.

Run with:  streamlit run app.py   (model comparison numbers come from `python house_prices.py`)
"""

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import house_prices as hp

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

NUMERIC_INPUTS = {  # column: (label, step)
    "OverallQual": ("Overall quality (1–10)", 1),
    "GrLivArea": ("Above-ground living area (sq ft)", 50),
    "TotalBsmtSF": ("Basement area (sq ft)", 50),
    "GarageCars": ("Garage capacity (cars)", 1),
    "YearBuilt": ("Year built", 1),
    "YearRemodAdd": ("Year remodelled", 1),
    "FullBath": ("Full bathrooms", 1),
    "LotArea": ("Lot area (sq ft)", 100),
}
INT_COLS = {"OverallQual", "GarageCars", "YearBuilt", "YearRemodAdd", "FullBath"}
CAT_INPUTS = {"Neighborhood": "Neighborhood", "KitchenQual": "Kitchen quality", "ExterQual": "Exterior quality"}
QUAL = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical", "Fa": "Fair", "Po": "Poor"}


@st.cache_data
def get_data():
    return hp.load_data()


@st.cache_resource(show_spinner="Training the XGBoost model (a few seconds, once)...")
def get_model(name: str):
    X, y = get_data()
    return hp.build_pipeline(name).fit(X, y)


@st.cache_data
def get_metrics():
    try:
        return json.loads(hp.METRICS_PATH.read_text())
    except FileNotFoundError:
        return None


def importance(model) -> pd.Series:
    pipe = model.regressor_
    names = pipe.named_steps["encode"].get_feature_names_out()
    raw = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    cats = pipe.named_steps["encode"].named_transformers_["cat"].feature_names_in_
    def base(n):
        n = n.split("__", 1)[1]
        for c in cats:
            if n.startswith(c + "_"):
                return c
        return n
    return raw.groupby(base).sum().sort_values(ascending=False)


try:
    X, y = get_data()
    metrics = get_metrics()
    best = metrics["best"] if metrics else "XGBoost"
    model = get_model(best)
except Exception as exc:
    st.error(f"Could not load data or train the model: {exc}")
    st.stop()

template_typical = {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0]) for c in X.columns}

st.title("🏠 House Price Predictor — Ames, Iowa")
st.caption(f"{best} model trained on {len(X):,} house sales (Kaggle *House Prices: Advanced Regression Techniques*).")

tab_pred, tab_model, tab_data = st.tabs(["💰 Predict a price", "📈 Model comparison", "📊 Market data"])

# ---------------- Prediction ----------------
with tab_pred:
    base_choice = st.radio("Start from", ["A typical house (median values)", "A house from the dataset"], horizontal=True)
    if base_choice.startswith("A house"):
        row_id = st.number_input("House Id (1–1460)", min_value=1, max_value=len(X), value=1, step=1)
        base = X.iloc[row_id - 1].to_dict()
        st.caption(f"Actual sale price of house {row_id}: **${y.iloc[row_id - 1]:,.0f}**")
        st.caption("This house is part of the training data, so its prediction is not an independent test "
                   "(see the cross-validated error in Model comparison).")
    else:
        base = dict(template_typical)

    with st.form("house"):
        st.write("Adjust the most influential features; all other features keep the starting house's values.")
        cols = st.columns(4)
        values = {}
        for i, (col, (label, step)) in enumerate(NUMERIC_INPUTS.items()):
            lo, hi = X[col].min(), X[col].max()
            v = base[col] if pd.notna(base[col]) else template_typical[col]
            num = int if col in INT_COLS else float  # whole-number fields use integer inputs
            values[col] = cols[i % 4].number_input(label, min_value=num(0), max_value=num(hi * 2), value=num(round(v)),
                                                   step=num(step), help=f"Range in data: {lo:,.0f} – {hi:,.0f}")
        ccols = st.columns(3)
        for i, (col, label) in enumerate(CAT_INPUTS.items()):
            opts = sorted(X[col].dropna().unique())
            cur = base[col] if base[col] in opts else template_typical[col]
            values[col] = ccols[i].selectbox(label, opts, index=opts.index(cur),
                                             format_func=lambda o: QUAL.get(o, o))
        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        errors, warnings = [], []
        if values["YearRemodAdd"] < values["YearBuilt"]:
            errors.append("Year remodelled cannot be earlier than year built.")
        if not 1 <= values["OverallQual"] <= 10:
            errors.append("Overall quality must be between 1 and 10.")
        if values["GrLivArea"] <= 0:
            errors.append("Living area must be greater than zero.")
        for col in NUMERIC_INPUTS:
            if not X[col].min() <= values[col] <= X[col].max():
                warnings.append(NUMERIC_INPUTS[col][0])
        if errors:
            for e in errors:
                st.error(e)
        else:
            if warnings:
                st.warning("Outside the range seen in training (less reliable): " + ", ".join(warnings))
            house = {**base, **values}  # features not shown keep the starting house's values
            frame = pd.DataFrame([house])[X.columns]
            try:
                price = float(model.predict(frame)[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                mae = metrics["results"][best]["mae"] if metrics else None
                left, right = st.columns([1, 2])
                left.metric("Predicted sale price", f"${price:,.0f}")
                if mae:
                    left.caption(f"Typical error in cross-validation: ±${mae:,.0f} (MAE)")
                nb = y[X["Neighborhood"] == values["Neighborhood"]]
                left.caption(f"{values['Neighborhood']}: median sale ${nb.median():,.0f} across {len(nb)} sales")

                # what-if: vary living area
                grid = np.linspace(X["GrLivArea"].quantile(0.02), X["GrLivArea"].quantile(0.98), 30)
                sweep = pd.DataFrame([{**house, "GrLivArea": g} for g in grid])[X.columns]
                curve = pd.DataFrame({"GrLivArea": grid, "price": model.predict(sweep)})
                line = alt.Chart(curve).mark_line().encode(
                    x=alt.X("GrLivArea:Q", title="Above-ground living area (sq ft)"),
                    y=alt.Y("price:Q", title="Predicted price ($)", axis=alt.Axis(format="$,.0f")),
                    tooltip=[alt.Tooltip("GrLivArea:Q", format=",.0f"), alt.Tooltip("price:Q", format="$,.0f")])
                dot = alt.Chart(pd.DataFrame({"GrLivArea": [values["GrLivArea"]], "price": [price]})).mark_point(
                    size=160, filled=True, color="#e4572e").encode(x="GrLivArea:Q", y="price:Q")
                right.altair_chart((line + dot).properties(title="What if the living area changed?", height=300),
                                   width="stretch")

# ---------------- Model comparison ----------------
with tab_model:
    if metrics:
        res = pd.DataFrame(metrics["results"]).T.reset_index(names="model").sort_values("log_rmse")
        c1, c2 = st.columns(2)
        c1.metric(f"Best model: {best}", f"{metrics['results'][best]['log_rmse']:.4f}", "log-RMSE (lower is better)",
                  delta_color="off")
        c2.metric("Mean absolute error", f"${metrics['results'][best]['mae']:,.0f}", "5-fold cross-validation",
                  delta_color="off")
        st.altair_chart(
            alt.Chart(res).mark_bar().encode(
                x=alt.X("log_rmse:Q", title="RMSE of log price (5-fold CV)"),
                y=alt.Y("model:N", sort="x", title=None),
                color=alt.condition(alt.datum.model == best, alt.value("#2a9d8f"), alt.value("#adb5bd")),
                tooltip=["model", alt.Tooltip("log_rmse:Q", format=".4f"), alt.Tooltip("log_rmse_std:Q", format=".4f"),
                         alt.Tooltip("mae:Q", format="$,.0f")],
            ).properties(height=260), width="stretch")
        st.caption("All six regressors are the ones imported in the notebook, each trained with the same preprocessing. "
                   "Recompute with `python house_prices.py`.")
    else:
        st.info("Run `python house_prices.py` to generate the model comparison.")
    imp = importance(model).head(15).reset_index()
    imp.columns = ["feature", "importance"]
    st.altair_chart(alt.Chart(imp).mark_bar(color="#2a9d8f").encode(
        x=alt.X("importance:Q", title="Share of model importance"), y=alt.Y("feature:N", sort="-x", title=None)
    ).properties(title=f"Top 15 features ({best})", height=alt.Step(22)), width="stretch")

# ---------------- Data ----------------
with tab_data:
    a, b = st.columns(2)
    a.altair_chart(alt.Chart(pd.DataFrame({"SalePrice": y})).mark_bar().encode(
        x=alt.X("SalePrice:Q", bin=alt.Bin(maxbins=40), title="Sale price ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("count()", title="Houses")).properties(title="Sale price distribution", height=280), width="stretch")
    by_nb = pd.DataFrame({"Neighborhood": X["Neighborhood"], "SalePrice": y})
    order = by_nb.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index.tolist()
    b.altair_chart(alt.Chart(by_nb).mark_boxplot(extent="min-max").encode(
        y=alt.Y("Neighborhood:N", sort=order, title=None),
        x=alt.X("SalePrice:Q", title="Sale price ($)", axis=alt.Axis(format="$,.0s"))
    ).properties(title="Sale price by neighborhood", height=alt.Step(14)), width="stretch")
    st.altair_chart(alt.Chart(pd.DataFrame({"GrLivArea": X["GrLivArea"], "SalePrice": y, "OverallQual": X["OverallQual"]})).mark_circle(
        size=25, opacity=0.6).encode(
        x=alt.X("GrLivArea:Q", title="Above-ground living area (sq ft)"),
        y=alt.Y("SalePrice:Q", title="Sale price ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("OverallQual:O", title="Overall quality", scale=alt.Scale(scheme="viridis"))
    ).properties(title="Price vs. living area", height=320), width="stretch")
