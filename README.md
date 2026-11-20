# 🏠 House Prices — Advanced Regression Techniques

A regression project on the Kaggle competition
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):
predict the sale price of houses in Ames, Iowa from 79 explanatory variables. Includes data exploration,
preprocessing, a cross-validated comparison of six regressors, and an interactive **Streamlit** demo.

**🔗 Live demo:** https://house-prices---advanced-regression-techniques-kpscjxhpxbebnh2q.streamlit.app/

## 🚀 Features

- **EDA notebook:** target distribution (log scale), missing-value analysis, duplicates check.
- **Preprocessing (from the notebook):**
  - drop columns with more than 15% missing values;
  - fill numeric gaps with the mean, or the median when the two differ by 40 or more;
  - fill categorical gaps with the most frequent value.
- **Modelling (`house_prices.py`):**
  - the same preprocessing as a scikit-learn transformer, fitted on training folds only;
  - one-hot encoding and standard scaling;
  - target trained on log(price);
  - 5-fold cross-validated comparison of the six regressors imported in the notebook.
- **Streamlit app:**
  - predict a price by editing the most influential features of a typical house or of any house in the dataset;
  - input validation and a training-range warning;
  - a "what if the living area changed" chart;
  - model comparison and feature importance;
  - market charts: price distribution, price by neighborhood, price vs. living area.

## 📊 Results

5-fold cross-validation, `KFold(shuffle=True, random_state=42)`, on the 1,460 training houses:

| Model | RMSE of log price | MAE |
|---|---|---|
| **XGBoost** | **0.1254 ± 0.0193** | **$14,853** |
| Lasso | 0.1397 ± 0.0416 | $16,376 |
| ElasticNet | 0.1403 ± 0.0402 | $16,489 |
| SVR | 0.1437 ± 0.0143 | $17,018 |
| Ridge | 0.1447 ± 0.0377 | $17,478 |
| Random Forest | 0.1448 ± 0.0202 | $17,682 |

The app uses the best model, XGBoost. Re-run `python house_prices.py` to regenerate `models/metrics.json`.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
House-Prices---Advanced-Regression-Techniques/
├── House Prices - Advanced Regression Techniques.csv     # Kaggle train.csv (1,460 houses)
├── House Prices - Advanced Regression Techniques.ipynb   # EDA + preprocessing notebook
├── house_prices.py      # Preprocessing transformer, pipelines, model comparison
├── models/metrics.json  # Cross-validation results
├── app.py               # Streamlit demo
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/House-Prices---Advanced-Regression-Techniques.git
cd House-Prices---Advanced-Regression-Techniques
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

Live demo: https://house-prices---advanced-regression-techniques-kpscjxhpxbebnh2q.streamlit.app/

Run the Streamlit demo locally (opens http://localhost:8501; trains the model in a few seconds on first load):

```bash
streamlit run app.py
```

Re-run the model comparison (about 40 seconds):

```bash
python house_prices.py
```

Notebook: run `pip install jupyter matplotlib seaborn`, then `jupyter notebook`.

## 👩‍💻 Author

Amira Farhan — Machine Learning & Computer Vision Engineer
