# 🏠 Home Value Advisor — House Price Prediction & Recommendation System

An end-to-end Machine Learning web application that predicts house prices,
explains *why* a home is priced the way it is, and gives personalized
recommendations to buyers, sellers, and real-estate agents.

Built with **Python, scikit-learn, and Streamlit** — single-file architecture,
ready to run locally or deploy to the cloud.

---

## 🎯 Real-World Problem It Solves

Home buyers and sellers often lack an objective, data-driven way to judge
whether a listed price is fair. Real-estate agents spend hours manually
comparing listings. This app solves that by:

- Giving an **instant, statistically grounded price estimate** for any home
  based on its features (area, bedrooms, amenities, location quality, etc.)
- Showing **which specific features are driving the price up or down**
- Recommending **concrete, actionable upgrades** (e.g. "add air conditioning",
  "furnish the home") with an estimated ₹ value impact
- Warning users when a price estimate looks unusually high or low compared
  to the market, so they can sanity-check before committing

---

## ✨ Features

| Section | What it does |
|---|---|
| 📊 **Overview & EDA** | Dataset summary stats, missing-value check, price distribution, price vs. area scatter, correlation heatmap, amenity-vs-price comparisons |
| 🤖 **Model Comparison** | Trains and benchmarks **7 regression algorithms** side by side (R², MAE, RMSE, 5-fold CV), predicted-vs-actual chart, feature importance |
| 🔮 **Predict & Get Recommendations** | Interactive form for property details → instant price prediction → area sensitivity chart → per-amenity value-impact chart → plain-English recommendations |

### Models compared
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. K-Nearest Neighbors Regressor

The app automatically selects the **best-performing model** (by R² score) as
the default for predictions, while still letting the user pick any model to
compare.

---

## 🗂️ Project Structure

```
.
├── app.py               # Single-file Streamlit application (EDA + training + UI)
├── Housing.csv           # Dataset (place in the same folder as app.py)
├── requirements.txt      # Python dependencies
└── README.md              # This file
```

Everything — data loading, preprocessing, model training, evaluation, and the
UI — lives in **one Python file (`app.py`)** for simplicity of deployment,
using Streamlit's caching (`@st.cache_data` / `@st.cache_resource`) so
training only happens once per session, not on every interaction.

---

## 📊 Dataset

The app expects a CSV named `Housing.csv` with the following columns
(this matches the popular Kaggle "Housing Prices Dataset"):

| Column | Type | Description |
|---|---|---|
| `price` | numeric | Target variable — sale price |
| `area` | numeric | Plot/floor area in sqft |
| `bedrooms` | numeric | Number of bedrooms |
| `bathrooms` | numeric | Number of bathrooms |
| `stories` | numeric | Number of floors |
| `mainroad` | yes/no | On a main road |
| `guestroom` | yes/no | Has a guest room |
| `basement` | yes/no | Has a basement |
| `hotwaterheating` | yes/no | Has hot water heating |
| `airconditioning` | yes/no | Has air conditioning |
| `parking` | numeric | Number of parking spots |
| `prefarea` | yes/no | Located in a preferred/desirable area |
| `furnishingstatus` | categorical | `furnished` / `semi-furnished` / `unfurnished` |

No missing values are present in the reference dataset. If you use your own
data with missing values, add cleaning steps to the `load_data()` function.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure `Housing.csv` is in the same folder as `app.py`

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## ☁️ Deployment

This app is ready to deploy on:

- **Streamlit Community Cloud** — push `app.py`, `Housing.csv`, and
  `requirements.txt` to a public GitHub repo, then connect the repo at
  [share.streamlit.io](https://share.streamlit.io).
- **Render / Railway / Heroku** — use the same `requirements.txt` and start
  command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Docker** — wrap the same install + run commands in a container.

---

## 🧠 How It Works (Technical Summary)

1. **Preprocessing**: `yes/no` columns are label-encoded to `1/0`;
   `furnishingstatus` is one-hot encoded; all features are standardized with
   `StandardScaler`.
2. **Train/test split**: 80/20 split, `random_state=42` for reproducibility.
3. **Training**: All 7 models are trained on the scaled training set.
4. **Evaluation**: R², MAE, RMSE on the held-out test set, plus 5-fold
   cross-validated R² for a more robust performance estimate.
5. **Prediction**: The user's inputs go through the same encoding + scaling
   pipeline, then are fed into the chosen model.
6. **Explainability**: 
   - *Area sensitivity chart* — re-predicts price across a range of areas
     with all other inputs held fixed.
   - *Amenity impact chart* — for each yes/no amenity, flips it and
     re-predicts to measure the ₹ impact of that single feature.
7. **Recommendations**: Simple rule-based logic reads the impact chart and
   market percentile to generate human-readable suggestions.

---

## ⚠️ Limitations & Disclaimer

- Predictions are based on a limited dataset and should **not** replace a
  professional property appraisal.
- The model captures patterns present in the training data only — unusual
  properties or new market conditions may not be reflected accurately.
- Currency is treated as-is from the dataset (₹ used as a display label);
  update the currency symbol in `app.py` if your dataset uses a different one.

---

## 🛠️ Possible Extensions

- Add geographic/location fields (city, zip code) with location-based pricing
- Swap in XGBoost/LightGBM for potentially higher accuracy
- Add SHAP values for per-prediction explainability
- Persist trained models to disk (`joblib`) to skip retraining on restart
- Add user authentication and saved-search history

---

## 📄 License

Free to use and modify for personal, educational, or portfolio purposes.
