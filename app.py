"""
================================================================================
 HOME VALUE ADVISOR — House Price Prediction & Recommendation System
What this app does:
    1. Loads and explores the housing dataset (EDA)
    2. Trains and compares multiple regression models
    3. Lets a user (buyer/seller/agent) enter property features
    4. Predicts the market price using the best-performing model
    5. Explains the prediction with interactive charts
    6. Gives plain-English recommendations to the user
================================================================================
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Home Value Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "Housing.csv"
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating",
               "airconditioning", "prefarea"]
FURNISH_COL = "furnishingstatus"
TARGET = "price"

# ==============================================================================
# STYLING
# ==============================================================================
st.markdown(
    """
    <style>
    /* ---- Global readability boost: no browser zoom needed ---- */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1 { font-size: 2.3rem !important; }
    h2, h3 { font-size: 1.6rem !important; }
    p, li, label, span { font-size: 1.05rem !important; }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        font-size: 1.05rem !important;
    }

    /* Metric widgets (st.metric) */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"] {
        font-size: 1rem !important;
    }
    [data-testid="stDataFrame"] * {
        font-size: 1rem !important;
    }

    /* Radio / selectbox / slider labels */
    [data-testid="stWidgetLabel"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    .metric-card {
        background-color: #f5f7fa;
        border-radius: 12px;
        padding: 22px;
        border: 1px solid #e2e6ea;
    }
    .big-price {
        font-size: 3rem;
        font-weight: 800;
        color: #1a5d1a;
    }
    .reco-box {
        background-color: #eef6ee;
        border-left: 5px solid #2e7d32;
        color: #000000;  
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 1.05rem;
    }
    .warn-box {
        background-color: #fdf3e7;
        border-left: 5px solid #e08a00;
        color: #8b4513; 
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# DATA LOADING
# ==============================================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data
def prepare_features(df: pd.DataFrame):
    """Encode categorical columns into a model-ready numeric frame."""
    work = df.copy()
    for c in BINARY_COLS:
        work[c] = work[c].map({"yes": 1, "no": 0})
    work = pd.get_dummies(work, columns=[FURNISH_COL], drop_first=True)
    # Ensure both dummy columns exist even if a category is missing
    for col in ["furnishingstatus_semi-furnished", "furnishingstatus_unfurnished"]:
        if col not in work.columns:
            work[col] = 0
    feature_cols = [c for c in work.columns if c != TARGET]
    return work, feature_cols


@st.cache_resource
def train_all_models(df: pd.DataFrame):
    """Train several regressors, evaluate them, and return everything the
    rest of the app needs (models, scaler, metrics, feature columns)."""
    work, feature_cols = prepare_features(df)
    X = work[feature_cols]
    y = work[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=100, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=8, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7),
    }

    results = []
    trained_models = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")
        results.append(
            {
                "Model": name,
                "R2 Score": round(r2, 4),
                "MAE": round(mae, 0),
                "RMSE": round(rmse, 0),
                "CV R2 (mean)": round(cv_scores.mean(), 4),
            }
        )
        trained_models[name] = model

    metrics_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    best_model_name = metrics_df.iloc[0]["Model"]

    return {
        "models": trained_models,
        "scaler": scaler,
        "metrics_df": metrics_df,
        "feature_cols": feature_cols,
        "best_model_name": best_model_name,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
    }


# ==============================================================================
# LOAD + TRAIN
# ==============================================================================
try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Please place the Housing.csv dataset "
        "in the same folder as app.py and restart the app."
    )
    st.stop()

bundle = train_all_models(raw_df)
models = bundle["models"]
scaler = bundle["scaler"]
metrics_df = bundle["metrics_df"]
feature_cols = bundle["feature_cols"]
best_model_name = bundle["best_model_name"]

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🏠 Home Value Advisor")
st.sidebar.caption("ML-powered house price prediction & buying guidance")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview & EDA", "🤖 Model Comparison", "🔮 Predict & Get Recommendations"],
)
st.sidebar.markdown("---")
st.sidebar.metric("Dataset size", f"{len(raw_df)} homes")
st.sidebar.metric("Best model", best_model_name,
                   f"R² = {metrics_df.iloc[0]['R2 Score']}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built for buyers, sellers, and real-estate agents to get a fast, "
    "data-driven estimate of a home's fair market value."
)

# ==============================================================================
# PAGE 1 — OVERVIEW & EDA
# ==============================================================================
if page == "📊 Overview & EDA":
    st.title("📊 Housing Market — Exploratory Data Analysis")
    st.write(
        "This dashboard explores the underlying housing dataset used to train "
        "the prediction models below, so you can understand *why* the model "
        "predicts what it predicts."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Homes in dataset", len(raw_df))
    c2.metric("Average price", f"₹{raw_df['price'].mean():,.0f}")
    c3.metric("Median price", f"₹{raw_df['price'].median():,.0f}")
    c4.metric("Avg. area (sqft)", f"{raw_df['area'].mean():,.0f}")

    st.markdown("### Raw data sample")
    st.dataframe(raw_df.head(15), use_container_width=True, height=350)

    st.markdown("### Missing values check")
    missing = raw_df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values found in the dataset — data is clean.")
    else:
        st.warning("Missing values detected:")
        st.dataframe(missing[missing > 0])

    st.markdown("---")
    st.markdown("### Price distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            raw_df, x="price", nbins=40, title="Distribution of House Prices",
            color_discrete_sequence=["#2e7d32"],
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            raw_df, y="price", points="outliers",
            title="Price Spread & Outliers", color_discrete_sequence=["#1565c0"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Price vs. Area")
    fig = px.scatter(
        raw_df, x="area", y="price", color="furnishingstatus",
        size="bedrooms", hover_data=["bathrooms", "stories"],
        title="Price vs Area (bubble size = bedrooms)",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation heatmap (numeric features)")
    numeric_df = raw_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdYlGn",
        title="Correlation Between Numeric Features",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How categorical amenities affect price")
    cat_choice = st.selectbox(
        "Choose a feature to compare against price",
        ["mainroad", "guestroom", "basement", "hotwaterheating",
         "airconditioning", "prefarea", "furnishingstatus"],
    )
    fig = px.box(
        raw_df, x=cat_choice, y="price", color=cat_choice,
        title=f"Price Distribution by '{cat_choice}'",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bedrooms, bathrooms & stories vs. price")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.box(raw_df, x="bedrooms", y="price", title="Bedrooms vs Price")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(raw_df, x="bathrooms", y="price", title="Bathrooms vs Price")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.box(raw_df, x="stories", y="price", title="Stories vs Price")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 2 — MODEL COMPARISON
# ==============================================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Multi-Model Comparison")
    st.write(
        "Seven regression algorithms were trained on an 80/20 train-test "
        "split and evaluated with 5-fold cross-validation. Higher R² and "
        "lower MAE/RMSE indicate a better model."
    )

    st.markdown("### Performance table")
    st.dataframe(
        metrics_df.style.highlight_max(subset=["R2 Score", "CV R2 (mean)"], color="#c8e6c9")
        .highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9"),
        use_container_width=True,
    )

    st.success(
        f"🏆 Best performing model: **{best_model_name}** "
        f"(R² = {metrics_df.iloc[0]['R2 Score']}, "
        f"CV R² = {metrics_df.iloc[0]['CV R2 (mean)']})"
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            metrics_df, x="Model", y="R2 Score", color="R2 Score",
            title="R² Score by Model (higher is better)",
            color_continuous_scale="Greens",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            metrics_df, x="Model", y="RMSE", color="RMSE",
            title="RMSE by Model (lower is better)",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Predicted vs. Actual price (best model)")
    best_model = models[best_model_name]
    X_test_s = scaler.transform(bundle["X_test"])
    preds = best_model.predict(X_test_s)
    comp_df = pd.DataFrame(
        {"Actual": bundle["y_test"].values, "Predicted": preds}
    )
    fig = px.scatter(
        comp_df, x="Actual", y="Predicted",
        title=f"{best_model_name}: Predicted vs Actual Price",
        trendline="ols",
    )
    max_val = max(comp_df["Actual"].max(), comp_df["Predicted"].max())
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                   name="Perfect prediction", line=dict(dash="dash", color="gray"))
    )
    st.plotly_chart(fig, use_container_width=True)

    if hasattr(best_model, "feature_importances_"):
        st.markdown("### What drives the best model's predictions?")
        imp_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": best_model.feature_importances_}
        ).sort_values("Importance", ascending=True)
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title=f"Feature Importance — {best_model_name}",
            color="Importance", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(best_model, "coef_"):
        st.markdown("### What drives the best model's predictions?")
        imp_df = pd.DataFrame(
            {"Feature": feature_cols, "Coefficient": best_model.coef_}
        ).sort_values("Coefficient", key=abs, ascending=True)
        fig = px.bar(
            imp_df, x="Coefficient", y="Feature", orientation="h",
            title=f"Feature Coefficients — {best_model_name}",
            color="Coefficient", color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3 — PREDICT & RECOMMEND
# ==============================================================================
else:
    st.title("🔮 Predict Your Home's Price")
    st.write(
        "Enter the property details below. The model will estimate a fair "
        "market price and explain what is helping or hurting the value, "
        "with personalized recommendations."
    )

    model_choice = st.selectbox(
        "Choose prediction model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        help="Defaults to the best-performing model, but you can compare others.",
    )

    st.markdown("### Property details")
    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input(
            "Area (sqft)", min_value=500, max_value=20000,
            value=int(raw_df["area"].median()), step=50,
        )
        bedrooms = st.slider("Bedrooms", 1, 6, int(raw_df["bedrooms"].median()))
        bathrooms = st.slider("Bathrooms", 1, 4, int(raw_df["bathrooms"].median()))
    with col2:
        stories = st.slider("Stories", 1, 4, int(raw_df["stories"].median()))
        parking = st.slider("Parking spots", 0, 3, int(raw_df["parking"].median()))
        furnishingstatus = st.selectbox(
            "Furnishing status", ["furnished", "semi-furnished", "unfurnished"]
        )
    with col3:
        mainroad = st.radio("On main road?", ["yes", "no"], horizontal=True)
        guestroom = st.radio("Guest room?", ["yes", "no"], horizontal=True)
        basement = st.radio("Basement?", ["yes", "no"], horizontal=True)
        hotwaterheating = st.radio("Hot water heating?", ["yes", "no"], horizontal=True)
        airconditioning = st.radio("Air conditioning?", ["yes", "no"], horizontal=True)
        prefarea = st.radio("Preferred area?", ["yes", "no"], horizontal=True)

    predict_btn = st.button("🔍 Predict Price", type="primary", use_container_width=True)

    def build_input_row(area, bedrooms, bathrooms, stories, mainroad, guestroom,
                         basement, hotwaterheating, airconditioning, parking,
                         prefarea, furnishingstatus):
        row = {
            "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": 1 if mainroad == "yes" else 0,
            "guestroom": 1 if guestroom == "yes" else 0,
            "basement": 1 if basement == "yes" else 0,
            "hotwaterheating": 1 if hotwaterheating == "yes" else 0,
            "airconditioning": 1 if airconditioning == "yes" else 0,
            "parking": parking,
            "prefarea": 1 if prefarea == "yes" else 0,
            "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
            "furnishingstatus_unfurnished": 1 if furnishingstatus == "unfurnished" else 0,
        }
        return pd.DataFrame([row])[feature_cols]

    if predict_btn or "last_prediction" in st.session_state:
        input_df = build_input_row(
            area, bedrooms, bathrooms, stories, mainroad, guestroom,
            basement, hotwaterheating, airconditioning, parking,
            prefarea, furnishingstatus,
        )
        input_scaled = scaler.transform(input_df)
        model = models[model_choice]
        predicted_price = float(model.predict(input_scaled)[0])
        st.session_state["last_prediction"] = predicted_price

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.4])

        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Estimated Market Price")
            st.markdown(
                f'<div class="big-price">₹{predicted_price:,.0f}</div>',
                unsafe_allow_html=True,
            )
            avg_price = raw_df["price"].mean()
            diff_pct = (predicted_price - avg_price) / avg_price * 100
            st.caption(
                f"{'Above' if diff_pct >= 0 else 'Below'} the dataset average of "
                f"₹{avg_price:,.0f} by {abs(diff_pct):.1f}%"
            )
            st.caption(f"Model used: **{model_choice}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # Percentile of this price in the market
            percentile = (raw_df["price"] < predicted_price).mean() * 100
            st.markdown("#### Market position")
            st.progress(min(max(percentile / 100, 0.0), 1.0))
            st.caption(f"This estimate is higher than {percentile:.0f}% of homes in the dataset.")

        with res_col2:
            st.markdown("#### Price sensitivity: Area")
            area_range = np.linspace(
                max(500, area - 3000), area + 3000, 25
            ).astype(int)
            sens_rows = []
            for a in area_range:
                r = build_input_row(
                    a, bedrooms, bathrooms, stories, mainroad, guestroom,
                    basement, hotwaterheating, airconditioning, parking,
                    prefarea, furnishingstatus,
                )
                p = model.predict(scaler.transform(r))[0]
                sens_rows.append({"Area": a, "Predicted Price": p})
            sens_df = pd.DataFrame(sens_rows)
            fig = px.line(
                sens_df, x="Area", y="Predicted Price",
                title="How price changes as area changes (other features fixed)",
                markers=True,
            )
            fig.add_vline(x=area, line_dash="dash", line_color="green",
                           annotation_text="Your input")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Impact of each amenity on your predicted price")
        st.caption(
            "Each bar shows how much the predicted price would change if that "
            "single feature were flipped, holding everything else constant."
        )
        toggle_features = [
            ("Main road access", "mainroad", mainroad),
            ("Guest room", "guestroom", guestroom),
            ("Basement", "basement", basement),
            ("Hot water heating", "hotwaterheating", hotwaterheating),
            ("Air conditioning", "airconditioning", airconditioning),
            ("Preferred area", "prefarea", prefarea),
        ]
        impact_rows = []
        for label, key, current_val in toggle_features:
            flipped_val = "no" if current_val == "yes" else "yes"
            kwargs = dict(
                area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories,
                mainroad=mainroad, guestroom=guestroom, basement=basement,
                hotwaterheating=hotwaterheating, airconditioning=airconditioning,
                parking=parking, prefarea=prefarea, furnishingstatus=furnishingstatus,
            )
            kwargs[key] = flipped_val
            r = build_input_row(**kwargs)
            p_flipped = model.predict(scaler.transform(r))[0]
            impact = predicted_price - p_flipped if current_val == "yes" else p_flipped - predicted_price
            impact_rows.append({"Amenity": label, "Value added (₹)": impact})
        impact_df = pd.DataFrame(impact_rows).sort_values("Value added (₹)")
        fig = px.bar(
            impact_df, x="Value added (₹)", y="Amenity", orientation="h",
            color="Value added (₹)", color_continuous_scale="RdYlGn",
            title="Estimated value each amenity adds to this specific home",
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------- RECOMMENDATIONS --------------------
        st.markdown("### 💡 Recommendations")

        recos = []
        warns = []

        top_gain = impact_df.iloc[-1]
        if top_gain["Value added (₹)"] > 50000:
            recos.append(
                f"Adding/upgrading **{top_gain['Amenity']}** could increase this "
                f"property's value by roughly ₹{top_gain['Value added (₹)']:,.0f}. "
                "This is the single highest-leverage improvement available."
            )

        if furnishingstatus == "unfurnished":
            furn_row = build_input_row(
                area, bedrooms, bathrooms, stories, mainroad, guestroom,
                basement, hotwaterheating, airconditioning, parking,
                prefarea, "furnished",
            )
            p_furnished = model.predict(scaler.transform(furn_row))[0]
            gain = p_furnished - predicted_price
            if gain > 0:
                recos.append(
                    f"Fully furnishing this home could raise its value by about "
                    f"₹{gain:,.0f}. Consider this if you're preparing to sell."
                )

        if airconditioning == "no":
            recos.append(
                "This home has no air conditioning — a common buyer deal-breaker "
                "in warmer climates. It's often a cost-effective upgrade for resale."
            )

        if parking == 0:
            recos.append(
                "No parking space is listed. Adding even one parking spot tends "
                "to meaningfully improve buyer interest and price."
            )

        if percentile > 85:
            warns.append(
                f"This estimate is in the **top {100 - percentile:.0f}%** of the "
                "market — verify comparable sales nearby before pricing this high, "
                "as it may be harder to find a matching buyer quickly."
            )
        elif percentile < 15:
            warns.append(
                f"This estimate is in the **bottom {percentile:.0f}%** of the "
                "market. If this is your own home, double-check whether all "
                "amenities were entered correctly — you may be undervaluing it."
            )

        if not recos:
            recos.append(
                "This property is already well-optimized across the amenities "
                "in this dataset — no single upgrade stands out as high-impact."
            )

        for r in recos:
            st.markdown(f'<div class="reco-box">✅ {r}</div>', unsafe_allow_html=True)
        for w in warns:
            st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

        st.caption(
            "Disclaimer: This tool provides a statistical estimate based on "
            "historical data and is not a substitute for a professional "
            "property appraisal."
        )