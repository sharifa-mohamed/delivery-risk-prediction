import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

from pipeline_definition import LateDeliveryPreprocessingPipeline

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    layout="wide"
)

BASE_DIR = Path("registered_models")
DEFAULT_DATA_PATH = Path("data/APL_Logistics.csv")

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_assets():
    pipeline = joblib.load(
        BASE_DIR /
        "data_preparation_pipeline" /
        "late_delivery_pipeline.pkl"
    )

    lr_model = joblib.load(
        BASE_DIR /
        "base_lr_registered" /
        "version_1_None" /
        "model_artifacts" /
        "artifacts" /
        "model.pkl"
    )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(
        str(
            BASE_DIR /
            "adv_xgb_registered" /
            "version_1_None" /
            "model_artifacts" /
            "artifacts" /
            "model.ubj"
        )
    )

    rf_model = joblib.load(
        BASE_DIR /
        "adv_rforest_registered" /
        "version_1_None" /
        "model_artifacts" /
        "artifacts" /
        "model.pkl"
    )

    return pipeline, lr_model, xgb_model, rf_model


pipeline, lr_model, xgb_model, rf_model = load_assets()


# ============================================================
# HELPERS & CACHED INFERENCE
# ============================================================

def risk_category(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    return "High Risk"


@st.cache_data(show_spinner="Running model inference...")
def get_prediction_cached(df, model_name):
    """
    Caches model predictions. Streamlit hashes the input dataframe 
    and model_name string. If neither changes, the cached results 
    are returned instantly.
    """
    if model_name == "Logistic Regression":
        X = pipeline.transform_lr(df)
        probs = lr_model.predict_proba(X)[:, 1]
    elif model_name == "Random Forest":
        X = pipeline.transform_tree(df)
        probs = rf_model.predict_proba(X)[:, 1]
    else:
        X = pipeline.transform_tree(df)
        probs = xgb_model.predict_proba(X)[:, 1]
    return probs.tolist()  # Convert to list/numpy array for safe caching serialization


def create_risk_driver_table(df):
    candidate_columns = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Customer Segment",
        "Order Status",
        "Category Name"
    ]

    available = [col for col in candidate_columns if col in df.columns]

    rows = []
    for col in available:
        temp = (
            df.groupby(col)["Late_Delivery_Probability"]
            .mean()
            .reset_index()
        )
        temp.columns = ["Factor", "Average_Risk"]
        temp["Feature"] = col
        rows.append(temp)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# SIDEBAR CONFIGURATION (ALL FILTERS & CONTROLS)
# ============================================================

st.sidebar.title("Dashboard Controls")

# 1. Data Source Section
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain Dataset override",
    type=["csv"]
)

df = None
try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding="cp1252")
        st.sidebar.success("Using uploaded dataset.")
    elif DEFAULT_DATA_PATH.exists():
        df = pd.read_csv(DEFAULT_DATA_PATH, encoding="cp1252")
        st.sidebar.info("Automatically loaded local APL Logistics dataset.")
    else:
        st.sidebar.warning(f"Local file '{DEFAULT_DATA_PATH}' not found. Please upload a CSV file.")
except Exception as e:
    st.sidebar.error(f"Error loading data: {str(e)}")


if df is not None:
    try:
        # 2. Dashboard Filters Section
        st.sidebar.header("Dashboard Filters")
        
        model_choice = st.sidebar.selectbox(
            "Prediction Model",
            ["Logistic Regression", "XGBoost", "Random Forest"]
        )
        
        risk_threshold = st.sidebar.slider(
            "High Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01
        )

        # Using the new cached prediction function
        probabilities = get_prediction_cached(df, model_choice)
        
        results = df.copy()
        results["Late_Delivery_Probability"] = np.round(probabilities, 4)
        results["Risk_Category"] = results["Late_Delivery_Probability"].apply(risk_category)

        # Categorical Sub-filters
        filtered = results.copy()

        if "Shipping Mode" in filtered.columns:
            modes = sorted(filtered["Shipping Mode"].dropna().unique())
            selected_modes = st.sidebar.multiselect("Shipping Mode", modes, default=modes)
            filtered = filtered[filtered["Shipping Mode"].isin(selected_modes)]

        if "Market" in filtered.columns:
            markets = sorted(filtered["Market"].dropna().unique())
            selected_markets = st.sidebar.multiselect("Market", markets, default=markets)
            filtered = filtered[filtered["Market"].isin(selected_markets)]

        if "Customer Segment" in filtered.columns:
            segments = sorted(filtered["Customer Segment"].dropna().unique())
            selected_segments = st.sidebar.multiselect("Customer Segment", segments, default=segments)
            filtered = filtered[filtered["Customer Segment"].isin(selected_segments)]


        # ============================================================
        # MAIN APP CONTENT DISPLAY
        # ============================================================
        st.title("Machine Learning Based Late Delivery Risk Prediction")
        st.markdown(
            f"""
            Predict shipment delays before delivery and identify
            high risk orders requiring operational intervention.  
            *Currently evaluating risks using the **{model_choice}** model.*
            """
        )

        st.subheader("Dataset Preview")
        st.dataframe(filtered.head(), width='stretch')
        st.markdown("---")

        # =================================================
        # KPI SECTION
        # =================================================
        st.header("Delay Risk Overview")
        total_orders = len(filtered)
        
        high_risk = (filtered["Late_Delivery_Probability"] >= risk_threshold).sum()
        medium_risk = ((filtered["Late_Delivery_Probability"] >= 0.30) & 
                       (filtered["Late_Delivery_Probability"] < risk_threshold)).sum()
        low_risk = (filtered["Late_Delivery_Probability"] < 0.30).sum()
        avg_risk = filtered["Late_Delivery_Probability"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders", f"{total_orders:,}")
        c2.metric("High Risk Orders", f"{high_risk:,}")
        c3.metric("Medium Risk Orders", f"{medium_risk:,}")
        c4.metric("Average Risk", f"{avg_risk:.2%}" if not np.isnan(avg_risk) else "0.00%")

        # =================================================
        # RISK DISTRIBUTION
        # =================================================
        st.subheader("Overall Risk Distribution")
        fig = px.histogram(
            filtered,
            x="Late_Delivery_Probability",
            nbins=25
        )
        st.plotly_chart(fig, width='stretch')

        # =================================================
        # ORDER LEVEL RISK PREDICTION
        # =================================================
        st.header("Order Level Risk Prediction")
        st.dataframe(filtered, width='stretch')

        if not filtered.empty:
            selected_index = st.selectbox("Select Order", filtered.index)
            selected_order = filtered.loc[[selected_index]]

            st.subheader("Selected Order Risk Score")
            c_score, c_cat = st.columns(2)
            c_score.metric("Late Delivery Probability", f"{selected_order['Late_Delivery_Probability'].iloc[0]:.2%}")
            c_cat.metric("Risk Category", selected_order["Risk_Category"].iloc[0])

            st.write(selected_order.T)
        else:
            st.warning("No data matches current filter criteria.")

        # =================================================
        # REGION ANALYSIS
        # =================================================
        st.header("Region & Mode Risk Analysis")
        
        if "Order Region" in filtered.columns and not filtered.empty:
            region_risk = (
                filtered.groupby("Order Region")["Late_Delivery_Probability"]
                .mean()
                .reset_index()
            )
            fig_region = px.bar(
                region_risk,
                x="Order Region",
                y="Late_Delivery_Probability",
                title="Average Risk by Region"
            )
            st.plotly_chart(fig_region, width='stretch')

        if "Market" in filtered.columns and "Order Region" in filtered.columns and not filtered.empty:
            heatmap = (
                filtered
                .pivot_table(
                    values="Late_Delivery_Probability",
                    index="Market",
                    columns="Order Region",
                    aggfunc="mean"
                )
            )
            fig_heatmap = px.imshow(
                heatmap,
                aspect="auto",
                title="Regional Risk Heatmap"
            )
            st.plotly_chart(fig_heatmap, width='stretch')

        # =================================================
        # SHIPPING MODE COMPARISON
        # =================================================
        if "Shipping Mode" in filtered.columns and not filtered.empty:
            mode_risk = (
                filtered.groupby("Shipping Mode")["Late_Delivery_Probability"]
                .mean()
                .reset_index()
            )
            fig_mode = px.bar(
                mode_risk,
                x="Shipping Mode",
                y="Late_Delivery_Probability",
                title="Shipping Mode Risk Comparison"
            )
            st.plotly_chart(fig_mode, width='stretch')

        # =================================================
        # KEY RISK DRIVERS
        # =================================================
        st.header("Key Risk Drivers")
        driver_df = create_risk_driver_table(filtered)

        if len(driver_df):
            driver_df = driver_df.sort_values("Average_Risk", ascending=False)
            st.dataframe(driver_df.head(25), width='stretch')

        # =================================================
        # OPERATIONS ACTION PANEL
        # =================================================
        st.header("Operations Action Panel")
        high_risk_orders = filtered[
            filtered["Late_Delivery_Probability"] >= risk_threshold
        ].sort_values("Late_Delivery_Probability", ascending=False)

        st.metric("Orders Requiring Immediate Attention", len(high_risk_orders))
        st.dataframe(high_risk_orders, width='stretch')

        # =================================================
        # DOWNLOADS
        # =================================================
        st.header("Export Results")
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="late_delivery_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.code(traceback.format_exc())