# Late Delivery Risk Prediction Dashboard

An interactive Streamlit web application designed to predict supply chain shipment delays before they occur. By providing order-level risk probabilities, this dashboard empowers logistics teams to perform proactive operational interventions and optimize shipping strategies.

## 🚀 System Architecture Overview

This project bridges experimental model research with local application deployment using a structured, four-tier framework:
1. **Experimental Preprocessing (`late_delivery_preprocessing_pipeline.ipynb`)**: Implements strict data cleansing, leakage prevention (dropping direct indicators like `Order Status` and target `Delivery Status`), advanced leak-proof feature mapping, target encoding, and iterative multicollinearity reduction via Variance Inflation Factor (VIF) filtering.
2. **Experimental Modeling & Training (`late_delivery_modeling_pipeline.ipynb`)**: Pulls the clean preprocessed runs, establishes metric scoring frameworks, runs randomized hyperparameter tuning search steps, evaluates performance criteria, and saves the final serialized metrics to MLflow.
3. **Model Management (`download_latest_model_assets.py`)**: Connects to a remote MLflow server via DagsHub, fetches the latest model versions and preprocessing artifacts, dynamically creates the `registered_models/` directory structure, and aggregates runtime package dependencies.
4. **Data Pipeline Definition (`pipeline_definition.py`)**: A serialized, custom standalone pipeline (`LateDeliveryPreprocessingPipeline`) that encapsulates feature engineering, categorical mappings, missing value imputations, and outlier scaling to guarantee production data mirrors training conditions.
5. **Application Layer (`app.py`)**: The interactive multi-model user interface built with Streamlit, rendering real-time inference, risk KPIs, and regional risk mapping.

---

## 📁 Repository Structure
```text
├── notebooks/                  # Experimental Phase (Development & Research)
│   ├── late_delivery_preprocessing_pipeline.ipynb  # Leakage clearance, feature maps, & VIF drops
│   └── late_delivery_modeling_pipeline.ipynb       # Hyperparameter tuning (LR, RF, XGB) & MLflow log
├── registered_models/          # Deployment Phase (Generated dynamically via MLflow sync script)
│   ├── data_preparation_pipeline/
│   │   └── late_delivery_pipeline.pkl
│   ├── adv_xgb_registered/
│   │   └── version_1_None/model_artifacts/artifacts/model_artifacts/model.ubj
│   ├── adv_rforest_registered/
│   │   └── version_1_None/model_artifacts/artifacts/model.pkl
│   └── base_lr_registered/
│       └── version_1_None/model_artifacts/artifacts/model.pkl
├── data/
│   └── APL_Logistics.csv      # Default fallback local dataset (CP1252 encoded)
├── app.py                      # Main Streamlit Dashboard application
├── pipeline_definition.py      # Core data transformation pipeline class architecture
├── download_latest_model_assets.py # Automation script syncing models/artifacts from MLflow
├── requirements.txt            # System dependencies
└── README.md                   # Project documentation
```
---

## 🔄 Notebook Experimental Workflows

### 1. Preprocessing Pipeline (`late_delivery_preprocessing_pipeline.ipynb`)
To ensure total integrity, the dataset goes through rigorous transformations structured to run exclusively without data leakage:
* Leakage and Redundancy Clearance: Direct indicators such as Order Status and columns heavily tied to actual post-shipment measurements are cleared immediately. Redundant categorical variables are scrubbed out to prevent synthetic target mappings.
* Leak-Proof Granular Features:
    * Shipping Pressure Index: Calculated safely using Order Item Quantity / Days for shipment (scheduled) with adjustments for zero days.
    * Mode Urgency Flags: Flags premium modes (Same Day, First Class) as high urgency metrics.
    * Complex/Monetary Stress Indexing: Models financial weight using order_complexity_score and discount_per_item.
    * Group Aggregations (Leak-Free Maps): Aggregates frequencies, sales means, and discounts for Customers, Products, Regions, and Departments grouped strictly within the training fold. Unseen validation groups are matched to baseline training means via fallback .fillna() rules.
* Outlier Mitigation: Numeric values are bounded safely between their calculated $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$ spans.
* Iterative VIF Multicollinearity Reduction: Feeds scaled features to an iterative loop calculating Variance Inflation Factors. Any numeric feature scoring above a threshold=10.0 (such as department_avg_discount or high_value_order) is progressively dropped until only independent variables remain for the Linear Pipeline.

### 2. Modeling & Training Pipeline (`late_delivery_modeling_pipeline.ipynb`)
Trains three model classes with hyperparameter optimizations monitored under tracking scopes:
* Baseline Logistic Regression: Fitted using features chosen from the VIF reduction step to secure optimal coefficient estimates.
* Default and Tuned Random Forests: Evaluates baseline ensembles against optimized forests via RandomizedSearchCV tracking OOB scores.
* Advanced XGBoost: Configured by evaluating positive/negative label ratios to establish structural scaling balances (scale_pos_weight), optimizing classification thresholds against severe delays.
* MLflow Tracking: Pushes metrics (accuracy, precision, recall, f1_score, roc_auc_score), feature importances, and parameters straight to DagsHub.

---

## 🛑 Prerequisites & Virtual Environment Setup

Ensure you have Python 3.9+ installed locally.

### 1. Clone the repository and navigate into it
git clone <your-repository-url>
cd <your-repository-name>

### 2. Configure a clean virtual environment
# Create the environment
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

### 3. Install required base packages
pip install -r requirements.txt

---

## 🔄 Synchronizing Model Assets (MLflow Integration)

Before spinning up the application dashboard, you must run the automation script to download your production model architectures, weights, and processing files tracked on MLflow.

Run the asset downloader:
python download_latest_model_assets.py

### Script Behavior & Mechanics:
* Authentication: The script securely connects to your DagsHub-hosted MLflow tracking server (https://dagshub.com/sharifa-mohamed/late_delivery_prediction.mlflow) using your environment credentials.
* Dynamic Pipeline Ingestion: It locates the latest run explicitly tagged as data_preparation_pipeline within the late_delivery_prediction experiment, downloading late_delivery_pipeline.pkl into registered_models/data_preparation_pipeline/.
* Model Version Resolution: It searches your Model Registry for the latest versions of your registered models (adv_xgb_registered, adv_rforest_registered, and base_lr_registered).
* Artifact Downloading: It creates structured subdirectories inside registered_models/ and downloads model binaries (model.pkl for LR and RF; universal binary format model.ubj for XGBoost).
* Dependency Management: It extracts dependencies for each specific model, creating a localized requirements.txt inside each model version's directory and outputting a single consolidated unified_requirements.txt file at the root of registered_models/.

---

## 💻 Pipeline Architecture & Inference Execution

Data fed into the application goes through a precise multi-stage transformation handled by pipeline_definition.py to eliminate data leakage and ensure compliance with model constraints:

* Structural Cleansing & Imputation: Redundant features are dropped dynamically based on dropped_columns, and missing records are filled using computed training dataset medians/modes stored in imputation_values.
* Feature Engineering:
    * Shipping Pressure: Computed as Order Item Quantity / Days for shipment (scheduled) (with zero-division protection).
    * Operational Flags: High urgency modes (is_high_urgency_mode) and bulk orders (is_bulk_order) are flagged numerically.
    * Value & Stress Metrics: order_complexity_score, discount_per_item, and high_value_order capture shipment weight and financial stakes.
    * Congestion Factors: regional_congestion_score maps regional volume shares with global mean fallbacks.
* Categorical & Target Mapping: Frequency and value maps are applied to high-cardinality values (Order Customer Id, Product Name, Order Region, Department Name) using cached global fallbacks for unseen production records.
* Encoding & Outlier Mitigation: Categorical data undergoes structural encoding via the stored encoder instance, and numeric feature columns are clipped strictly using training data outlier thresholds (outlier_bounds).
* Model-Specific Downstream Transformations:
    * Linear Pipeline (transform_lr): Applies standard scaling via the fit scaler instance over the precise 26 numeric columns used in training and reindexes columns to selected_lr_columns to avoid VIF-flagged multicollinearity.
    * Tree Pipeline (transform_tree): Bypasses scaling and slices columns directly to match tree_columns configurations required by the Random Forest and XGBoost architectures.

---

## 📊 Running the Streamlit Dashboard

Once your binary assets have finished downloading into the /registered_models folder, start your dashboard application locally:

streamlit run app.py

### Application Runtime Workflow:
* Asset Loading: Caches the preprocessing pipeline and the three model binaries using @st.cache_resource for swift initialization.
* Data Ingestion: Searches for a local file at data/APL_Logistics.csv encoded in cp1252 format. Users can override this dataset in real-time using the sidebar file uploader element.
* Cached Batch Inference: Utilizes @st.cache_data to serialize and cache predictions based on the active dataframe and selected model, speeding up multi-filter rendering.
* Live Interactive Analytics: Plots global metrics, interactive Plotly visualizations, regional risk heatmaps, individual drill-down order audits, and operational panels filtering orders requiring immediate dispatch interventions.
