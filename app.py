import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import shap
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# Page Configuration
# ============================================================
#
# This section defines the title, icon, and layout of the
# Streamlit fraud intelligence dashboard.

st.set_page_config(
    page_title="FinLora Fraud Intelligence Dashboard",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# Dashboard Title
# ============================================================
#
# This section introduces the FinLora Fraud Intelligence
# Dashboard and identifies its purpose for fraud analysis.

st.title("FinLora Fraud Intelligence Dashboard")
st.caption("Machine-learning insights for transaction fraud analysis")


# ============================================================
# Load Saved Fraud Detection Model and Preprocessing Pipeline
# ============================================================
#
# This section loads the trained Random Forest fraud detection
# model and the preprocessing pipeline saved during the
# modeling stage.
#
# Keeping the model and preprocessing pipeline in the same
# project folder allows the Streamlit dashboard to use the
# same trained components developed during model training.
#
# The saved model is used to generate fraud predictions,
# while the preprocessing pipeline prepares transaction data
# in the format expected by the model.

random_forest_model = joblib.load(
    "random_forest_fraud_model.pkl"
)

preprocessing_pipeline = joblib.load(
    "Finlora_Preprocessing_Pipeline.pkl"
)


# ============================================================
# Load Test Data
# ============================================================
#
# This section loads the test transaction data and the
# corresponding actual fraud labels saved during the
# modeling stage.
#
# The test data is used by the Streamlit dashboard to
# generate predictions using the saved Random Forest model.
#
# Loading the data directly into the dashboard ensures that
# the application does not depend on variables created inside
# the Jupyter Notebook.

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()


# ============================================================
# Generate Fraud Predictions
# ============================================================
#
# This section uses the saved preprocessing pipeline and
# Random Forest model to generate predictions for the test
# transactions.
#
# The preprocessing pipeline transforms the test data into the
# same format used during model training.
#
# The Random Forest model then predicts whether each
# transaction is potentially fraudulent or legitimate.
#
# Fraud probabilities are also generated so that analysts
# can see the model's estimated probability of fraud.

X_test_processed = preprocessing_pipeline.transform(X_test)

fraud_predictions = random_forest_model.predict(
    X_test_processed
)

fraud_probabilities = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# Model Loading Verification
# ============================================================
#
# This section confirms that the saved Random Forest model
# and preprocessing pipeline have been loaded successfully.

st.success(
    "Fraud detection model and preprocessing pipeline loaded successfully."
)


# ============================================================
# Transaction Overview
# ============================================================
#
# This section provides a high-level summary of the test
# transactions analyzed by the fraud detection model.
#
# The displayed values are calculated directly from the
# generated model predictions so that the dashboard remains
# consistent with the underlying test data.

total_transactions = len(fraud_predictions)
potentially_fraudulent = int(fraud_predictions.sum())
legitimate_transactions = int((fraud_predictions == 0).sum())

st.header("Transaction Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test Transactions", f"{total_transactions:,}")

with col2:
    st.metric("Potentially Fraudulent", f"{potentially_fraudulent:,}")

with col3:
    st.metric("Legitimate", f"{legitimate_transactions:,}")


# ============================================================
# Fraud vs. Legitimate Transaction Distribution
# ============================================================
#
# This section displays the distribution of transactions
# classified as potentially fraudulent and legitimate by
# the selected Random Forest model.

st.header("Fraud vs. Legitimate Transactions")

distribution_data = {
    "Transaction Type": [
        "Potentially Fraudulent",
        "Legitimate"
    ],
    "Count": [
        potentially_fraudulent,
        legitimate_transactions
    ]
}

st.bar_chart(
    distribution_data,
    x="Transaction Type",
    y="Count"
)


# ============================================================
# Fraud Pattern Analysis
# ============================================================
#
# This section highlights recurring characteristics observed
# among transactions flagged as potentially fraudulent by the
# selected Random Forest model.
#
# The counts are calculated directly from the model-flagged
# transactions displayed in the dashboard.
#
# These patterns describe characteristics found among
# model-flagged transactions. They should not be interpreted
# as evidence that any individual characteristic directly
# causes fraud.

flagged_transactions = X_test.copy()
flagged_transactions["fraud_probability"] = fraud_probabilities
flagged_transactions["predicted_fraud"] = fraud_predictions
flagged_transactions["actual_fraud"] = y_test.values

flagged_transactions = flagged_transactions[
    flagged_transactions["predicted_fraud"] == 1
].copy()

location_mismatch_count = int(
    flagged_transactions["location_mismatch"].sum()
)

low_kyc_count = int(
    (flagged_transactions["kyc_tier"] == "low").sum()
)

new_device_count = int(
    flagged_transactions["new_device"].sum()
)

st.header("Fraud Pattern Analysis")

fraud_pattern_data = {
    "Fraud Pattern": [
        "Location Mismatch",
        "Low KYC Tier",
        "New Device"
    ],
    "Flagged Transactions": [
        location_mismatch_count,
        low_kyc_count,
        new_device_count
    ]
}

st.bar_chart(
    fraud_pattern_data,
    x="Fraud Pattern",
    y="Flagged Transactions"
)


# ============================================================
# Flagged Transactions View
# ============================================================
#
# This section displays transactions that were flagged as
# potentially fraudulent by the selected Random Forest model.
#
# The table allows fraud analysts to review the transaction
# characteristics associated with model-generated fraud alerts.
#
# These are model predictions and should be treated as alerts
# for further review rather than confirmed cases of fraud.

st.header("Flagged Transactions")

st.dataframe(
    flagged_transactions,
    use_container_width=True
)


# ============================================================
# Model Evaluation Metrics
# ============================================================
#
# This section calculates and displays the evaluation results
# of the selected Random Forest fraud detection model using the
# actual test labels and model predictions.
#
# Precision, fraud recall, and F1 score help analysts assess
# the model's performance on potentially fraudulent transactions.

precision = precision_score(
    y_test,
    fraud_predictions,
    zero_division=0
)

fraud_recall = recall_score(
    y_test,
    fraud_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    fraud_predictions,
    zero_division=0
)

st.header("Model Evaluation")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Precision", f"{precision:.4f}")

with col2:
    st.metric("Fraud Recall", f"{fraud_recall:.4f}")

with col3:
    st.metric("F1 Score", f"{f1:.4f}")


# ============================================================
# Confusion Matrix
# ============================================================
#
# This section displays the confusion matrix results from
# the selected Random Forest model.
#
# The values show how many transactions were correctly or
# incorrectly classified as legitimate or fraudulent.

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    fraud_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

confusion_matrix_data = {
    "Actual / Predicted": [
        "Legitimate → Legitimate",
        "Legitimate → Fraud",
        "Fraud → Legitimate",
        "Fraud → Fraud"
    ],
    "Transactions": [
        int(tn),
        int(fp),
        int(fn),
        int(tp)
    ]
}

st.dataframe(
    confusion_matrix_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Feature Importance Analysis
# ============================================================
#
# This section displays the most important features identified
# during the Random Forest model analysis.
#
# The feature importance values below are the values obtained
# during the completed model explainability analysis.
#
# The features are sorted from lowest to highest importance
# for the horizontal chart so that the most important feature
# appears at the top.
#
# These results describe model behavior and should not be
# interpreted as evidence that a feature directly causes fraud.

feature_importance_data = pd.DataFrame({
    "Feature": [
        "Customer Transaction Count Before",
        "Account Age Days",
        "Device Trust Score",
        "Amount Source",
        "Amount USD",
        "Historical Average Amount USD",
        "Transaction-to-Historical-Average Ratio",
        "Amount Deviation from Customer Average",
        "Fee",
        "Location Mismatch",
        "Time Since Previous Transaction",
        "Low KYC Tier",
        "Exchange Rate Source to Destination",
        "Transactions Previous 24 Hours",
        "New Device"
    ],
    "Importance": [
        0.164591,
        0.142453,
        0.097446,
        0.060624,
        0.058224,
        0.057870,
        0.054966,
        0.051231,
        0.051136,
        0.046575,
        0.045269,
        0.031660,
        0.019859,
        0.016434,
        0.012953
    ]
})

feature_importance_chart = feature_importance_data.sort_values(
    "Importance",
    ascending=True
)

st.header("Feature Importance")

fig_feature_importance = px.bar(
    feature_importance_chart,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 Random Forest Feature Importance",
    height=650
)

fig_feature_importance.update_layout(
    xaxis_title="Importance",
    yaxis_title="Feature",
    yaxis={"categoryorder": "array", "categoryarray": feature_importance_chart["Feature"].tolist()}
)

st.plotly_chart(
    fig_feature_importance,
    use_container_width=True
)


# ============================================================
# SHAP Model Explainability
# ============================================================
#
# This section calculates SHAP values for a sample of the
# processed test transactions using the selected Random Forest
# model.
#
# SHAP (SHapley Additive exPlanations) helps explain how the
# model's input features contribute to its predictions.
#
# The mean absolute SHAP value is used to identify the features
# with the greatest average influence on the model's fraud
# predictions.
#
# SHAP results describe model behavior and feature contribution.
# They should not be interpreted as proof that a feature
# directly causes fraud.

st.header("SHAP Model Explainability")

shap_sample_size = min(500, len(X_test_processed))

X_shap = X_test_processed[:shap_sample_size]

explainer = shap.TreeExplainer(random_forest_model)

shap_values = explainer.shap_values(X_shap)

if isinstance(shap_values, list):
    shap_fraud_values = shap_values[1]
else:
    if len(shap_values.shape) == 3:
        shap_fraud_values = shap_values[:, :, 1]
    else:
        shap_fraud_values = shap_values

feature_names = preprocessing_pipeline.get_feature_names_out()

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Importance": abs(shap_fraud_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    "SHAP Importance",
    ascending=False
).head(15)


# ============================================================
# Clean SHAP Feature Names
# ============================================================
#
# The preprocessing pipeline adds technical prefixes such as
# num__, cat__, and bin__ to feature names.
#
# This section removes those technical prefixes and converts
# underscores into spaces so that the SHAP chart is easier for
# fraud analysts to read.

shap_importance["Feature"] = (
    shap_importance["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("bin__", "", regex=False)
    .str.replace("_", " ", regex=False)
    .str.title()
)

shap_importance_chart = shap_importance.sort_values(
    "SHAP Importance",
    ascending=True
)

fig_shap = px.bar(
    shap_importance_chart,
    x="SHAP Importance",
    y="Feature",
    orientation="h",
    title="Top 15 SHAP Features",
    height=700
)

fig_shap.update_layout(
    xaxis_title="Mean Absolute SHAP Importance",
    yaxis_title="Feature"
)

st.plotly_chart(
    fig_shap,
    use_container_width=True
)

# ============================================================
# SHAP Feature Importance Table
# ============================================================
#
# This section displays all 15 SHAP features and their
# corresponding mean absolute SHAP importance values.
#
# The table is given additional height so that all 15
# features can be viewed without being hidden by the
# default dataframe display size.

st.dataframe(
    shap_importance.sort_values(
        "SHAP Importance",
        ascending=False
    ),
    use_container_width=True,
    height=600,
    hide_index=True
)

st.caption(
    "SHAP importance shows the average magnitude of each feature's "
    "contribution to the model's fraud predictions."
)