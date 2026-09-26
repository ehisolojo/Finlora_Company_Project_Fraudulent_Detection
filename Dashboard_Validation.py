# ============================================================
# FinLora Dashboard Validation
# ============================================================
#
# Purpose:
# This script independently validates the calculations used
# by the FinLora Fraud Intelligence Dashboard.
#
# The validation uses the same saved Random Forest model,
# preprocessing pipeline, X_test.csv, and y_test.csv used
# by the Streamlit dashboard.
#
# The purpose is to confirm that:
# - the test transaction count is correct
# - fraud and legitimate prediction counts are correct
# - model evaluation metrics are correct
# - the confusion matrix is correct
# - fraud pattern counts are correct
#
# This script does not modify the model, pipeline, or data.
# ============================================================

# ============================================================
# Load Saved Model, Preprocessing Pipeline, and Test Data
# ============================================================
#
# This section loads the same files used by the Streamlit
# dashboard so that the validation uses the same underlying
# model and test dataset.

import joblib
import pandas as pd

random_forest_model = joblib.load(
    "random_forest_fraud_model.pkl"
)

preprocessing_pipeline = joblib.load(
    "Finlora_Preprocessing_Pipeline.pkl"
)

X_test = pd.read_csv("X_test.csv")

y_test = pd.read_csv("y_test.csv").squeeze()

print("Validation files loaded successfully.")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# ============================================================
# Generate Fraud Predictions
# ============================================================
#
# This section applies the saved preprocessing pipeline to the
# test transactions and then uses the saved Random Forest model
# to generate fraud predictions and fraud probabilities.
#
# These are the same calculations performed by the Streamlit
# dashboard.

X_test_processed = preprocessing_pipeline.transform(X_test)

fraud_predictions = random_forest_model.predict(
    X_test_processed
)

fraud_probabilities = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]

print("Predictions generated successfully.")
print("Number of test transactions:", len(fraud_predictions))
print(
    "Potentially fraudulent transactions flagged:",
    int(fraud_predictions.sum())
)
print(
    "Legitimate transactions:",
    int((fraud_predictions == 0).sum())
)

# ============================================================
# Validate Model Evaluation Metrics
# ============================================================
#
# This section calculates the same evaluation metrics displayed
# in the Streamlit dashboard and confirms the model results
# independently.

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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

cm = confusion_matrix(
    y_test,
    fraud_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nModel Evaluation Results")
print("------------------------")
print("Precision:", round(precision, 4))
print("Fraud Recall:", round(fraud_recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix")
print("----------------")
print("True Negatives:", int(tn))
print("False Positives:", int(fp))
print("False Negatives:", int(fn))
print("True Positives:", int(tp))

# ============================================================
# Validate Fraud Pattern Counts
# ============================================================
#
# This section verifies the fraud pattern counts displayed
# in the Streamlit dashboard.
#
# The analysis is performed only on transactions flagged
# as potentially fraudulent by the Random Forest model.

flagged_transactions = X_test.copy()

flagged_transactions["predicted_fraud"] = fraud_predictions

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

print("\nFraud Pattern Validation")
print("------------------------")
print(
    "Location Mismatch:",
    location_mismatch_count
)
print(
    "Low KYC Tier:",
    low_kyc_count
)
print(
    "New Device:",
    new_device_count
)