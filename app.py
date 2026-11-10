
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Machine Learning Fraud Detection",
    page_icon="",
    layout="wide"
)

st.title(" Machine Learning-Based Financial Fraud Detection")
st.write(
    "A Machine Learning System for Financial Transaction "
    "Fraud Detection and Risk Analysis"
)

st.divider()

# ------------------------------------------------
# 2. LOAD SAVED MODEL
# ------------------------------------------------

MODEL_PATH = "saved_models/fraud_detection_model.pkl"
THRESHOLD_PATH = "saved_models/fraud_threshold.pkl"


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, threshold


if not os.path.exists(MODEL_PATH):
    st.error("Fraud detection model file not found!")
    st.stop()

if not os.path.exists(THRESHOLD_PATH):
    st.error("Threshold file not found!")
    st.stop()

model, threshold = load_model()

# ------------------------------------------------
# 3. MODEL INFORMATION
# ------------------------------------------------

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric("ML Algorithm", "Logistic Regression")
col2.metric("Classification Threshold", f"{threshold:.2f}")
col3.metric("Model Status", "Loaded Successfully")

st.divider()

# ------------------------------------------------
# 4. USER INPUT
# ------------------------------------------------

st.subheader("Enter Financial Transaction Details")

col1, col2 = st.columns(2)

with col1:

    numItems = st.number_input(
        "Number of Items",
        min_value=1,
        value=1,
        step=1
    )

    localTime = st.number_input(
        "Local Time",
        value=0.0
    )

    paymentMethodAgeDays = st.number_input(
        "Payment Method Age (Days)",
        min_value=0.0,
        value=1.0
    )

with col2:

    paymentMethod = st.text_input(
        "Payment Method",
        placeholder="Example: creditcard"
    )

    Category = st.text_input(
        "Transaction Category",
        placeholder="Example: electronics"
    )

st.divider()

# ------------------------------------------------
# 5. FEATURE ENGINEERING
# ------------------------------------------------

if st.button("Analyze Transaction", type="primary"):

    if not paymentMethod.strip() or not Category.strip():

        st.warning(
            "Please enter Payment Method and Category."
        )

        st.stop()

    if not np.isfinite(localTime) or not np.isfinite(paymentMethodAgeDays):
        st.error("Please enter valid numerical values.")
        st.stop()

    # Normalize categorical values

    payment_method_clean = paymentMethod.strip().lower()
    category_clean = Category.strip().lower()

    # Log transformation

    payment_age_log = np.log1p(
        paymentMethodAgeDays
    )

    # Zero flag

    payment_age_zero_flag = int(
        paymentMethodAgeDays == 0
    )

    # Item group generation

    if numItems <= 1:

        item_group = "Single"

    elif numItems <= 5:

        item_group = "Few"

    else:

        item_group = "Multiple"

    # Create transaction dataframe

    input_data = pd.DataFrame([{

        "numItems": numItems,

        "localTime": localTime,

        "paymentMethodAgeDays": paymentMethodAgeDays,

        "payment_age_log": payment_age_log,

        "payment_age_zero_flag": payment_age_zero_flag,

        "paymentMethod": payment_method_clean,

        "Category": category_clean,

        "item_group": item_group

    }])

    # ------------------------------------------------
    # 6. FRAUD PREDICTION
    # ------------------------------------------------

    fraud_probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        fraud_probability >= threshold
    )

    # ------------------------------------------------
    # 7. DISPLAY TRANSACTION DETAILS
    # ------------------------------------------------

    st.subheader("Transaction Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    # ------------------------------------------------
    # 8. DISPLAY PREDICTION RESULTS
    # ------------------------------------------------

    st.subheader("Fraud Detection Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Classification Threshold",
            f"{threshold * 100:.0f}%"
        )

    st.progress(float(fraud_probability))

    # ------------------------------------------------
    # 9. RISK CLASSIFICATION
    # ------------------------------------------------

    if prediction == 1:

        st.error(
            " FRAUDULENT TRANSACTION DETECTED"
        )

        st.write(
            "The transaction has been classified as Fraud "
            "by the Machine Learning model."
        )

        st.warning(
            "Further manual verification is recommended."
        )

    else:

        st.success(
            " LEGITIMATE CLASSIFICATION"
        )

        st.write(
            "The transaction is below the configured "
            "fraud classification threshold."
        )

    st.divider()

    st.caption(
        "This application uses a trained Logistic Regression "
        "Machine Learning model. Predictions are experimental "
        "and should not be treated as confirmed fraud decisions."
    )

# ------------------------------------------------
# 10. FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Financial Fraud Detection Using Machine Learning | "
    "Logistic Regression | Python | Scikit-learn | Streamlit"
)
