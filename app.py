"""
app.py
======
Streamlit Deployment App - Loan Approval Prediction

Loads the trained models bundle (loan_approval_models.pkl) produced by
train_models.py and lets the user:
    1. Pick which ML model to use (SVM / KNN / Decision Tree / Random Forest)
    2. Enter applicant details through a form
    3. Get an instant "Approved" / "Rejected" prediction (+ confidence, where
       the model supports probability estimates)

Run:
    streamlit run app.py
"""

#import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

MODEL_BUNDLE_PATH = "loan_approval_models.pkl"


# --------------------------------------------------------------------------
# LOAD MODEL BUNDLE (cached so it only loads once)
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


try:
    bundle = load_bundle(MODEL_BUNDLE_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_BUNDLE_PATH}`. "
        "Run `python train_models.py` first to generate it."
    )
    st.stop()

models = bundle["models"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]
feature_columns = bundle["feature_columns"]
scale_columns = bundle["scale_columns"]
accuracies = bundle["accuracies"]

education_encoder = label_encoders["education"]
self_employed_encoder = label_encoders["self_employed"]
target_encoder = label_encoders["loan_status"]


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("🏦 Loan Approval Prediction")
st.write(
    "Fill in the applicant's details below, choose a model, and click "
    "**Predict** to see whether the loan is likely to be **Approved** or "
    "**Rejected**."
)

# --------------------------------------------------------------------------
# SIDEBAR - MODEL SELECTION
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    options=list(models.keys()),
)

st.sidebar.metric(
    label=f"{model_name} Test Accuracy",
    value=f"{accuracies[model_name] * 100:.2f}%",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 All Model Accuracies")
acc_df = pd.DataFrame(
    {"Model": list(accuracies.keys()), "Accuracy": [f"{v*100:.2f}%" for v in accuracies.values()]}
)
st.sidebar.dataframe(acc_df, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------------------------
with st.form("loan_form"):
    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=2, step=1
        )
        education = st.selectbox("Education", options=list(education_encoder.classes_))
        self_employed = st.selectbox(
            "Self Employed", options=list(self_employed_encoder.classes_)
        )
        income_annum = st.number_input(
            "Annual Income (₹)", min_value=0, value=5000000, step=100000
        )
        loan_amount = st.number_input(
            "Loan Amount (₹)", min_value=0, value=15000000, step=100000
        )
        loan_term = st.number_input(
            "Loan Term (years)", min_value=1, max_value=40, value=10, step=1
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score", min_value=300, max_value=900, value=650, step=1
        )
        residential_assets_value = st.number_input(
            "Residential Assets Value (₹)", min_value=0, value=5000000, step=100000
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets Value (₹)", min_value=0, value=3000000, step=100000
        )
        luxury_assets_value = st.number_input(
            "Luxury Assets Value (₹)", min_value=0, value=8000000, step=100000
        )
        bank_asset_value = st.number_input(
            "Bank Asset Value (₹)", min_value=0, value=3000000, step=100000
        )

    submitted = st.form_submit_button("🔍 Predict")


# --------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------
def build_input_dataframe():
    raw = {
        "no_of_dependents": no_of_dependents,
        "education": education_encoder.transform([education])[0],
        "self_employed": self_employed_encoder.transform([self_employed])[0],
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    input_df = pd.DataFrame([raw])

    # Scale the same columns that were scaled during training
    input_df[scale_columns] = scaler.transform(input_df[scale_columns])

    # Ensure column order exactly matches what the models were trained on
    input_df = input_df[feature_columns]

    return input_df


if submitted:
    input_df = build_input_dataframe()
    model = models[model_name]

    prediction = model.predict(input_df)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if predicted_label.strip().lower() == "approved":
        st.success(f"✅ Loan Status: **{predicted_label}** (using {model_name})")
    else:
        st.error(f"❌ Loan Status: **{predicted_label}** (using {model_name})")

    # Show confidence if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame(
            {
                "Class": target_encoder.classes_,
                "Probability": [f"{p * 100:.2f}%" for p in proba],
            }
        )
        st.write("**Prediction Confidence:**")
        st.dataframe(proba_df, hide_index=True, use_container_width=True)

    with st.expander("See processed model input"):
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Models trained on `loan_approval_dataset.csv` using SVM, KNN, Decision Tree, "
    "and Random Forest classifiers. Built with scikit-learn & Streamlit."
)
