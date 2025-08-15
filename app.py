import joblib, pandas as pd
import streamlit as st

# Load model & data
model = joblib.load("churn_model_pipeline.pkl")
df = pd.read_csv("saas_churn_clean.csv")

# Predict churn probabilities
X = df.drop(columns=["churn"])
df["churn_prob"] = model.predict_proba(X)[:, 1]

# --- Inputs ---
threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.8)
save_rate = st.slider("Save rate (%)", 0, 100, 30) / 100
retention_cost = st.number_input("Retention cost per customer", value=20)
monthly_revenue = st.number_input("Monthly revenue per customer", value=50)
months_remaining = st.number_input("Months remaining", value=12)

# --- Calculations ---
df["CLV"] = monthly_revenue * months_remaining
df["targeted"] = df["churn_prob"] >= threshold
df["expected_saved_revenue"] = df["CLV"] * save_rate
df["retention_cost"] = retention_cost
df.loc[~df["targeted"], ["expected_saved_revenue", "retention_cost"]] = 0

total_targeted = df["targeted"].sum()
total_saved = df["expected_saved_revenue"].sum()
total_cost = df["retention_cost"].sum()
net_roi = total_saved - total_cost
roi_multiple = total_saved / total_cost if total_cost > 0 else float("nan")

# --- Display ---
st.metric("Customers targeted", f"{total_targeted:,}")
st.metric("Expected saved revenue", f"${total_saved:,.0f}")
st.metric("Retention cost", f"${total_cost:,.0f}")
st.metric("Net ROI", f"${net_roi:,.0f}")
st.metric("ROI multiple", f"{roi_multiple:.2f}x")

st.subheader("Top targeted customers")
st.dataframe(
    df.loc[df["targeted"], ["security_no", "plan_tier", "churn_prob", "CLV"]]
      .sort_values("churn_prob", ascending=False)
      .head(20)
)
