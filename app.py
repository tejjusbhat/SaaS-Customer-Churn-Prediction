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

import altair as alt

st.subheader("Churn probability distribution")
hist = alt.Chart(pd.DataFrame({"churn_prob": df["churn_prob"]})).mark_bar().encode(
    alt.X("churn_prob:Q", bin=alt.Bin(maxbins=40), title="Churn probability"),
    alt.Y("count()", title="Customers")
)
rule = alt.Chart(pd.DataFrame({"thr": [threshold]})).mark_rule(color="red").encode(
    x="thr:Q"
)
st.altair_chart(hist + rule, use_container_width=True)

import numpy as np

st.subheader("ROI vs. threshold")
ths = np.linspace(0.0, 0.99, 50)
monthly_rev = df["avg_transaction_value"] if "avg_transaction_value" in df.columns else monthly_revenue
clv_series = monthly_rev * months_remaining

rows = []
for t in ths:
    mask = df["churn_prob"] >= t
    total_saved = (clv_series[mask] * save_rate).sum()
    total_cost  = retention_cost * mask.sum()
    net_roi     = total_saved - total_cost
    rows.append({"threshold": t, "net_roi": net_roi, "targeted": int(mask.sum())})

roi_curve = pd.DataFrame(rows)
line = alt.Chart(roi_curve).mark_line().encode(
    x=alt.X("threshold:Q", title="Threshold"),
    y=alt.Y("net_roi:Q", title="Net ROI ($)")
)
bars = alt.Chart(roi_curve).mark_bar(opacity=0.3).encode(
    x="threshold:Q",
    y=alt.Y("targeted:Q", title="Customers targeted"),
    color=alt.value("#999")
).interactive()

st.altair_chart(line, use_container_width=True)
st.caption("Tip: slide the threshold in the sidebar and compare against this curve.")

if "plan_tier" in df.columns:
    st.subheader("Expected saved revenue by plan tier")
    by_tier = (df.loc[df["churn_prob"] >= threshold]
                 .assign(expected_saved=lambda d: (d.get("avg_transaction_value", monthly_revenue)*months_remaining*save_rate))
                 .groupby("plan_tier")["expected_saved"].sum()
                 .reset_index())
    st.bar_chart(by_tier.set_index("plan_tier"))

st.subheader("Cumulative gains (lift) curve")
tmp = df[["churn", "churn_prob"]].copy()
tmp = tmp.sort_values("churn_prob", ascending=False).reset_index(drop=True)
tmp["cum_positives"] = tmp["churn"].cumsum()
total_pos = tmp["churn"].sum()
tmp["pct_customers"] = (np.arange(len(tmp)) + 1) / len(tmp)
tmp["recall"] = tmp["cum_positives"] / total_pos

gain = alt.Chart(tmp).mark_line().encode(
    x=alt.X("pct_customers:Q", title="Percent of customers targeted"),
    y=alt.Y("recall:Q", title="Percent of churners captured")
)
random = alt.Chart(pd.DataFrame({"pct_customers":[0,1], "recall":[0,1]})).mark_line(strokeDash=[4,4], color="gray").encode(
    x="pct_customers:Q", y="recall:Q"
)
st.altair_chart(gain + random, use_container_width=True)

from sklearn.metrics import precision_recall_curve

st.subheader("Precision–Recall vs threshold")
precision, recall, thr = precision_recall_curve(df["churn"], df["churn_prob"])
pr_df = pd.DataFrame({"threshold": np.r_[0, thr], "precision": precision, "recall": recall})

p_line = alt.Chart(pr_df).mark_line().encode(x="threshold:Q", y=alt.Y("precision:Q", title="Precision"))
r_line = alt.Chart(pr_df).mark_line(color="#888").encode(x="threshold:Q", y="recall:Q")
st.altair_chart(p_line + r_line, use_container_width=True)
st.caption("Precision (how many targeted were true churners) vs Recall (how many churners we captured).")

st.subheader("Top targeted customers")
cols = [c for c in ["security_no","plan_tier","api_calls_90d","logins_90d","days_since_active","churn_prob"] if c in df.columns]
topN = df.loc[df["churn_prob"] >= threshold, cols].sort_values("churn_prob", ascending=False).head(50)
st.dataframe(topN, use_container_width=True)

st.download_button(
    "Download targeted customers CSV",
    data=df.loc[df["churn_prob"] >= threshold].to_csv(index=False).encode("utf-8"),
    file_name="targeted_customers.csv",
    mime="text/csv"
)
