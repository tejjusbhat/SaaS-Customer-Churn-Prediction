# SaaS Customer Churn Prediction with ROI Simulator
Predicts churn, explains it with SHAP, and optimizes retention ROI with an interactive dashboard.

SaaS companies lose a ton of revenue due to customer churn. The goal of this project is to make a customer churn model that can predict churn risk, explain drivers and simulate ROI through an interactive dashboard.

## What is churn?
Churn refers to when subscribers decide to stop using a service. This directly impacts the monthly revenue of a SaaS, even a 2-3% annual churn reduction can lead to millions of dollars of retained revenue for a mid-sized SaaS with thousands of customers. Retained customers tend to have a higher lifetime value because they continue to pay monthly.

This project connects machine learning predictions directly to business ROI by:
- Predicting churn probability for each customer.
- Explaining why the model flags them (via SHAP) so the retention team knows what to address.
- Running an ROI simulator that models how different thresholds, save rates, and retention costs affect profitability.
- The Simulator can be tried on: https://saas-churn-prediction.streamlit.app/

## Dataset
The dataset is a modified version of "d0r1h/customer_churn" from huggingface with added features like aggregate time series data and api usage.

## Results
Model values
```
 precision    recall  f1-score   support

           0       0.94      0.91      0.93      3396
           1       0.93      0.95      0.94      4003

    accuracy                           0.93      7399
   macro avg       0.93      0.93      0.93      7399
weighted avg       0.93      0.93      0.93      7399

ROC AUC: 0.9747857687417594
```
