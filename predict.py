import joblib
import pandas as pd

scaler = joblib.load("customer_segmentation_scaler.pkl")
model = joblib.load("grocery_customer_segmentation_model.pkl")

sample = pd.DataFrame([{
    "age": 34,
    "annual_income": 850000,
    "purchase_frequency": 48,
    "average_basket_value": 1800,
    "discount_usage_rate": 0.65,
    "online_purchase_share": 0.70,
    "recency_days": 8
}])

features = [
    "age",
    "annual_income",
    "purchase_frequency",
    "average_basket_value",
    "discount_usage_rate",
    "online_purchase_share",
    "recency_days"
]

scaled_sample = scaler.transform(sample[features])
segment = model.predict(scaled_sample)[0]

print("Predicted Customer Segment:", segment)
