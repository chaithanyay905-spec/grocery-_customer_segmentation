import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("data/grocery_customers.csv")

features = [
    "age",
    "annual_income",
    "purchase_frequency",
    "average_basket_value",
    "discount_usage_rate",
    "online_purchase_share",
    "recency_days"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

df["customer_segment"] = clusters
score = silhouette_score(X_scaled, clusters)

print("Silhouette Score:", round(score, 4))
print("\nCustomer segment counts:")
print(df["customer_segment"].value_counts().sort_index())

joblib.dump(scaler, "customer_segmentation_scaler.pkl")
joblib.dump(model, "grocery_customer_segmentation_model.pkl")
df.to_csv("segmented_customers.csv", index=False)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.65)
plt.title("Grocery Customer Segmentation")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("customer_segments.png")
print("\nModel and segmented data saved.")
