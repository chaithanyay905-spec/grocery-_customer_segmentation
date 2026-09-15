# Grocery Customer Segmentation

A machine learning project using Python and Scikit-learn to group grocery customers into meaningful customer segments.

## Algorithm
- K-Means Clustering
- StandardScaler
- PCA visualization
- Silhouette score evaluation

## Features
- Age
- Annual income
- Purchase frequency
- Average basket value
- Discount usage rate
- Online purchase share
- Recency in days

## Output
The model assigns each customer to one of 4 segments:
- Segment 0
- Segment 1
- Segment 2
- Segment 3

The meaning of each segment can be interpreted by comparing average feature values.

## Installation

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

## Predict a customer segment

```bash
python predict.py
```

## Run Web Frontend

```bash
streamlit run app.py
```

The included dataset is synthetic and intended for academic and educational use only.
