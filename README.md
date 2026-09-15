# grocery-_customer_segmentation
Grocery customer segmentation using K-Means clustering, Scikit-learn, and Streamlit to identify customer personas and support targeted marketing strategies.
#  Grocery Customer Segmentation

An end-to-end **Machine Learning customer segmentation project** that uses **K-Means clustering** to group grocery customers into meaningful behavioral personas based on demographics, purchasing habits, spending behavior, discount usage, online shopping activity, and recency.

The project includes a **Streamlit web application** for interactive customer classification, batch segmentation, and cluster visualization.

##  Features

* Customer segmentation using **K-Means Clustering**
* Feature normalization using **StandardScaler**
* 7 customer demographic and behavioral features
* **PCA** visualization of customer clusters
* Silhouette Score for cluster evaluation
* Individual customer segment prediction
* Batch customer segmentation using CSV files
* Interactive **Streamlit dashboard**
* Pre-trained model and scaler included
* Downloadable segmented customer dataset
* Marketing recommendations for each customer persona

##  Machine Learning Approach

The project uses **K-Means clustering with 4 clusters**.

### Input Features

The model uses the following features:

* `age` — Customer age
* `annual_income` — Annual household income
* `purchase_frequency` — Number of purchases per year
* `average_basket_value` — Average value of each purchase
* `discount_usage_rate` — Discount/promotion usage rate
* `online_purchase_share` — Percentage of purchases made online
* `recency_days` — Days since the customer's last order

### Preprocessing

`StandardScaler` is applied to normalize the seven input features before clustering.

### Clustering

```text
Algorithm: K-Means
Number of Clusters: 4
Random State: 42
n_init: 10
```

PCA is then used to reduce the seven-dimensional feature space to two dimensions for visualization.

##  Customer Segments

The Streamlit application interprets the four clusters as customer personas:

| Segment | Persona                       | Description                                                                         |
| ------- | ----------------------------- | ----------------------------------------------------------------------------------- |
| 0       |  Young Affluent Deal Hunters | Younger, higher-income customers with stronger discount usage                       |
| 1       |  High-Value Loyal Champions | Frequent shoppers with larger basket sizes and relatively lower price sensitivity   |
| 2       |  Mature Bargain Seekers    | Older customers with high discount usage and substantial basket sizes               |
| 3       |  Mature Premium Regulars    | Older, affluent customers with consistent premium purchasing and lower coupon usage |

> **Note:** K-Means cluster IDs are model-generated labels. The persona names are interpretations based on the average characteristics of each cluster.

##  Dataset

The included synthetic dataset contains **1,200 grocery customer records** and 7 input features.

The dataset is provided for **educational and academic purposes**.

##  Project Structure

```text
Grocery_Customer_Segmentation_Sklearn/
│
├── app.py
├── train_model.py
├── predict.py
│
├── data/
│   └── grocery_customers.csv
│
├── customer_segmentation_scaler.pkl
├── grocery_customer_segmentation_model.pkl
├── segmented_customers.csv
├── customer_segments.png
├── requirements.txt
└── README.md
```

##  Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Grocery_Customer_Segmentation_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  Train the Model

To train the K-Means clustering model:

```bash
python train_model.py
```

This will:

1. Load the grocery customer dataset
2. Select the seven segmentation features
3. Standardize the features
4. Train the K-Means model
5. Calculate the Silhouette Score
6. Assign customers to clusters
7. Save the trained model and scaler
8. Generate the PCA visualization
9. Export the segmented customer dataset

##  Predict a Customer Segment

Run:

```bash
python predict.py
```

The script uses the saved model and scaler to classify a sample customer into one of the four customer segments.

##  Run the Streamlit Application

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The application provides three main sections:

###  Customer Persona Profiler

Enter customer information and classify the customer into a segment.

###  Batch Customer Analytics

Upload a CSV containing customer information and assign segments to multiple customers at once.

The resulting segmented dataset can be downloaded directly from the application.

###  Cluster Insights & PCA

View the K-Means methodology and a 2D PCA visualization of the customer clusters.

##  Model Evaluation

The project uses the **Silhouette Score** to evaluate how well-separated the customer clusters are.

The current implementation reports a Silhouette Score of approximately **0.11**.

Because the score is relatively low, the clusters should be viewed as exploratory customer groupings rather than highly distinct natural categories.

##  Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Matplotlib**
* **Streamlit**

##  Business Applications

Customer segmentation can help grocery businesses:

* Create targeted marketing campaigns
* Personalize discounts and promotions
* Identify high-value customers
* Improve customer retention
* Develop loyalty programs
* Optimize digital marketing strategies
* Identify premium and price-sensitive customer groups

##  Future Improvements

Potential improvements include:

* Optimizing the number of clusters using the Elbow Method and Silhouette Analysis
* Testing alternative clustering algorithms such as DBSCAN or Gaussian Mixture Models
* Adding more customer behavioral features
* Performing automated cluster profiling
* Improving cluster separation
* Adding interactive analytics and charts
* Deploying the Streamlit application online
* Adding model monitoring and retraining workflows

##  License

This project is intended for educational and academic purposes.
