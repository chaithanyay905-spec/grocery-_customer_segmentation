import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Grocery Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Retail AI Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #064E3B 0%, #0F172A 60%, #1E293B 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        margin-bottom: 10px;
    }

    .segment-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .segment-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #34D399;
        margin: 6px 0;
    }

    .strategy-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_models():
    scaler_path = get_asset_path("customer_segmentation_scaler.pkl")
    model_path = get_asset_path("grocery_customer_segmentation_model.pkl")
    return joblib.load(scaler_path), joblib.load(model_path)

try:
    scaler, model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Segment Persona Metadata
SEGMENT_INFO = {
    0: {
        "name": "Young Affluent Deal Hunters",
        "icon": "⚡",
        "color": "#38BDF8",
        "desc": "Younger demographics with high income who actively leverage seasonal discounts and promos.",
        "marketing": "Send high-value flash sales, personalized app notifications, and bundle promo coupons."
    },
    1: {
        "name": "High-Value Loyal Champions",
        "icon": "🌟",
        "color": "#10B981",
        "desc": "Highest purchase frequency and largest basket sizes with moderate price sensitivity.",
        "marketing": "Enroll in VIP tier rewards, free priority delivery, and early access to gourmet/exclusive items."
    },
    2: {
        "name": "Mature Bargain Seekers",
        "icon": "🏷️",
        "color": "#F59E0B",
        "desc": "Older demographic with significant discount utilization (77%+) and sizable basket volume.",
        "marketing": "Weekly digital circulars, loyalty cashback rewards, and bulk purchase discount incentives."
    },
    3: {
        "name": "Mature Premium Regulars",
        "icon": "💎",
        "color": "#A78BFA",
        "desc": "Affluent mature shoppers who buy premium products consistently with very low coupon usage.",
        "marketing": "Organic & artisanal product showcases, personalized sommelier/nutrition recommendations, and concierge service."
    }
}

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Retail Analytics & K-Means Clustering AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🛒 Grocery Customer Segmentation</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Cluster grocery shoppers into behavioral personas using purchasing frequency, basket size, recency, discount affinity, and income.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Customer Persona Profiler", "📁 Batch Customer Analytics (CSV)", "📊 Cluster Insights & PCA Space"])

# --- TAB 1: Customer Persona Profiler ---
with tabs[0]:
    st.subheader("Customer Shopping Behavior & Demographics")

    # Quick Preset Buttons
    p_cols = st.columns([1, 1, 1, 1, 2])
    with p_cols[0]:
        p0 = st.button("⚡ Segment 0 Profile", width="stretch")
    with p_cols[1]:
        p1 = st.button("🌟 Segment 1 Profile", width="stretch")
    with p_cols[2]:
        p2 = st.button("🏷️ Segment 2 Profile", width="stretch")
    with p_cols[3]:
        p3 = st.button("💎 Segment 3 Profile", width="stretch")

    if p0:
        st.session_state["age"] = 35
        st.session_state["income"] = 1550000
        st.session_state["freq"] = 42
        st.session_state["basket"] = 1400.0
        st.session_state["discount"] = 62
        st.session_state["online"] = 51
        st.session_state["recency"] = 44
    elif p1:
        st.session_state["age"] = 34
        st.session_state["income"] = 1130000
        st.session_state["freq"] = 65
        st.session_state["basket"] = 3300.0
        st.session_state["discount"] = 35
        st.session_state["online"] = 47
        st.session_state["recency"] = 73
    elif p2:
        st.session_state["age"] = 57
        st.session_state["income"] = 1190000
        st.session_state["freq"] = 48
        st.session_state["basket"] = 3100.0
        st.session_state["discount"] = 77
        st.session_state["online"] = 47
        st.session_state["recency"] = 70
    elif p3:
        st.session_state["age"] = 62
        st.session_state["income"] = 1450000
        st.session_state["freq"] = 50
        st.session_state["basket"] = 2300.0
        st.session_state["discount"] = 25
        st.session_state["online"] = 52
        st.session_state["recency"] = 49

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 👤 Demographics & Financial Capacity")
        age = st.slider(
            "Customer Age", 18, 80,
            value=int(st.session_state.get("age", 34)),
            key="input_age"
        )
        annual_income = st.slider(
            "Annual Household Income ($/₹)", 300000, 2500000,
            value=int(st.session_state.get("income", 850000)), step=25000,
            key="input_income"
        )
        online_share = st.slider(
            "Online Purchase Share (%)", 0, 100,
            value=int(st.session_state.get("online", 70)), step=1,
            key="input_online", help="Percentage of grocery shopping done via digital app/web."
        )

    with c_right:
        st.markdown("#### 🛍️ Shopping Habits & Channel Dynamics")
        purchase_freq = st.slider(
            "Annual Purchase Frequency (Orders / Year)", 5, 100,
            value=int(st.session_state.get("freq", 48)), step=1,
            key="input_freq"
        )
        basket_value = st.slider(
            "Average Basket Value ($/₹)", 300.0, 5000.0,
            value=float(st.session_state.get("basket", 1800.0)), step=50.0,
            key="input_basket"
        )
        discount_rate = st.slider(
            "Discount / Promo Usage Rate (%)", 0, 100,
            value=int(st.session_state.get("discount", 65)), step=1,
            key="input_discount"
        )
        recency = st.slider(
            "Recency (Days Since Last Order)", 1, 120,
            value=int(st.session_state.get("recency", 8)), step=1,
            key="input_recency"
        )

    st.markdown("---")
    eval_btn = st.button("🚀 Classify Customer Segment", type="primary", width="stretch")

    features_list = [
        "age", "annual_income", "purchase_frequency",
        "average_basket_value", "discount_usage_rate",
        "online_purchase_share", "recency_days"
    ]

    sample_input = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "purchase_frequency": purchase_freq,
        "average_basket_value": basket_value,
        "discount_usage_rate": discount_rate / 100.0,
        "online_purchase_share": online_share / 100.0,
        "recency_days": recency
    }])

    scaled_input = scaler.transform(sample_input[features_list])
    segment_id = int(model.predict(scaled_input)[0])
    info = SEGMENT_INFO[segment_id]

    st.markdown("### 📋 Segment Persona Classification")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="segment-card" style="border-color: {info['color']};">
            <span style="font-size: 2.8rem;">{info['icon']}</span>
            <div style="color: {info['color']}; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">
                Segment {segment_id}
            </div>
            <div class="segment-title" style="color: {info['color']}; font-size: 1.8rem;">
                {info['name']}
            </div>
            <p style="color: #94A3B8; font-size: 0.92rem; margin-top: 8px;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("#### 🎯 Strategic Marketing Playbook")
        st.markdown(f"""
        <div class="strategy-box" style="border-left-color: {info['color']};">
            <strong style="color: {info['color']}; font-size: 1.05rem;">Recommended Retention & Growth Action:</strong><br>
            <p style="color: #E2E8F0; font-size: 0.95rem; margin-top: 6px;">
                {info['marketing']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📌 Key Profile Indicators")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write(f"• **Discount Affinity:** {discount_rate}%")
            st.write(f"• **Digital Share:** {online_share}%")
        with col_m2:
            st.write(f"• **Shopping Velocity:** {purchase_freq} orders/year")
            st.write(f"• **Basket Size:** ${basket_value:.2f}")

    with st.expander("🔍 View Raw Features Vector"):
        st.dataframe(sample_input, width="stretch")

# --- TAB 2: Batch Customer Analytics ---
with tabs[1]:
    st.subheader("Batch Customer Base Clustering")
    st.write("Upload a retail dataset or analyze the 1,200 customer benchmark records.")

    csv_upload = st.file_uploader("Upload Customer Data CSV", type=["csv"], key="cust_csv")
    df_customers = None

    if csv_upload is not None:
        df_customers = pd.read_csv(csv_upload)
        st.info(f"Loaded {len(df_customers)} records from file.")
    else:
        sample_path = get_asset_path("segmented_customers.csv")
        if not os.path.exists(sample_path):
            sample_path = get_asset_path("data/grocery_customers.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load pre-segmented customer dataset (`segmented_customers.csv`)", value=True):
                df_customers = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_customers)} records from benchmark dataset.")

    if df_customers is not None:
        missing = [c for c in features_list if c not in df_customers.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Cluster & Segment Customers", type="primary"):
                with st.spinner("Assigning clusters..."):
                    scaled_batch = scaler.transform(df_customers[features_list])
                    clusters = model.predict(scaled_batch)

                    res_df = df_customers.copy()
                    res_df["Segment_ID"] = clusters
                    res_df["Segment_Name"] = [SEGMENT_INFO[c]["name"] for c in clusters]

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Shoppers", len(res_df))
                    m2.metric("Segment 0 (Affluent Deals)", sum(clusters == 0))
                    m3.metric("Segment 1 (Champions)", sum(clusters == 1))
                    m4.metric("Segment 3 (Premium)", sum(clusters == 3))

                    seg_filter = st.selectbox("Filter by Persona:", ["All"] + [f"Segment {k}: {v['name']}" for k, v in SEGMENT_INFO.items()])
                    if seg_filter != "All":
                        filter_id = int(seg_filter.split(":")[0].replace("Segment ", ""))
                        view = res_df[res_df["Segment_ID"] == filter_id]
                    else:
                        view = res_df

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Segmented Customers CSV",
                        data=csv_export,
                        file_name="segmented_grocery_customers.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Cluster Insights & PCA Space ---
with tabs[2]:
    st.subheader("K-Means Architecture & PCA Visualization")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Segmentation Methodology
        - **Algorithm**: `KMeans(n_clusters=4, random_state=42, n_init=10)`
        - **Preprocessing**: `StandardScaler` applied across all 7 behavioral features.
        - **Dimensionality Reduction**: Principal Component Analysis (PCA) used to project 7D feature space into 2D plane for visual inspection.
        - **Validation Score**: Silhouette Score ~0.11
        """)

    with c2:
        img_path = get_asset_path("customer_segments.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="2D PCA Customer Cluster Space", width="stretch")
        else:
            st.info("Chart image not found.")

st.caption("Grocery Customer Segmentation Suite • Scikit-learn & Streamlit")
