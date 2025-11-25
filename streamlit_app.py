import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load("house_price_model.pkl")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🏡 Premium House Price Predictor",
                   page_icon="🏠",
                   layout="centered")

# Background CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
    color: white;
}

.css-10trblm, .css-1v0mbdj {
    color: white !important;
}

.glass-card {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("Abdul Qadir")
    st.markdown("""
    **AI & Data Science Student (IIT Jodhpur)**  
    - ML | DL | Python  
    - End-to-End Deployment Expert  
    - Building real ML products 🚀  

    **GitHub:** Abdulqadir05  
    """)
    st.markdown("---")
    st.write("Made with ❤️ using Streamlit")

# -------------------- MAIN TITLE --------------------
st.markdown("<h1 style='text-align:center;'>🏡 Premium House Price Prediction</h1>",
            unsafe_allow_html=True)

st.markdown("<p style='text-align:center;'>Enter details below and get instant AI-powered prediction.</p>",
            unsafe_allow_html=True)

# -------------------- INPUT FORM --------------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📌 Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("🏠 Area (sq ft)", min_value=500, max_value=10000, step=50)
        bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=10)
        bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10)
        stories = st.number_input("🏢 Stories", min_value=1, max_value=10)
        parking = st.number_input("🚗 Parking", min_value=0, max_value=5)

    with col2:
        price_per_sqft = st.number_input("💲 Price per Sqft", min_value=100, max_value=20000)
        luxury_score = st.number_input("🌟 Luxury Score", min_value=0, max_value=30)

        mainroad_yes = st.selectbox("🛣 Main Road Access", ["No", "Yes"])
        guestroom_yes = st.selectbox("🛏 Guest Room", ["No", "Yes"])
        basement_yes = st.selectbox("🏚 Basement", ["No", "Yes"])
        hotwaterheating_yes = st.selectbox("🔥 Hot Water Heating", ["No", "Yes"])
        airconditioning_yes = st.selectbox("❄ Air Conditioning", ["No", "Yes"])
        prefarea_yes = st.selectbox("📍 Preferred Area", ["No", "Yes"])

    furnishing = st.radio("🛋 Furnishing Status",
                          ["Unfurnished", "Semi-Furnished", "Furnished"])

    st.markdown("</div>", unsafe_allow_html=True)

# Convert Yes/No → 1/0
def enc(x): return 1 if x == "Yes" else 0

mainroad_yes = enc(mainroad_yes)
guestroom_yes = enc(guestroom_yes)
basement_yes = enc(basement_yes)
hotwaterheating_yes = enc(hotwaterheating_yes)
airconditioning_yes = enc(airconditioning_yes)
prefarea_yes = enc(prefarea_yes)

furnishing_semi = 1 if furnishing == "Semi-Furnished" else 0
furnishing_un = 1 if furnishing == "Unfurnished" else 0

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Price", use_container_width=True):

    cols = [
        "area", "bedrooms", "bathrooms", "stories", "parking",
        "price_per_sqft", "luxury_score", "mainroad_yes", "guestroom_yes",
        "basement_yes", "hotwaterheating_yes", "airconditioning_yes",
        "prefarea_yes", "furnishingstatus_semi_furnished",
        "furnishingstatus_unfurnished"
    ]

    input_df = pd.DataFrame([[
        area, bedrooms, bathrooms, stories, parking,
        price_per_sqft, luxury_score, mainroad_yes, guestroom_yes,
        basement_yes, hotwaterheating_yes, airconditioning_yes,
        prefarea_yes, furnishing_semi, furnishing_un
    ]], columns=cols)

    pred = model.predict(input_df)[0]

    # Result UI card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='text-align:center;'>💰 Predicted Price: <br> ₹ {pred:,.0f}</h2>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

