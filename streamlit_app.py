import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- LOAD MODEL --------------------
model = joblib.load("house_price_model.pkl")

# For safety, see what features the model expects
EXPECTED_COLS = list(model.feature_names_in_)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🏡 Premium House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------- CUSTOM CSS (BACKGROUND + GLASS UI) --------------------
page_bg = """
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #141e30 0%, #243b55 100%);
    color: white;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.95);
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    padding: 24px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-top: 15px;
}

/* Centered title */
h1, h2, h3 {
    color: #ffffff;
}

/* Make number inputs text white */
input {
    color: #ffffff !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- SIDEBAR: YOUR PROFILE --------------------
with st.sidebar:
    # If you get a proper hosted URL for your photo, put it here:
    # st.image("https://raw.githubusercontent.com/Abdulqadir05/House-Price-Prediction-RandomForest/main/1000095826.jpg", width=140)
    st.markdown("### 👑 Abdul Qadir")
    st.markdown("""
**BS in Applied AI & Data Science**  
_IIT Jodhpur_  

- 🧠 Machine Learning & Regression  
- 🧮 End-to-End ML Pipelines  
- 🚀 FastAPI & Streamlit Deployment  

**GitHub:** `Abdulqadir05`  
""")
    st.markdown("---")
    st.caption("Made with ❤️ using Python, scikit-learn & Streamlit")

# -------------------- HEADER --------------------
st.markdown(
    "<h1 style='text-align:center;'>🏡 Premium House Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Fill the details below to get an AI-powered house price estimate.</p>",
    unsafe_allow_html=True
)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("📌 Property Details")

# -------------------- INPUT FORM --------------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("🏠 Area (sq ft)", min_value=500, max_value=100000, value=3000, step=50)
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2, step=1)
    stories = st.number_input("🏢 Stories", min_value=1, max_value=10, value=2, step=1)
    parking = st.number_input("🚗 Parking Slots", min_value=0, max_value=10, value=1, step=1)

with col2:
    price_per_sqft = st.number_input("💲 Price per Sqft", min_value=100, max_value=20000, value=1500, step=50)
    luxury_score = st.number_input("🌟 Luxury Score", min_value=0, max_value=30, value=5, step=1)

    mainroad_opt = st.selectbox("🛣 On Main Road?", ["No", "Yes"])
    guestroom_opt = st.selectbox("🛏 Guest Room Available?", ["No", "Yes"])
    basement_opt = st.selectbox("🏚 Basement Available?", ["No", "Yes"])
    hotwater_opt = st.selectbox("🔥 Hot Water Heating?", ["No", "Yes"])
    ac_opt = st.selectbox("❄ Air Conditioning?", ["No", "Yes"])
    prefarea_opt = st.selectbox("📍 Preferred Area?", ["No", "Yes"])

furnishing = st.radio(
    "🛋 Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"],
    horizontal=True
)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- ENCODING HELPERS --------------------
def yn_to_int(val: str) -> int:
    return 1 if val == "Yes" else 0

mainroad_yes = yn_to_int(mainroad_opt)
guestroom_yes = yn_to_int(guestroom_opt)
basement_yes = yn_to_int(basement_opt)
hotwaterheating_yes = yn_to_int(hotwater_opt)
airconditioning_yes = yn_to_int(ac_opt)
prefarea_yes = yn_to_int(prefarea_opt)

furnishing_semi = 1 if furnishing == "Semi-Furnished" else 0
furnishing_un = 1 if furnishing == "Unfurnished" else 0
# Furnished -> both 0, because it's baseline category

# -------------------- BUILD INPUT ROW MATCHING MODEL FEATURES --------------------
def build_input_row(expected_cols):
    """
    Create a dict mapping each expected column to the right value
    from the UI inputs above.
    This is robust even if column names have hyphen / underscore, etc.
    """
    row = {}
    for col in expected_cols:
        c = col.lower()

        if c == "area":
            row[col] = area
        elif c == "bedrooms":
            row[col] = bedrooms
        elif c == "bathrooms":
            row[col] = bathrooms
        elif c == "stories":
            row[col] = stories
        elif c == "parking":
            row[col] = parking
        elif "price_per_sqft" in c or "price per sqft" in c:
            row[col] = price_per_sqft
        elif "luxury_score" in c:
            row[col] = luxury_score

        elif "mainroad" in c:
            row[col] = mainroad_yes
        elif "guestroom" in c:
            row[col] = guestroom_yes
        elif "basement" in c:
            row[col] = basement_yes
        elif "hotwater" in c:
            row[col] = hotwaterheating_yes
        elif "airconditioning" in c or "air_conditioning" in c:
            row[col] = airconditioning_yes
        elif "prefarea" in c:
            row[col] = prefarea_yes

        elif "furnishing" in c and "semi" in c:
            row[col] = furnishing_semi
        elif "furnishing" in c and "unfurnished" in c:
            row[col] = furnishing_un

        else:
            # Any unexpected feature -> 0 (safe default)
            row[col] = 0

    return row

# -------------------- PREDICTION BUTTON --------------------
if st.button("🔮 Predict Price", use_container_width=True):
    input_row = build_input_row(EXPECTED_COLS)
    input_df = pd.DataFrame([input_row], columns=EXPECTED_COLS)

    pred = model.predict(input_df)[0]

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='text-align:center;'>💰 Predicted Price:</h2>"
        f"<h1 style='text-align:center;'>₹ {pred:,.0f}</h1>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("⚙ Model: RandomForestRegressor (tuned with RandomizedSearchCV, R² ≈ 0.80)")

