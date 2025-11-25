import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="🏡 House Price Prediction", layout="centered")

# -------------- TITLE -----------------
st.title("🏡 House Price Prediction App")
st.markdown("Enter house details below to get the predicted price.")

st.divider()

# -------------- INPUT FORM ------------
st.subheader("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("🏠 Area (sq ft)", min_value=500, max_value=10000, step=50)
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, step=1)
    stories = st.number_input("🏢 Stories", min_value=1, max_value=10, step=1)
    parking = st.number_input("🚗 Parking", min_value=0, max_value=5, step=1)

with col2:
    price_per_sqft = st.number_input("💲 Price per sq ft", min_value=100, max_value=20000, step=50)
    luxury_score = st.number_input("⭐ Luxury Score", min_value=0, max_value=30, step=1)

    mainroad_yes = st.selectbox("Main Road?", ["No", "Yes"])
    guestroom_yes = st.selectbox("Guest Room?", ["No", "Yes"])
    basement_yes = st.selectbox("Basement?", ["No", "Yes"])
    hotwaterheating_yes = st.selectbox("Hot Water Heating?", ["No", "Yes"])
    airconditioning_yes = st.selectbox("Air Conditioning?", ["No", "Yes"])
    prefarea_yes = st.selectbox("Preferred Area?", ["No", "Yes"])

furnishing = st.radio(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

# Convert categorical to 0/1
def encode(x): return 1 if x == "Yes" else 0

mainroad_yes = encode(mainroad_yes)
guestroom_yes = encode(guestroom_yes)
basement_yes = encode(basement_yes)
hotwaterheating_yes = encode(hotwaterheating_yes)
airconditioning_yes = encode(airconditioning_yes)
prefarea_yes = encode(prefarea_yes)

furnishing_semi = 1 if furnishing == "Semi-Furnished" else 0
furnishing_un = 1 if furnishing == "Unfurnished" else 0

# ------------- PREDICT BUTTON --------
if st.button("🔮 Predict Price"):
    
    input_data = np.array([[
        area, bedrooms, bathrooms, stories, parking,
        price_per_sqft, luxury_score,
        mainroad_yes, guestroom_yes, basement_yes,
        hotwaterheating_yes, airconditioning_yes, prefarea_yes,
        furnishing_semi, furnishing_un
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 **Predicted Price:** ₹ {prediction:,.0f}")
