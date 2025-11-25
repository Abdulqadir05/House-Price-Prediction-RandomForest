# 🏡 House Price Prediction – Random Forest Regressor

A complete **End-to-End Machine Learning Regression Project** built using the **Housing.csv** dataset.  
This project predicts house prices based on features such as area, bedrooms, bathrooms, parking, and various categorical attributes.

This repository follows a **professional DS/ML workflow** including:
- Data Loading  
- EDA (Exploratory Data Analysis)  
- Data Cleaning & Preprocessing  
- Feature Encoding  
- Model Building (RandomForestRegressor)  
- Model Evaluation  
- Feature Importance  
- Visualization  

---

## 📁 Project Structure

```
house_price/
│── house_price_model.py      # Full ML code (E2E pipeline)
│── housing.csv               # Dataset (required for running the model)
│── README.md                 # Project documentation
```

---

## 🚀 Features

### ✔ Complete EDA
- Dataset overview  
- Missing value handling  
- Outlier check  
- Duplicate removal  
- Correlation matrix  
- Pairplot & heatmap  

### ✔ Data Preprocessing
- One-hot encoding  
- Feature selection  
- Train/Test split  

### ✔ Machine Learning Model
**Random Forest Regressor**
- `n_estimators = 300`  
- `max_depth = 10`  
- `min_samples_split = 15`  
- `min_samples_leaf = 5`  

### ✔ Evaluation Metrics
- MAE  
- MSE  
- RMSE  
- R² Score  

### ✔ Visualizations
- Correlation Heatmap  
- Actual vs Predicted plot  
- Residual distribution  
- Feature importance bar chart  

---

## 📊 Results

| Metric | Score |
|--------|--------|
| MAE | ~ 1,088,047 |
| MSE | ~ 2.11e+12 |
| RMSE | ~ 1,455,863 |
| R² Score | ~ 0.58 |

> The model explains **58% of the variance** in house prices.  
Further improvements can be achieved via feature engineering, outlier handling, and advanced models like XGBoost.

---

## 📦 Requirements

Install required libraries:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

## ▶️ How to Run the Project

1. Clone the repository  
```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction-RF.git
```

2. Navigate into folder  
```bash
cd House-Price-Prediction-RF/house_price
```

3. Run the model  
```bash
python house_price_model.py
```

---

## 🔥 Future Improvements

- Add XGBoost & LightGBM models  
- Hyperparameter tuning with RandomizedSearchCV  
- Feature engineering for price per sqft, luxury score, etc.  
- Deployment using FastAPI / Streamlit  
- Save model using Pickle / Joblib  

---

## 👨‍💻 Author

**Abdul Qadir**  
Aspiring AI/Data Scientist | ML Practitioner  
2nd Semester — Growing fast in ML/AI 🚀

---

## ⭐ Contribute

Feel free to fork this project, raise issues, or contribute improvements!

