# 🏡 House Price Prediction – Random Forest Regressor (Optimized)

This project is a **complete End-to-End Machine Learning Regression Pipeline** that predicts house prices using the **Housing.csv** dataset.  
The workflow includes data cleaning, feature engineering, outlier handling, hyperparameter tuning using **RandomizedSearchCV**, followed by model evaluation & visualization.

This version achieves a strong performance with:

### ✅ **R² Score: 0.8021**  
### ✅ **MAE: ~5.94 Lakhs**  
### ✅ **RMSE: ~8.59 Lakhs**  

---

# 📌 Project Highlights

### ✔ Complete EDA
- Dataset overview  
- Missing value treatment  
- Duplicates handling  
- Correlation analysis  
- Pairplot & Heatmap visualizations  

### ✔ Data Preprocessing
- One-hot encoding  
- Outlier removal (top 1%)  
- Feature transformations  
- New engineered features  

### ✔ Feature Engineering
| Feature | Description |
|--------|-------------|
| `price_per_sqft` | price ÷ area |
| `luxury_score`   | bathrooms + stories + parking |

These engineered features significantly boosted model performance.

---

# 🚀 Model & Tuning

### **Random Forest Regressor** with **RandomizedSearchCV**

Best hyperparameters found:

```python
{
 'max_depth': 19,
 'max_features': 'log2',
 'min_samples_leaf': 1,
 'min_samples_split': 5,
 'n_estimators': 249
}
```

This configuration gave the best R² and lowest error metrics.

---

# 📊 Model Evaluation

| Metric | Score |
|--------|--------|
| **MAE** | 594,662 |
| **MSE** | 7.39e+11 |
| **RMSE** | 859,743 |
| **R² Score** | **0.8021** |

---

# 🔥 Feature Importance

Most influential features in predicting house prices:

| Feature | Importance |
|---------|------------|
| area | 0.27 |
| price_per_sqft | 0.18 |
| luxury_score | 0.13 |
| bathrooms | 0.07 |
| airconditioning_yes | 0.06 |
| bedrooms | 0.05 |
| parking | 0.04 |
| stories | 0.04 |
| prefarea_yes | 0.03 |
| furnishingstatus_unfurnished | 0.03 |
| guestroom_yes | 0.03 |

---

# 📈 Visualizations Included

- Pairplot  
- Correlation heatmap  
- Feature importance bar chart  
- Actual vs Predicted scatter plot  
- Residual distribution plot  

These help understand the model behavior & error patterns.

---

# 📦 Installation

Install required libraries:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn scipy
```

---

# ▶️ How to Run

1. Clone the repository  
```bash
git clone https://github.com/Abdulqadir05/House-Price-Prediction-RandomForest.git
```

2. Navigate to project  
```bash
cd House-Price-Prediction-RandomForest/house_price
```

3. Run the model  
```bash
python house_price_model.py
```

---

# 🛠 Future Improvements

- Add XGBoost / LightGBM  
- Deploy using Streamlit / FastAPI  
- Add model saving (.pkl)  
- Add cross-validation graphs  
- Residual error heatmaps  

---

# 👨‍💻 Author

**Abdul Qadir**  
AI/ML Practitioner • Data Science Student • IIT Jodhpur  
Building strong ML fundamentals & real-world projects.

---

# ⭐ Contributions

Feel free to fork, star ⭐, or create pull requests!



