import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint

# Load dataset
df = pd.read_csv("Housing.csv") 
print(df.head())

# Data Preprocessing
print("Data Info:", df.info())
print("Data Description:\n", df.describe())
print("Data Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum() / len(df) * 100)
print("Duplicate Rows:", df.duplicated().sum() / len(df) * 100)
df.drop_duplicates(inplace=True)
correlation_matrix = df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']].corr()
print("Correlation Matrix:\n", correlation_matrix)
df['price_per_sqft'] = df['price'] / df['area']
df['luxury_score'] = df['bathrooms'] + df['stories'] + df['parking']

# Visualizations
sns.pairplot(df)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm" )
plt.title("Correlation Heatmap")    
plt.show()

# 🔥 Outlier removal (Top 1% remove)
df = df[df['price'] < df['price'].quantile(0.99)]
df = df[df['area'] < df['area'].quantile(0.99)]


# Encoding Categorical Variables
df = pd.get_dummies(df, drop_first=True)
print(df.head())

# Features / Target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomizedSearchCV Params

param_dist = {
    'n_estimators': randint(200, 600),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

# Random Search Setup

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=25,          
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit RandomSearch on training data
random_search.fit(X_train, y_train)

# Best model from search
best_rf = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Use the best model to predict
y_pred = best_rf.predict(X_test)

# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
# Feature Importance
rf_model = best_rf
importances = rf_model.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance Table:\n", fi_df)


# Bar Plot
plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
plt.title("Feature Importance (Random Forest Regressor)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Predicted vs Actual Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest Regressor: Actual vs Predicted Prices")
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

