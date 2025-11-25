import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
# Visualizations
sns.pairplot(df)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm" )
plt.title("Correlation Heatmap")    
plt.show()

# Encoding Categorical Variables
df = pd.get_dummies(df, drop_first=True)
print(df.head())

# Features / Target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42
    ))
])

# Model Training
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Feature Importance
rf_model = model.named_steps['rf']
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
