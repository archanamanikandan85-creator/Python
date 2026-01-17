import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATASET 
# Note: Replace 'accidents.csv' with the path to your Kaggle dataset
# For example: pd.read_csv('/kaggle/input/road-accidents-severity/RTA Dataset.csv')
try:
    df = pd.read_csv('accidents.csv')
    print("Dataset Loaded Successfully!")
except:
    print("Please upload your dataset file first.")

# 2. PREPROCESSING
# Filling missing values with the most frequent value (Mode)
df = df.fillna(df.mode().iloc[0])

# Selecting features (Speed, Weather, Light, Age) and Target (Severity)
# We convert text categories into numbers (One-Hot Encoding)
X = pd.get_dummies(df.drop('Accident_severity', axis=1)) 
y = df['Accident_severity']

# 3. DATA SPLITTING
# Training: 80% | Testing: 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL DEVELOPMENT (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATION
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred))

# 6. MATHEMATICAL INSIGHT: FEATURE IMPORTANCE
# This shows which factors contribute most to accidents
importances = model.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10 factors

plt.figure(figsize=(10,6))
plt.title('Top 10 Factors Increasing Accident Risk')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance Score')
plt.show()# Python
This project develops an AI-driven safety system using Machine Learning (Random Forest/Neural Networks) to predict road accident risks. By analyzing variables like speed, weather, and driver behavior, the system calculates accident probability and provides real-time alerts, aiming to reduce fatalities through data-backed prevention.
