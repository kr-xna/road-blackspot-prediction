"""
Road crash blackspot prediction.

Trains a logistic regression model to predict whether a road location is a
crash blackspot from socio-economic and land-use features, then reports
accuracy, precision, recall, F1 and ROC AUC.

Usage:
    python blackspot_prediction.py

Expects Blackspot.csv in the working directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score)


# Load Data
data = pd.read_csv('Blackspot.csv', encoding='ISO-8859-1')

# Clean Data
data['AGE_65YRS_OVER_PCNT'] = data['AGE_65YRS_OVER_PCNT'].fillna(data['AGE_65YRS_OVER_PCNT'].median())
data['Lq_Licenses'] = data['Lq_Licenses'].fillna(data['Lq_Licenses'].mean())
data['Intersection'] = (data['Intersection'] == 'Intersection').astype(int)
data['Blackspot'] = (data['Blackspot'] == 'Blackspot').astype(int)

# Plot target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Blackspot', data=data)
plt.title('Blackspot Distribution')
plt.show()

# Feature selection
features = ['SEIFA', 'CARS_ZERO_HH_PCNT', 'Lq_Licenses', 'Supermarket']
X = data[features]
y = data['Blackspot']

# Train-Test Split (stratified so both sets keep the same blackspot ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features — fitted on the training set only, and never applied to the 0/1 target
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

# Coefficients — direction and strength of each feature's effect on blackspot odds
for name, coef in sorted(zip(features, model.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"  {name:<20} {coef:+.3f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

print("Model evaluation complete.")
