import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the data
file_path = "C:\\Users\\James Corbett\\OneDrive\\WHR 15.xlsx"
data = pd.read_excel(file_path)

# Data Preprocessing
data_processed = data.drop(columns=['Country name']).dropna()
data_processed['Life Ladder Class'] = pd.qcut(data_processed['Life Ladder'], 3, labels=["low", "medium", "high"])
X = data_processed.drop(columns=['Life Ladder', 'Life Ladder Class'])
y = data_processed['Life Ladder Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Building and Training
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Model Evaluation
y_pred = random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Output results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Feature Importance Plot
feature_importances = random_forest_model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances', fontsize=16)
plt.barh(range(len(indices)), feature_importances[indices], color=sns.color_palette("viridis", len(indices)))
plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=12)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=random_forest_model.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=random_forest_model.classes_, yticklabels=random_forest_model.classes_)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.show()
