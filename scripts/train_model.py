import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. Load Data
# ==========================================
print("Loading dataset...")
df = pd.read_csv('features.csv')

# Separate features (X) and labels (y)
X = df[['V_rms', 'I_rms', 'THD_V', 'Frequency']]
y = df['Label']

# ==========================================
# 2. Split Data into Training and Testing
# ==========================================
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ==========================================
# 3. Train the AI Model
# ==========================================
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. Evaluate the Model
# ==========================================
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n==========================================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"==========================================\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. Plot Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Power Quality Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ==========================================
# 6. Show Feature Importance
# ==========================================
importances = model.feature_importances_
features = X.columns
print("\nFeature Importance (How much each feature matters to the AI):")
for feature, imp in zip(features, importances):
    print(f"  {feature}: {imp:.4f}")