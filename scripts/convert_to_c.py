import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import emlearn

# ==========================================
# 1. Load Data & Train Model
# ==========================================
print("Training model for C conversion...")
df = pd.read_csv('features.csv')
X = df[['V_rms', 'I_rms', 'THD_V', 'Frequency']]
y = df['Label']

# We use 10 trees instead of 100 to keep the C code small enough for an STM32
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X, y)

# ==========================================
# 2. Convert to C Code
# ==========================================
print("Converting AI model to C code...")
# Convert the sklearn model to an emlearn model
cmodel = emlearn.convert(model, method='inline')

# Save as a C header file
cmodel.save(file='model.h', name='power_quality_model')

print("\nSuccess! Generated 'model.h'.")
print("You can now include this file in your STM32 firmware.")