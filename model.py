print("🚀 Starting model training...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv('dataset.csv')

# Features & target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale
print("⚙️ Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
print("⚖️ Applying SMOTE...")
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_scaled, y)

# Split
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

# Train model
print("🤖 Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('fraud_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ Model trained successfully!")