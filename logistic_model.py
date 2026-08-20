import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------
df = pd.read_csv('landslide_dataset.csv')
print("Dataset loaded. Shape:", df.shape)

# -------------------------------------------------
# 2. Encode target labels
# -------------------------------------------------
label_encoder = LabelEncoder()
df['Evaluation_encoded'] = label_encoder.fit_transform(df['Evaluation'])
# Save the mapping for later interpretation
target_classes = label_encoder.classes_   # e.g., ['High Risk', 'Low Risk', 'Medium Risk']
print("Target classes:", target_classes)

# Features and target
X = df[['Slope_Angle', 'Rainfall_Intensity_mmh', 'Rainfall_Infiltration_sec']]
y = df['Evaluation_encoded']

# -------------------------------------------------
# 3. Split data (to assess final performance)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# 4. Scale features
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 5. Train Logistic Regression
# -------------------------------------------------
model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# 6. Evaluate (optional, for your reference)
# -------------------------------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal model accuracy on test set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_classes))

# Confusion matrix (as a sanity check)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)

# -------------------------------------------------
# 7. Save the model, scaler, and label encoder
# -------------------------------------------------
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\n✅ All artifacts saved:")
print("   - logistic_model.pkl")
print("   - scaler.pkl")
print("   - label_encoder.pkl")