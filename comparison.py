import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Load and prepare data
# ---------------------------
df = pd.read_csv('landslide_dataset.csv')
print("Dataset head:\n", df.head())

# Encode target labels: Low Risk=0, Medium Risk=1, High Risk=2
le = LabelEncoder()
df['Evaluation_encoded'] = le.fit_transform(df['Evaluation'])
# Keep mapping for later interpretation
target_names = le.classes_   # ['High Risk', 'Low Risk', 'Medium Risk'] - order depends on fit

# Features
X = df[['Slope_Angle', 'Rainfall_Intensity_mmh', 'Rainfall_Infiltration_sec']]
y = df['Evaluation_encoded']

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)

# ---------------------------
# 2. Scale features (needed for Logistic Regression)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. Define models
# ---------------------------
models = {
    'Logistic Regression': LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

# We'll store results
results = {}
confusion_matrices = {}
feature_importances = {}

# ---------------------------
# 4. Train & evaluate each model
# ---------------------------
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    # For Logistic Regression use scaled data; others use raw
    if name == 'Logistic Regression':
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train
    model.fit(X_train_use, y_train)
    # Predict
    y_pred = model.predict(X_test_use)
    y_proba = model.predict_proba(X_test_use) if hasattr(model, 'predict_proba') else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances[name] = importances
    elif name == 'Logistic Regression':
        # For logistic regression, we can use coefficients (average absolute per feature)
        # For multinomial, we have coefficients for each class; take mean absolute over classes
        coefs = model.coef_  # shape (n_classes, n_features)
        mean_abs_coef = np.mean(np.abs(coefs), axis=0)
        feature_importances[name] = mean_abs_coef
    else:
        feature_importances[name] = None
    
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

# ---------------------------
# 5. Visualizations
# ---------------------------
# 5.1 Performance comparison bar chart
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind='bar', figsize=(10,6), colormap='viridis')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()

# 5.2 Confusion matrices heatmaps
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# 5.3 Feature importance (for each model that provides it)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, imp) in zip(axes, feature_importances.items()):
    if imp is not None:
        # Normalize importance (for logistic regression we already have mean absolute coef)
        if name == 'Logistic Regression':
            # We can plot them as bars; use the feature names
            features = X.columns
            # Sort for better visualization
            idx = np.argsort(imp)
            ax.barh(features[idx], imp[idx], color='skyblue')
            ax.set_title(f'Feature Importance - {name}\n(mean |coef|)')
        else:
            # Tree-based models
            features = X.columns
            sorted_idx = np.argsort(imp)
            ax.barh(features[sorted_idx], imp[sorted_idx], color='skyblue')
            ax.set_title(f'Feature Importance - {name}')
        ax.set_xlabel('Importance')
    else:
        ax.text(0.5, 0.5, 'No importance available', ha='center', va='center')
        ax.set_title(name)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 5.4 Logistic Regression specific: coefficients and odds ratios
if 'Logistic Regression' in models:
    lr_model = models['Logistic Regression']
    coefs = lr_model.coef_  # shape (3,3)
    intercepts = lr_model.intercept_
    odds_ratios = np.exp(coefs)  # exponentiate coefficients
    # Create a DataFrame for clarity
    coef_df = pd.DataFrame(coefs, columns=X.columns, index=target_names)
    odds_df = pd.DataFrame(odds_ratios, columns=X.columns, index=target_names)
    print("\n--- Logistic Regression Coefficients (per class) ---")
    print(coef_df)
    print("\n--- Odds Ratios (exp(coef)) ---")
    print(odds_df)
    
    # Plot coefficients per feature per class
    coef_df.T.plot(kind='bar', figsize=(10,6))
    plt.title('Logistic Regression Coefficients per Feature and Class')
    plt.ylabel('Coefficient')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('logistic_coefficients.png')
    plt.show()

# 5.5 (Optional) ROC curves for multi-class (One-vs-Rest)
# We'll compute ROC for each class for Logistic Regression (since it has proba)
if 'Logistic Regression' in models:
    lr = models['Logistic Regression']
    y_proba_lr = lr.predict_proba(X_test_scaled)
    # Compute ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba_lr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8,6))
    for i in range(len(target_names)):
        plt.plot(fpr[i], tpr[i], label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Logistic Regression (One-vs-Rest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curves_logistic.png')
    plt.show()

print("\n--- All done! Check the generated PNG files for visualizations. ---")