import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load data and artifacts
# ------------------------------
print("Loading data and artifacts...")
df = pd.read_csv('landslide_dataset.csv')

# Load the saved scaler (to standardize exactly as in deployment)
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Encode target
df['Evaluation_encoded'] = label_encoder.fit_transform(df['Evaluation'])
target_names = label_encoder.classes_   # ['High Risk', 'Low Risk', 'Medium Risk']
print("Target classes:", target_names)

X = df[['Slope_Angle', 'Rainfall_Intensity_mmh', 'Rainfall_Infiltration_sec']]
y = df['Evaluation_encoded']

# Use the same split as before (for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale using the saved scaler (fit_transform already done in training script)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 2. Fit with statsmodels
# ------------------------------
X_train_sm = sm.add_constant(X_train_scaled)  # adds intercept column
model_sm = MNLogit(y_train, X_train_sm)
result_sm = model_sm.fit(maxiter=1000, method='lbfgs', disp=True)

# ------------------------------
# 3. Print full summary
# ------------------------------
print("\n" + "="*80)
print("FULL STATISTICAL SUMMARY")
print("="*80)
print(result_sm.summary())
print("\n" + "="*80)
print("SUMMARY2 (with AIC/BIC)")
print("="*80)
print(result_sm.summary2())

# ------------------------------
# 4. Extract coefficients, p-values, and odds ratios per class
# ------------------------------
# The baseline class is the first in sorted order: 'High Risk'
baseline = target_names[0]
print(f"\nBaseline (reference) class: {baseline}")

# Get parameters and confidence intervals as DataFrames with MultiIndex
params = result_sm.params           # shape: (n_classes, n_vars) with MultiIndex (class, var)
conf_int = result_sm.conf_int()     # same MultiIndex, but two columns: 0 and 1 for lower/upper
pvals = result_sm.pvalues           # same MultiIndex

# For each non‑baseline class, print a table
for cls in target_names[1:]:
    print(f"\n--- Analysis for class: {cls} (vs {baseline}) ---")
    # Extract data for this class using .xs
    coefs = params.xs(cls, level=0)
    lower = conf_int.xs(cls, level=0).iloc[:, 0]
    upper = conf_int.xs(cls, level=0).iloc[:, 1]
    pval = pvals.xs(cls, level=0)
    std_err = result_sm.bse.xs(cls, level=0)
    z = result_sm.tvalues.xs(cls, level=0)
    odds = np.exp(coefs)
    
    # Build a DataFrame
    table = pd.DataFrame({
        'Coefficient': coefs,
        'Std Error': std_err,
        'z': z,
        'P>|z|': pval,
        'CI Lower': lower,
        'CI Upper': upper,
        'Odds Ratio': odds
    })
    # Rename index to meaningful feature names (including Intercept)
    table.index = ['Intercept'] + list(X.columns)
    print(table.round(4))

# ------------------------------
# 5. Model fit statistics
# ------------------------------
print("\n" + "="*50)
print("MODEL FIT STATISTICS")
print("="*50)
print(f"Log-Likelihood:          {result_sm.llf:.2f}")
print(f"Log-Likelihood (Null):   {result_sm.llnull:.2f}")
print(f"Pseudo R-squared (McFadden): {result_sm.prsquared:.4f}")
print(f"AIC:                     {result_sm.aic:.2f}")
print(f"BIC:                     {result_sm.bic:.2f}")

# ------------------------------
# 6. Visualize coefficients with confidence intervals
# ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
features = X.columns

for ax, feature in enumerate(features):
    # Get coefficient and CI for this feature for each non‑baseline class
    classes = target_names[1:]
    coefs = []
    ci_low = []
    ci_high = []
    for cls in classes:
        # For the given feature (column name), get the value
        # The column index corresponds to the feature order: 'const' is column 0, then features
        # We'll extract using the feature name from params
        # params has columns named: 'const', 'Slope_Angle', 'Rainfall_Intensity_mmh', 'Rainfall_Infiltration_sec'
        # But after .xs, we have a Series with index the feature names
        coef_series = params.xs(cls, level=0)
        coefs.append(coef_series[feature])
        # Confidence intervals for that feature
        ci_df = conf_int.xs(cls, level=0)
        ci_low.append(ci_df.loc[feature, 0])
        ci_high.append(ci_df.loc[feature, 1])
    
    # Plot horizontal error bars
    y_pos = np.arange(len(classes))
    ax.errorbar(coefs, y_pos, xerr=[coefs - ci_low, ci_high - coefs],
                fmt='o', capsize=5, color='steelblue')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_title(f'Effect of {feature}')
    ax.set_xlabel('Coefficient')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Logistic Regression Coefficients with 95% CI\n(Reference: High Risk)', y=1.02)
plt.tight_layout()
plt.savefig('logistic_analysis_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------
# 7. Save full analysis to a text file
# ------------------------------
with open('logistic_analysis_report.txt', 'w') as f:
    f.write("LOGISTIC REGRESSION ANALYSIS REPORT\n")
    f.write("="*80 + "\n")
    f.write(result_sm.summary().as_text())
    f.write("\n\n")
    f.write("ODDS RATIOS AND P-VALUES DETAIL\n")
    f.write("="*80 + "\n")
    for cls in target_names[1:]:
        f.write(f"\n--- Class: {cls} (vs {baseline}) ---\n")
        coefs = params.xs(cls, level=0)
        odds = np.exp(coefs)
        pval = pvals.xs(cls, level=0)
        for feat in coefs.index:
            f.write(f"{feat:>25} | Coef: {coefs[feat]:>8.4f} | Odds: {odds[feat]:>8.4f} | P>|z|: {pval[feat]:>8.4f}\n")

print("\n✅ Analysis complete! Summary saved to 'logistic_analysis_report.txt'")
print("📊 Coefficient plot saved as 'logistic_analysis_coefficients.png'")