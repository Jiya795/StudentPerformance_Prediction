"""
Training script for the Student Performance Predictor.

Improvements over the original version:
- Cross-validation (not just a single train/test split) for honest performance numbers
- Model comparison: Random Forest vs Linear/Logistic baselines
- Small hyperparameter search for the Random Forest models
- Data-leakage check: G1/G2 (period grades) are extremely predictive of G3 by
  construction. We report accuracy WITH and WITHOUT them so this is transparent
  instead of hidden behind one inflated accuracy number.
- All metrics are written to metrics.json so app.py/templates never hardcode
  numbers that go stale after retraining.
"""

import json
import time
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error,
    confusion_matrix, precision_score, recall_score, f1_score
)

os.makedirs('static/charts', exist_ok=True)

df = pd.read_csv('data.csv')
df['result'] = (df['G3'] >= 10).astype(int)

FEATURES = ['studytime', 'failures', 'absences', 'G1', 'G2', 'internet',
            'higher', 'Medu', 'Fedu', 'health', 'famrel', 'freetime']
FEATURES_NO_LEAK = [f for f in FEATURES if f not in ('G1', 'G2')]

X = df[FEATURES]
y_class = df['result']
y_reg = df['G3']

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

metrics = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

# ---------------------------------------------------------------------------
# 1. Data leakage check: how much do G1/G2 alone explain G3?
# ---------------------------------------------------------------------------
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)
X_noleak = df[FEATURES_NO_LEAK]
scaler_noleak = StandardScaler()
X_noleak_scaled = scaler_noleak.fit_transform(X_noleak)

acc_with_leak = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42),
                                 X_full_scaled, y_class, cv=cv_class, scoring='accuracy').mean()
acc_without_leak = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42),
                                    X_noleak_scaled, y_class, cv=cv_class, scoring='accuracy').mean()
r2_with_leak = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42),
                                X_full_scaled, y_reg, cv=cv_reg, scoring='r2').mean()
r2_without_leak = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42),
                                   X_noleak_scaled, y_reg, cv=cv_reg, scoring='r2').mean()

metrics['leakage_analysis'] = {
    "note": "G1/G2 are prior-period grades and are strongly correlated with the "
            "final grade G3 by construction. Numbers below show how much of the "
            "model's accuracy actually comes from G1/G2 vs. the behavioural/"
            "demographic features. The deployed model still uses G1/G2 because "
            "the product intentionally asks users for their in-progress grades, "
            "but this is reported here for transparency.",
    "classifier_accuracy_with_G1_G2": round(float(acc_with_leak), 4),
    "classifier_accuracy_without_G1_G2": round(float(acc_without_leak), 4),
    "regressor_r2_with_G1_G2": round(float(r2_with_leak), 4),
    "regressor_r2_without_G1_G2": round(float(r2_without_leak), 4),
}

# ---------------------------------------------------------------------------
# 2. Model comparison (baseline vs Random Forest) via cross-validation
# ---------------------------------------------------------------------------
acc_logreg = cross_val_score(LogisticRegression(max_iter=1000), X_train_s, yc_train,
                              cv=cv_class, scoring='accuracy').mean()
acc_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train_s, yc_train,
                          cv=cv_class, scoring='accuracy').mean()

r2_linreg = cross_val_score(LinearRegression(), X_train_s, yr_train, cv=cv_reg, scoring='r2').mean()
r2_rf = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X_train_s, yr_train,
                         cv=cv_reg, scoring='r2').mean()

metrics['model_comparison'] = {
    "classifier": {
        "logistic_regression_cv_accuracy": round(float(acc_logreg), 4),
        "random_forest_cv_accuracy": round(float(acc_rf), 4),
        "chosen": "random_forest" if acc_rf >= acc_logreg else "logistic_regression",
    },
    "regressor": {
        "linear_regression_cv_r2": round(float(r2_linreg), 4),
        "random_forest_cv_r2": round(float(r2_rf), 4),
        "chosen": "random_forest" if r2_rf >= r2_linreg else "linear_regression",
    }
}

# ---------------------------------------------------------------------------
# 3. Small hyperparameter search for the Random Forest candidates
# ---------------------------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 6, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                         cv=cv_class, scoring='accuracy', n_jobs=-1)
grid_clf.fit(X_train_s, yc_train)

grid_reg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                         cv=cv_reg, scoring='r2', n_jobs=-1)
grid_reg.fit(X_train_s, yr_train)

metrics['hyperparameter_search'] = {
    "classifier_rf_best_params": grid_clf.best_params_,
    "classifier_rf_best_cv_accuracy": round(float(grid_clf.best_score_), 4),
    "regressor_rf_best_params": grid_reg.best_params_,
    "regressor_rf_best_cv_r2": round(float(grid_reg.best_score_), 4),
}

# ---------------------------------------------------------------------------
# 3b. Actually pick the winning model type instead of assuming Random Forest.
# The tuned RF is compared against the plain baseline on the same CV folds;
# whichever generalizes better on cross-validation is the one we deploy.
# ---------------------------------------------------------------------------
if grid_clf.best_score_ >= acc_logreg:
    clf = grid_clf.best_estimator_
    clf_choice = "random_forest (tuned)"
else:
    clf = LogisticRegression(max_iter=1000).fit(X_train_s, yc_train)
    clf_choice = "logistic_regression"

if grid_reg.best_score_ >= r2_linreg:
    reg = grid_reg.best_estimator_
    reg_choice = "random_forest (tuned)"
else:
    reg = LinearRegression().fit(X_train_s, yr_train)
    reg_choice = "linear_regression"

metrics['deployed_model'] = {
    "classifier": clf_choice,
    "regressor": reg_choice,
    "reasoning": "Each candidate was scored with the same 5-fold cross-validation; "
                 "the model with the higher CV score was deployed rather than "
                 "defaulting to Random Forest."
}

# ---------------------------------------------------------------------------
# 4. Final evaluation on the held-out test set
# ---------------------------------------------------------------------------
yc_pred = clf.predict(X_test_s)
acc = accuracy_score(yc_test, yc_pred)
prec = precision_score(yc_test, yc_pred, zero_division=0)
rec = recall_score(yc_test, yc_pred, zero_division=0)
f1 = f1_score(yc_test, yc_pred, zero_division=0)
cm = confusion_matrix(yc_test, yc_pred)

yr_pred = reg.predict(X_test_s)
r2 = r2_score(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5

metrics['final_test_set_performance'] = {
    "classifier_accuracy": round(float(acc), 4),
    "classifier_precision": round(float(prec), 4),
    "classifier_recall": round(float(rec), 4),
    "classifier_f1": round(float(f1), 4),
    "regressor_r2": round(float(r2), 4),
    "regressor_rmse": round(float(rmse), 4),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "total_samples": int(len(df)),
    "n_features": len(FEATURES),
}

print(f"Classifier test accuracy: {acc:.2%} (precision {prec:.2f}, recall {rec:.2f}, f1 {f1:.2f})")
print(f"Regressor test R2: {r2:.3f} (RMSE {rmse:.2f})")
print(f"Leakage check -> accuracy with G1/G2: {acc_with_leak:.2%} vs without: {acc_without_leak:.2%}")

# ---------------------------------------------------------------------------
# Save models + metrics
# ---------------------------------------------------------------------------
pickle.dump(clf, open('clf_model.pkl', 'wb'))
pickle.dump(reg, open('reg_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(FEATURES, open('features.pkl', 'wb'))

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
COLORS = {'blue': '#378ADD', 'green': '#639922', 'red': '#E24B4A', 'purple': '#7F77DD',
          'amber': '#EF9F27', 'teal': '#1D9E75'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                      'axes.spines.top': False, 'axes.spines.right': False})

if hasattr(clf, 'feature_importances_'):
    fi_values = clf.feature_importances_
    fi_label = 'Importance Score'
else:
    fi_values = np.abs(clf.coef_[0])
    fi_label = 'Absolute Coefficient (Logistic Regression)'
fi = pd.Series(fi_values, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
colors = [COLORS['blue'] if v > fi.median() else COLORS['teal'] for v in fi]
ax.barh(fi.index, fi.values, color=colors)
ax.set_xlabel(fi_label)
ax.set_title(f'Feature Importance ({clf_choice})', fontweight='bold', pad=12)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('static/charts/feature_importance.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
bins = range(0, 22, 2)
ax.hist(df[df['result'] == 1]['G3'], bins=bins, alpha=0.7, color=COLORS['green'], label='Pass')
ax.hist(df[df['result'] == 0]['G3'], bins=bins, alpha=0.7, color=COLORS['red'], label='Fail')
ax.set_xlabel('Final Grade (G3)')
ax.set_ylabel('Number of Students')
ax.set_title('Grade Distribution', fontweight='bold', pad=12)
ax.axvline(x=10, color='gray', linestyle='--', linewidth=1, label='Pass mark (10)')
ax.legend()
plt.tight_layout()
plt.savefig('static/charts/grade_dist.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
st_avg = df.groupby('studytime')['G3'].mean()
ax.bar(st_avg.index, st_avg.values, color=COLORS['purple'], width=0.5)
labels = {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}
ax.set_xticks(list(labels.keys()))
ax.set_xticklabels(list(labels.values()))
ax.set_ylabel('Average Grade')
ax.set_title('Study Time vs Average Grade', fontweight='bold', pad=12)
for i, v in zip(st_avg.index, st_avg.values):
    ax.text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('static/charts/studytime_grade.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
corr = df[FEATURES + ['G3']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.1f', annot_kws={'size': 8},
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('static/charts/correlation.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
            cbar=False, annot_kws={'size': 14})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Test Set)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('static/charts/confusion_matrix.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
bars_labels = ['With G1/G2', 'Without G1/G2']
bars_values = [acc_with_leak * 100, acc_without_leak * 100]
ax.bar(bars_labels, bars_values, color=[COLORS['red'], COLORS['teal']], width=0.5)
ax.set_ylabel('Classifier CV Accuracy (%)')
ax.set_title('Impact of G1/G2 on Reported Accuracy', fontweight='bold', pad=12)
for i, v in enumerate(bars_values):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('static/charts/leakage_comparison.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

print("All charts saved. metrics.json written. Model training complete!")
