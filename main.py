import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# 1. Load data
url = 'diabetes.csv'
df = pd.read_csv(url)
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# 2. Data Preprocessing: Replace invalid zero values with the median.
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3. Separate features from labels
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  #

# 5. Feature Selection (Pearson + Lasso + Random Forest)
# 5.1 Pearson correlation coefficient filtering (retain features with absolute correlation coefficient > 0.1)
pearson_corr = X_scaled.corrwith(y).abs()
pearson_features = pearson_corr[pearson_corr > 0.1].index.tolist()

# 5.2 Lasso regression filtering (retain features with non-zero coefficients)

lasso = LassoCV(cv=5, max_iter=10000).fit(X_scaled, y)
lasso_features = X_scaled.columns[lasso.coef_ != 0].tolist()

# 5.3 Random Forest feature importance filtering (retain features with importance > 0.05)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
rf_importance = pd.Series(rf.feature_importances_, index=X_scaled.columns)
rf_features = rf_importance[rf_importance > 0.05].index.tolist()

# 5.4 Take the intersection of features from the three methods (retain at least 3 features)
selected_features = list(set(pearson_features) & set(lasso_features) & set(rf_features))
if len(selected_features) < 3:
    selected_features = list(set(pearson_features) | set(lasso_features) | set(rf_features))

X_selected = X_scaled[selected_features]

# 6. Split the dataset (stratified sampling)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 7. perform XGBoost hyperparameter tuning
param_grid = {
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'gamma': [0.1, 0.2]
}

xgb = XGBClassifier(
    n_estimators=200,
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

# 8. train and evaluate the model

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (Tuned)": best_xgb
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Recall": round(recall_score(y_test, y_pred), 2),
        "F1": round(f1_score(y_test, y_pred), 2),
        "AUC": round(roc_auc_score(y_test, y_proba), 3)  #
    }
    results.append(metrics)

# 9. print results
print("\nModel Performance Comparison:")
print("| Model               | Accuracy | Recall | F1  | AUC    |")
print("|---------------------|----------|--------|-----|--------|")
for res in results:
    print(f"| {res['Model']:<19} | {res['Accuracy']:>8} | {res['Recall']:>6} | {res['F1']:>4} | {res['AUC']:>6} |")

# 10. SHAP conduct interpretability analysis.
explainer = shap.Explainer(best_xgb, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
plt.title("XGBoost的SHAP Characteristic Importance Analysis ")
plt.show()
