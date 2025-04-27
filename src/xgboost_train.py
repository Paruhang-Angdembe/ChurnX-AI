from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric and handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)  # Replace missing with 0

# Define which columns are numeric and which are categorical.
# Adjust these lists based on your actual dataset.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    "Contract", 
    "InternetService",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "PaymentMethod"
    ]
# Auto-select features (adjust thresholds as needed)

# Create transformers for numeric and categorical data.
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

# Create a preprocessor that applies these transformations.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create a pipeline that preprocesses data and then trains the classifier.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=1000,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    ))
])

# Prepare features and target.
# If you have a customer identifier, you may want to keep it for grouping, but drop it from features.
if "customerID" in df.columns:
    groups = df["customerID"].to_numpy()
    X = df.drop(["Churn"], axis=1)  # Retain customerID for grouping
else:
    groups = None
    X = df.drop("Churn", axis=1)

y = df["Churn"].map({"Yes": 1, "No": 0}).to_numpy()


# Define stratified 5-fold cross-validation
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics.
scoring = {
    "accuracy": "accuracy",
    "f1" : "f1",
    "roc_auc": "roc_auc"
}

# Perform cross-validation (pass groups if available).
if groups is not None:
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, groups=groups)
else:
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display the mean scores.
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.2f}")
print(f"Mean F1 Score: {cv_results['test_f1'].mean():.2f}")
print(f"Mean ROC AUC: {cv_results['test_roc_auc'].mean():.2f}")

# Optionally, train on the full dataset and save the pipeline for inference.
pipeline.fit(X, y)
joblib.dump(pipeline, "artifacts/full_pipeline.joblib")

# This ensures that same preprocessing is applied during inference"
