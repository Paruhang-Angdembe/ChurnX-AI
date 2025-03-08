import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")# Import necessary libraries

# Define which columns are numeric and which are categorical.
# Adjust these lists based on your actual dataset.
numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "InternetService"]

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

# Create a pipeline that first preprocesses the data, then trains a classifier.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Prepare your features and target.
X = df.drop("Churn", axis=1)
y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the pipeline.
pipeline.fit(X_train, y_train)

# Evaluate the pipeline.
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Save the entire pipeline.
joblib.dump(pipeline, "artifacts/full_pipeline.joblib")
