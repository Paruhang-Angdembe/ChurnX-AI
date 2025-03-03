# Baseline Model Training
"""
A basline models gives you a performance reference point.
Even if it's a simple model like Logistic Regression, it helps you undertand how well you can predict churn with minimal complexity. 
This baseline will be what you try to beat with more advance models later.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

X_train = pd.read_csv("data/processed_X_train.csv")
X_test = pd.read_csv("data/processed_X_test.csv")
y_train = pd.read_csv("data/processed_y_train.csv").squeeze() # Convert DataFrame to Series
y_test = pd.read_csv("data/processed_y_test.csv").squeeze()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
"""
We're using logisctic regression as a starting point because its fast and provides clear insights into the feature contributions. 
The metrics, Accuracy, F1, ROC, AUC, give a comprehensive view of performance, especially important for imbalanced data.

If you notice non-linear patterns or interactions in your EDA,
Consider exploring other models like Random Forests, Gradient Boosting,or other non-linear models or even neural networks if needed.
"""
# Evaluating the Model:
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

joblib.dump(model, "artifacts/model.joblib")

'''
Accuracy: 0.79: 
    79% of all predictions were correct. Good for balance data, but can be misleading if the data is imbalanced.

F1 Score: 0.58
    Balances precision and recall, giving insight into how well the model performs on the minority class. A score of 0.58 suggests room for improvement,
    especially if false negatives or false positives are critical.
    
ROC AUC: 0.84
    Indicates that the model has good discriminative ability between churners and non-churners.
    
Precision and Recall:
    If your model predicts that a customer will churn, precision tells you how many of those predictions are actually correct.
    High precision means that when your model flags a case as a churn, its actuall right.
    
    Recall tells you how well your model captures all the cases of churn. High recall means that most actual churners are detected by your model.
'''