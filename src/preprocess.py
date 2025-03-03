import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

df.dropna(inplace=True) # Handling missing values as it can skew your model or even cause errors

df_encoded = pd.get_dummies(df, drop_first=True) # Encoding Categorical Variables; text-based categorical features into a numerical format

# print(df_encoded.columns)  


'''
Notes:
    If a categorical feature has too many unqiue values (high cardinality), one-hot encoding may create too many features. 
    In that case, target encoding or label encoding might be more appropriate.
    Some tree-based models (like Random Forests) are less sensitive to encoding, so sometimes simple label encoding is sufficient.
'''

# Splitting data
X = df_encoded.drop("Churn_Yes", axis=1) 
y = df_encoded["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
'''
Notes:
    X: This varialbe represents the input features. It includes all the columns in the dataset except the one youre trying to predict.
    y: This is the Target variable. It contains the output you want to predict, "Churn" column.
    
    X_train & y_train: These are the training sets--roughly 80% of the data (since test_size=0.2) used to teach the model how to make predictions.
    X_test and y_test: These are the testing sets--the remaining 20% of the data used to evaluate how well the model performs on unseen data.
    
    With a fixed random_state, the order of randomness is predetermined. That ensures not only the same number of rows in each set (e.g, 80% in training, 20% testing)
    but also the exact same rows each time you run the code.

    Reproducible, It means that if you or someone elseruns the code with the same data, you'll both get identical training and testing sets, making it easier to compare results over time.
    
    Stratification, ensures the clas distribution (churn vs non-churn) is maintained in both sets, which is crucial for imbalanced datasets.
'''

# Saving Preprocessed Data
X_train.to_csv("data/processed_X_train.csv", index=False)
X_test.to_csv("data/processed_X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed_y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed_y_test.csv", index=False)

