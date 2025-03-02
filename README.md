# ChurnX-AI

## Overview
ChurnX-AI is a hybrid AI assistant designed to predict customer churn and provide natural language explanations for those predictions. The project aims to combine traditional machine learning with a local LLM (Large Language Model)or [`RAG`], to help businesses understand why customers might leave, enabling them to take proactive measures. The system will process customer data, train predictive models, deploy an inference service, and integrate a user-friendly front-end dashboard along with an explanation module. 


---

## Project Progress

### Phase 1: EDA (Completed)
- **Exploratory Data Analysis:**  
  - Analyzed the Telco Customer Churn dataset to understand distributions, missing values, and potential predictive features.
  - Found an imbalance in the churn variable (`Yes` vs. `No`).
  - Identified numerical and categorical columns; created histograms and count plots.
  - Generated a correlation heatmap to explore relationships among numeric features.

- **Key Insights:**  
  - Majority of customers are non-churners (imbalanced dataset).
  - Weak to moderate correlations among numeric features (tenure, monthly charges, etc.).
  - Data cleansing steps (handling missing values, etc.) will be crucial before modeling.

### Architecture Diagram 
 `/docs/Architecture-Diagram.drawio.png`

---

## Next Steps
- **Data Preprocessing:**
  - Handle missing values (drop or impute).
  - Convert categorical variables (one-hot encoding or label encoding).
  - Split data into train/test sets.

- **Baseline Model Training:**
  - Train a simple model (Logistic Regression or Random Forest).
  - Evaluate performance (Accuracy, F1, ROC AUC).
  - Save the model artifact (`model.joblib`) in an `/artifacts` folder.

- **Documentation:**
  - Update README with instructions on how to run preprocessing and training scripts.
  - Include performance metrics (Accuracy, F1, ROC AUC) once baseline is established.

---

## Repository Structure

```plaintext
├── data
│   └── telco_customer_churn.csv
├── docs
│   └── architecture_diagram.png  (To be added soon)
├── notebooks
│   └── EDA.ipynb
├── src
│   ├── preprocess.py  (Planned for Phase 2)
│   └── train.py       (Planned for Phase 2)
├── tests
│   └── ...
└── README.md          (This file)
