# ChurnX-AI

## Overview
ChurnX-AI is a hybrid AI assistant designed to predict customer churn and provide natural language explanations for those predictions. The project aims to combine traditional machine learning with a local LLM (Large Language Model)or `RAG`, to help businesses understand why customers might leave, enabling them to take proactive measures. The system will process customer data, train predictive models, deploy an inference service, and integrate a user-friendly front-end dashboard along with an explanation module. 

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

### Phase 2: Data Preprocessing & Baseline Model Training (Completed)
- **Data Preprocessing:**  
  - Handled missing values by dropping rows with missing data.
  - Applied one-hot encoding to convert categorical variables.
  - Split the data into training (80%) and testing (20%) sets while stratifying based on churn.
  - Saved the processed data as CSV files.
- **Baseline Model Training:**  
  - Trained a Logistic Regression model as a baseline.
  - Evaluated model performance:
    - **Accuracy:** 0.79
    - **F1 Score:** 0.58
    - **ROC AUC:** 0.84
  - Saved the trained model artifact (`model.joblib`) in the `/artifacts` folder.
- **Notes:**  
  - Future improvements may involve experimenting with more complex models (e.g., Random Forests, Gradient Boosting) if performance requirements are not met.

### Architecture Diagram 
![Architecture Diagram](docs/Architecture-Diagram.drawio.png)

---

## Next Steps: Phase 3 - Inference Service

- **Inference Service with FastAPI & Docker:**
  - **Develop a FastAPI Application:**  
    Build a FastAPI app that loads the baseline model and serves predictions via an API endpoint (e.g., `/predict_churn`).
  - **Local Testing:**  
    Test the API using the interactive Swagger UI provided by FastAPI or tools like Postman.
  - **Containerization:**  
    Create a Dockerfile and containerize your API for easier deployment.
- **Future Enhancements:**
  - Integrate a front-end dashboard (React/Next.js) to interact with the API.
  - Extend the system with an LLM explanation module to generate natural language explanations.
  - Explore advanced model tuning, feature engineering, and CI/CD pipelines for continuous improvements.

---

## Repository Structure

```plaintext
├── data
│   └── telco_customer_churn.csv
├── docs
│   └── Architecture-Diagram.drawio.png
├── notebooks
│   └── EDA.ipynb
├── src
│   ├── preprocess.py   (Phase 2)
│   └── train.py        (Phase 2)
├── artifacts
│   └── model.joblib    (Phase 2)
├── tests
│   └── ...
└── README.md           (This file)
