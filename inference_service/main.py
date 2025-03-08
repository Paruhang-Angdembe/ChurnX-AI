from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Define a Pydantic model for input data
# Make sure to include a;; features your model was trained on
class ChurnInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: float
    Contract: str
    InternetService: str
    # Add additional fields as needed
    # The field names should match the feature names in the training data
    
# Global variable to store the loaded pipeline
pipeline = None

# Use a global dictionary to store resources
ml_models = {}
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try: 
        pipeline = joblib.load("artifacts/full_pipeline.joblib")
        print("Pipeline loaded successfully!")
    except Exception as e:
        print("Error loading pipeline:", e)
        raise HTTPException(status_code=500, detail="Pipeline not loaded.")
    
    # Yield control back to the app; the code below will run on shutdown.
    yield
    # Shutdown: clean up resources
    pipeline = None
    print("Cleaned up Pipeline.")
    
    
# Create the FastAPI app with the lifespan parameter.
app = FastAPI(title= "ChurnX-AI inference Service", lifespan=lifespan)

@app.post("/predict_churn")
async def predict_churn(input_data: ChurnInput):
    try:
        # Convert input data (Pydantic model) into a Dataframe.
        data = pd.DataFrame([input_data.model_dump()])
        #  Use the pipeline to both preprocess and predict
        prediction = pipeline.predict(data)
        prediction_proba = pipeline.predict_proba(data)[:,1]
        return{
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# {
#   "tenure": 10,
#   "MonthlyCharges": 0,
#   "TotalCharges": 0,
#   "SeniorCitizen": 0,
#   "Contract": "Month-to-month",
#   "InternetService": "DSL"
# }
'''
    Means that for the provided inpout data, the model predicts a "0"-"no churn".
    The model estimates there's roughly a 28.67% chance of the customer churning, and it 
    ulimately classifies the customer as not likely to churn.
'''