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
    # Add additional fields as needed
    # The field names should match the feature names in the training data
    
# Use a global dictionary to store resources
ml_models = {}
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the ML model
    try: 
        ml_models["model"] = joblib.load("../artifacts/model.joblib")
        print("Model loaded successfully")
    except Exception as e:
        print("Error loading model:", e)
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Yield control back to the app; the code below will run on shutdown.
    yield
    # Shutdown: clean up resources
    ml_models.clear()
    print("Cleaned up ML models.")
    
    
# Create the FastAPI app with the lifespan parameter.
app = FastAPI(title= "ChurnX-AI inference Service", lifespan=lifespan)

@app.post("/predict_churn")
async def predict_churn(input_data: ChurnInput):
    try:
        # Convert input data (Pydantic model) into a Dataframe.
        data = pd.DataFrame([input_data.model_dump()])
        # Retrieve the loaded model from pour global dictionary
        model = ml_models.get("model")
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        # Make Predictoin.
        prediction = model.predict(data)
        prediction_proba = model.predict_proba(data)[:, 1]
        return{
            "prediction": int(prediction[0]), # 0 for no churn, 1 for churn
            "probability": float(prediction_proba[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))