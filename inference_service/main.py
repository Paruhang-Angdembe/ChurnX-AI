from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import pandas as pd
# from openai import OpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")

# openai.api_key = os.getenv("OPENAI_API_KEY")


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Set your OpenAI API key from the environment

# Define a Pydantic model for input data.
class ChurnInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: float
    Contract: str
    InternetService: str

# Global variable to store the loaded pipeline
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        # Load the full pipeline (including preprocessing and model)
        pipeline = joblib.load("artifacts/full_pipeline.joblib")
        print("Pipeline loaded successfully!")
    except Exception as e:
        print("Error loading pipeline:", e)
        raise HTTPException(status_code=500, detail="Pipeline not loaded.")
    yield
    pipeline = None
    print("Cleaned up pipeline.")

app = FastAPI(title=":ChurnX-AI Inference Service with API-based LLM Explanation", lifespan=lifespan)

@app.post("/predict_churn")
async def predict_churn(input_data: ChurnInput, tuning: float = Query(1.0, description="Tuning parameter placeholder")):
    try: 
        data = pd.DataFrame([input_data.model_dump()])
        prediction = pipeline.predict(data)
        prediction_proba = pipeline.predict_proba(data)[:,1]
        return{
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0]),
            "tuning_used": tuning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_llm_explanation_api(prompt: str)->str:
    '''
    Generate an explanation using OpenAI's API.
    This funciton sends a prompt to OpenAI and returns the generated explanation.
    '''        
    try:
        messages =[
            {
                "role":"system", "content": "You are a helpful assistant that provides concise explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = client.chat.completions.create(model=MODEL_NAME,
        messages=messages,
        max_tokens=150,
        temperature=0.7)
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"



@app.post("/explain_churn")
async def explain_churn(input_data: ChurnInput, tuning: float = Query(1.0, description="Tuning parameter placeholder")):
    try:
        data = pd.DataFrame([input_data.model_dump()])
        prediction = pipeline.predict(data)
        prediction_proba = pipeline.predict_proba(data)[:, 1]
        # Build a prompt for the LLM using input data and prediction results.
        prompt_text = (
            f"Customer details: {input_data.model_dump()}.\n"
            f"The model predicted churn = {int(prediction[0])} with probability = {float(prediction_proba[0]):.2f}.\n"
            "Provide a concise explanation of why the customer might churn."
        )
        explanation = generate_llm_explanation_api(prompt_text)
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0]),
            "explanation": explanation,
            "tuning_used": tuning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
