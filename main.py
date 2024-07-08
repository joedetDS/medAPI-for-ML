from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

app = FastAPI()

#set up the CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"]
    )

class ModelInput(BaseModel):
    age: int
    gender: int
    bmi: float
    children: int
    smoke: int
    region: int

insure_model = joblib.load("regressor.joblib")

#Create the end point
@app.post("/med_prediction")
def insurance_pred(input_parameters: ModelInput):
    age = input_parameters.age
    gender = input_parameters.gender
    bmi = input_parameters.bmi
    children = input_parameters.children
    smoke = input_parameters.smoke
    region = input_parameters.region

    input_list = [age, gender, bmi, children, smoke, region]

    prediction = insure_model.predict([input_list])
    
    return prediction[0]
