from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# CORS middleware setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define your input model
class ModelInput(BaseModel):
    age: int
    gender: int
    bmi: float
    children: int
    smoke: int
    region: int

# Load your model
insure_model = joblib.load("regressor.joblib")

# Endpoint for prediction
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
    
    return {"prediction": prediction[0]}

# Optional: Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}
