import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

model = lgb.Booster(model_file='model.txt')

app = FastAPI()

# Define the input schema
class CarFeatures(BaseModel):
    transmission: float
    mileage: float
    fuelType: float
    mpg: float
    engineSize: float
    brand: float
    car_age: float

@app.post("/predict")
def predict(features: CarFeatures):
    mileage_per_year = features.mileage / (features.car_age + 1),
    data = np.array([[features.transmission, features.fuelType, features.mpg, features.engineSize, features.brand, features.car_age, mileage_per_year]])
    prediction = model.predict(data)
    return {"prediction": prediction[0]}
