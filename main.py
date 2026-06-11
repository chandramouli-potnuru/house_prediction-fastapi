from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(area: float, bedrooms: int, bathrooms: int):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    return {
        "predicted_price": float(prediction[0])
    }
