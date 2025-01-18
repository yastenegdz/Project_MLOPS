# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API de Prédiction Immobilière")

# Chargement du modèle pré-entraîné 
model = joblib.load("best_model.pkl")  # Sauvegarde du modèle lors de l'entraînement avec joblib

# Définition d'un schéma de données pour l'entrée
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Prédiction Immobilière"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convertir les données en tableau numpy pour la prédiction
    input_data = np.array([[ 
        features.MedInc, 
        features.HouseAge, 
        features.AveRooms, 
        features.AveBedrms, 
        features.Population, 
        features.AveOccup, 
        features.Latitude, 
        features.Longitude 
    ]])
    
    prediction = model.predict(input_data)
    # Retourne la prédiction sous forme de dictionnaire
    return {"predicted_med_house_value": prediction[0]}
