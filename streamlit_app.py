import streamlit as st
import requests

st.title("Test de l'API ImmoPrix")

st.write("Entrez les caractéristiques de la maison pour obtenir une prédiction du prix médian.")

MedInc = st.number_input("MedInc", value=3.0)
HouseAge = st.number_input("HouseAge", value=20.0)
AveRooms = st.number_input("AveRooms", value=5.0)
AveBedrms = st.number_input("AveBedrms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("AveOccup", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("Prédire le prix médian"):
    data = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    if response.status_code == 200:
        prediction = response.json()["predicted_med_house_value"]
        st.success(f"Prix médian prédit : {prediction:.2f} (en centaines de milliers $)")
    else:
        st.error("Erreur lors de la prédiction")
