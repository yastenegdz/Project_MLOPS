import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) CHARGEMENT DU JEU DE DONNÉES
housing = fetch_california_housing(as_frame=True)
df = housing.data.copy()
df['MedHouseVal'] = housing.target  # Ajout de la cible dans le DataFrame

# Vérification du nom des colonnes
print("Colonnes du DataFrame :", df.columns)

# 2) ANALYSE EXPLORATOIRE
# Aperçu rapide (dimensions, types de variables, quelques lignes)
print("\nAperçu du DataFrame :")
print(df.info())
print(df.head())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# 3) DÉTECTION DES VALEURS MANQUANTES ET ABERRANTES
missing_values = df.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(missing_values)

# boxplots 
for col in df.columns:
    plt.figure(figsize=(6, 4))  
    sns.boxplot(x=df[col])     
    plt.title(f"Boxplot de la variable '{col}'")  
    plt.show()    


# 4) NORMALISATION / STANDARDISATION ET DIVISION EN ENSEMBLES D’ENTRAÎNEMENT ET DE TEST


def get_train_test_data():
    from sklearn.preprocessing import StandardScaler

    data = fetch_california_housing(as_frame=True)
    df = data.data.copy()
    df['MedHouseVal'] = data.target

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


print("\n=== FIN DE LA MISSION 1 : Exploration et Préparation des données ===")
