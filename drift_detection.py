import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import RegressionPreset
from evidently.options import ColumnMapping



# Exemple : on part du DataFrame d’entraînement 
# On crée df_prod en modifiant un peu les distributions

X_train = pd.read_csv("X_train.csv")
df_prod = X_train.copy()

# On crée artificiellement du "drift" sur une feature 
# par ex. un décalage de +1 sur MedInc et *1.2 sur HouseAge
df_prod["MedInc"] = df_prod["MedInc"] + 1.0
df_prod["HouseAge"] = df_prod["HouseAge"] * 1.2

data_drift_report = Report(
    metrics=[DataDriftPreset()])

data_drift_report.run(
    reference_data=X_train,
    current_data=df_prod,
    column_mapping=None  
)

# 4) Générer le rapport
data_drift_report.save_html("data_drift_report.html")
print("Data drift report généré : data_drift_report.html")