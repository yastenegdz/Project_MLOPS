#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:28:06 2024

@author: yastenegdz
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
model=mlflow.sklearn.load_model('runs:/570c012525804343a3ad801ce89d53b5/model')
predictions= model.predict(X_test)
print(predictions.shape)
