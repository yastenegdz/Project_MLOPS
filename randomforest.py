import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Définir les hyperparamètres
n_estimators = 100
max_depth = 5

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

mlflow.set_experiment ("randomforest")

# Suivi de l'expérience
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Enregistrer les métriques et hyperparamètres
    accuracy = clf.score(X_test, y_test)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Enregistrer le modèle
    mlflow.sklearn.log_model(clf, "model")