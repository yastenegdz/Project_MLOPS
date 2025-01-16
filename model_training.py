import mlflow
import matplotlib.pyplot as plt
import mlflow.sklearn   
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from projet import get_train_test_data
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import joblib
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment ("model_training")

def evaluate_model(model, X_test, y_test):
    """
    Calcule les métriques RMSE, MAE et R2.
    """
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    return rmse, mae, r2


def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    """
    - Lance un 'run' MLflow.
    - Fait de la cross-validation sur X_train, y_train.
    - Entraîne le modèle complet sur X_train.
    - Évalue sur X_test.
    - Log les métriques (CV + test) et le modèle dans MLflow.
    """
    import mlflow
    import mlflow.sklearn

    with mlflow.start_run(run_name=model_name):
        # 1) Cross-validation sur le TRAIN
        # --------------------------------
        # Cross-val sur la MSE
        cv_mse_scores = cross_val_score(
            model, X_train, y_train,
            scoring='neg_mean_squared_error',
            cv=5
        )
        # Moyenne de MSE (à rendre positive)
        cv_mse = -cv_mse_scores.mean()
        cv_rmse = cv_mse ** 0.5

        # Cross-val sur la MAE
        cv_mae_scores = cross_val_score(
            model, X_train, y_train,
            scoring='neg_mean_absolute_error',
            cv=5
        )
        cv_mae = -cv_mae_scores.mean()

        # Cross-val sur R²
        cv_r2_scores = cross_val_score(
            model, X_train, y_train,
            scoring='r2',
            cv=5
        )
        cv_r2 = cv_r2_scores.mean()

        # 2) Entraînement final sur X_train
        # ---------------------------------
        model.fit(X_train, y_train)

        # 3) Évaluation finale sur X_test
        # -------------------------------
        preds_test = model.predict(X_test)
        test_rmse = mean_squared_error(y_test, preds_test, squared=False)
        test_mae  = mean_absolute_error(y_test, preds_test)
        test_r2   = r2_score(y_test, preds_test)

        # 4) Affichage console
        # --------------------
        print(f"\n=== {model_name} ===")
        print(f"Cross-Val RMSE (moyen) : {cv_rmse:.3f}")
        print(f"Cross-Val MAE  (moyen) : {cv_mae:.3f}")
        print(f"Cross-Val R²   (moyen) : {cv_r2:.3f}")
        print(f"Test RMSE              : {test_rmse:.3f}")
        print(f"Test MAE               : {test_mae:.3f}")
        print(f"Test R²                : {test_r2:.3f}")

        # 5) Logging MLflow
        # -----------------
        # Paramètre : le nom du modèle
        mlflow.log_param("model_name", model_name)

        # Metrics cross-val
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_mae",  cv_mae)
        mlflow.log_metric("cv_r2",   cv_r2)

        # Metrics test
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae",  test_mae)
        mlflow.log_metric("test_r2",   test_r2)

        # Log du modèle lui-même
        mlflow.sklearn.log_model(model, artifact_path="model")

    # Fin de la run
    return (model, cv_rmse, cv_mae, cv_r2, test_rmse, test_mae, test_r2)


def analyze_feature_importances(model, feature_names):
    """
    Affiche l'importance globale de chaque feature à l'aide de l'attribut 'feature_importances_'.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]  # tri décroissant

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[sorted_idx], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title("Importance des features")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 0) Charger les données
    X_train, X_test, y_train, y_test = get_train_test_data()

    # Liste pour stocker (nom_modèle, rmse, mae, r2)
    results = []

    # 1) Linear Regression
    lin_reg = LinearRegression()
    (
        model,cv_rmse_lr, cv_mae_lr, cv_r2_lr, test_rmse_lr, test_mae_lr, test_r2_lr
    )= train_and_log_model(
        model_name="LinearRegression_Baseline",
        model=lin_reg,
        X_train=X_train, 
        y_train=y_train,
        X_test=X_test, 
        y_test=y_test
    )
    # On stocke le résultat
    results.append(("LinearRegression_Baseline", model,cv_rmse_lr, cv_mae_lr, cv_r2_lr, test_rmse_lr, test_mae_lr, test_r2_lr))

    # 2) Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    (
        model,cv_rmse_rf, cv_mae_rf, cv_r2_rf, test_rmse_rf, test_mae_rf, test_r2_rf 
    )= train_and_log_model(
        model_name="RandomForest",
        model=rf,
        X_train=X_train, 
        y_train=y_train,
        X_test=X_test, 
        y_test=y_test
    )
    results.append(("RandomForest", model,cv_rmse_rf, cv_mae_rf, cv_r2_rf, test_rmse_rf, test_mae_rf, test_r2_rf))

    # 3) KNN Regressor
    knn_reg = KNeighborsRegressor(n_neighbors=5)
    (
        model,cv_rmse_knn, cv_mae_knn, cv_r2_knn, test_rmse_knn, test_mae_knn, test_r2_knn 
    )=train_and_log_model(
        model_name="KNNRegressor",
        model=knn_reg,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    results.append(("KNNRegressor", model,cv_rmse_knn, cv_mae_knn, cv_r2_knn, test_rmse_knn, test_mae_knn, test_r2_knn))

    # 4) SVR
    svr = SVR()
    (
        model,cv_rmse_svr, cv_mae_svr, cv_r2_svr, test_rmse_svr, test_mae_svr, test_r2_svr 
    )= train_and_log_model(
        model_name="SVR",
        model=svr,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    results.append(("SVR",model, cv_rmse_svr, cv_mae_svr, cv_r2_svr, test_rmse_svr, test_mae_svr, test_r2_svr ))



    # 5) Affichage comparatif
    print("\n=== Résultats comparatifs ===")
    for model_name, model,cv_rmse, cv_mae, cv_r2, test_rmse, test_mae, test_r2 in results:
        print(f"\n{model_name} => CV_RMSE: {cv_rmse:.3f}, CV_MAE: {cv_mae:.3f}, CV_R²: {cv_r2:.3f}, Test_RMSE: {test_rmse:.3f}, Test_MAE: {test_mae:.3f}, Test_R²: {test_r2:.3f}")

    # 6) Sélection du meilleur modèle en fonction d’un critère (ex. RMSE)
    results_sorted = sorted(results, key=lambda x: x[5])  # Trie par Test_RMSE (x[5])
    best_model_name, best_model, best_cv_rmse, best_cv_mae, best_cv_r2, best_test_rmse, best_test_mae, best_test_r2 = results_sorted[0]
    print(f"\n>>> Meilleur modèle sur la base du plus petit RMSE : {best_model_name} , RMSE={best_test_rmse:.3f}, MAE={best_test_mae:.3f}, R²={best_test_r2:.3f})")

    raw_data = fetch_california_housing(as_frame=True)
    feature_names = raw_data.feature_names  # ["MedInc", "HouseAge", "AveRooms", ... etc.]


    # Après avoir trouvé que rf_best est le meilleur modèle
    joblib.dump(best_model, "best_model.pkl")


    # 7) Analyser l'importance des features
    analyze_feature_importances(best_model, feature_names)