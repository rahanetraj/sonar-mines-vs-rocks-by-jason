"""
Script d'entraînement du modèle Sonar (Mines vs Rochers).
Charge les données, prétraite, entraîne un RandomForest et logue via MLflow.
"""

import os
import sys
import json
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

# --- Chargement des paramètres depuis params.yaml ---
ROOT = Path(__file__).resolve().parent.parent

def load_params() -> dict:
    with open(ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(data_path: Path) -> tuple:
    """Charge le CSV et retourne X, y ainsi que le LabelEncoder."""
    feature_names = [f"F{i}" for i in range(1, 61)]
    col_names = feature_names + ["Label"]

    df = pd.read_csv(data_path, header=None, names=col_names)

    # Encodage : R=0, M=1
    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Label"])

    X = df[feature_names].values
    y = df["Target"].values
    return X, y, le, feature_names

def prepare(params: dict) -> tuple:
    """Étape PREPARE : split + scaler. Sauvegarde scaler.pkl."""
    data_path = ROOT / params["data"]["path"]
    X, y, le, feature_names = load_data(data_path)

    # Découpage stratifié 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
        stratify=y,
    )

    # Normalisation (fit sur train uniquement)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Sauvegarde du scaler
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[PREPARE] Scaler sauvegardé — Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, le

def train(params: dict) -> None:
    """Étape TRAIN : entraîne le RandomForest et logue dans MLflow."""
    X_train_sc, X_test_sc, y_train, y_test, le = prepare(params)

    # Paramètres du modèle
    n_estimators = params["model"]["n_estimators"]
    max_depth    = params["model"]["max_depth"]   # None si null dans YAML
    random_state = params["model"]["random_state"]
    test_size    = params["model"]["test_size"]
    exp_name     = params["mlflow"]["experiment_name"]
    tracking_uri = str(ROOT / params["mlflow"]["tracking_uri"].lstrip("./"))

    # Configuration MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="random_forest_baseline"):
        # Log des paramètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        # Entraînement
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train_sc, y_train)

        # Évaluation sur le jeu de test
        y_pred      = clf.predict(X_test_sc)
        y_pred_prob = clf.predict_proba(X_test_sc)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # Log des métriques
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Sauvegarde du modèle dans MLflow
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # Sauvegarde locale du modèle
        models_dir = ROOT / "models"
        with open(models_dir / "model.pkl", "wb") as f:
            pickle.dump(clf, f)

        # Sauvegarde des métriques DVC
        metrics_dir = ROOT / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        with open(metrics_dir / "train_metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc}, f, indent=2)

        print(f"[TRAIN] Accuracy : {acc:.4f} | F1 : {f1:.4f} | AUC-ROC : {roc_auc:.4f}")
        print(f"[TRAIN] Modèle sauvegardé dans models/model.pkl")
        print(f"[TRAIN] Run MLflow : {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement Sonar")
    parser.add_argument(
        "--stage",
        choices=["prepare", "train"],
        default="train",
        help="Étape DVC à exécuter (prepare ou train)",
    )
    args = parser.parse_args()
    params = load_params()

    if args.stage == "prepare":
        prepare(params)
    else:
        train(params)
