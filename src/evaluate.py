"""
Script d'évaluation du modèle Sonar.
Charge le modèle et le scaler, génère le rapport de classification
et la matrice de confusion, logue les métriques dans MLflow.
"""

import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

import mlflow

ROOT = Path(__file__).resolve().parent.parent

def load_params() -> dict:
    with open(ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_artifacts() -> tuple:
    """Charge le modèle et le scaler depuis models/."""
    models_dir = ROOT / "models"

    with open(models_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

def load_data(params: dict) -> tuple:
    """Recharge et resplit les données avec les mêmes paramètres."""
    data_path = ROOT / params["data"]["path"]
    feature_names = [f"F{i}" for i in range(1, 61)]
    col_names = feature_names + ["Label"]

    df = pd.read_csv(data_path, header=None, names=col_names)

    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Label"])

    X = df[feature_names].values
    y = df["Target"].values

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
        stratify=y,
    )
    return X_test, y_test, le

def save_confusion_matrix(cm: np.ndarray, labels: list, output_path: Path) -> None:
    """Sauvegarde la matrice de confusion en PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=1, ax=ax,
    )
    ax.set_title("Matrice de Confusion — Sonar (Mines vs Rochers)", fontweight="bold")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATE] Matrice de confusion sauvegardée : {output_path}")

def main() -> None:
    params = load_params()
    model, scaler = load_artifacts()
    X_test, y_test, le = load_data(params)

    # Normalisation du jeu de test
    X_test_sc = scaler.transform(X_test)

    # Prédictions
    y_pred      = model.predict(X_test_sc)
    y_pred_prob = model.predict_proba(X_test_sc)[:, 1]

    # Métriques
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Rapport de classification
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\n[EVALUATE] Rapport de classification :")
    print(report)

    # Matrice de confusion → PNG
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, list(le.classes_), ROOT / "outputs" / "confusion_matrix.png")

    # Sauvegarde des métriques DVC
    metrics_dir = ROOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics = {"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc}
    with open(metrics_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Log MLflow (attache à la dernière run active si possible)
    tracking_uri = str(ROOT / params["mlflow"]["tracking_uri"].lstrip("./"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(ROOT / "outputs" / "confusion_matrix.png"))

    # Résumé final
    print("\n" + "=" * 50)
    print("       RÉSUMÉ ÉVALUATION — MODÈLE SONAR")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-score  : {f1:.4f}  (macro)")
    print(f"  AUC-ROC   : {roc_auc:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
