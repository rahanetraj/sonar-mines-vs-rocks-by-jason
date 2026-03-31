"""
Détection de data drift avec Evidently.
Compare les données de référence (train) avec les données courantes (prédictions récentes).
Génère un rapport HTML dans monitoring/reports/drift_report.html
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

REPORTS_DIR = ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [f"F{i}" for i in range(1, 61)]

def load_reference_data(params_path: Path = None) -> pd.DataFrame:
    """Charge les données d'entraînement comme référence."""
    import yaml
    params_path = params_path or ROOT / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    data_path = ROOT / params["data"]["path"]
    col_names = FEATURE_NAMES + ["Label"]
    df = pd.read_csv(data_path, header=None, names=col_names)

    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Label"])
    X = df[FEATURE_NAMES]
    y = df["Target"]

    X_train, _, _, _ = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
        stratify=y,
    )
    return X_train.reset_index(drop=True)


def load_current_data(predictions_log: Path = None) -> pd.DataFrame:
    """
    Charge les données courantes depuis un CSV de logs de prédictions.
    Si le fichier n'existe pas, simule avec du bruit sur les données de référence.
    """
    predictions_log = predictions_log or ROOT / "monitoring" / "predictions_log.csv"

    if predictions_log.exists():
        df = pd.read_csv(predictions_log)
        # Garde uniquement les colonnes features
        available = [c for c in FEATURE_NAMES if c in df.columns]
        if len(available) == 60:
            print(f"[DRIFT] Données courantes chargées : {len(df)} échantillons")
            return df[FEATURE_NAMES].reset_index(drop=True)

    print("[DRIFT] Fichier predictions_log.csv introuvable — simulation de drift artificiel.")
    ref = load_reference_data()
    # Simulation : ajout de bruit gaussien pour créer un drift détectable
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=0.05, scale=0.08, size=ref.shape)
    current = (ref.values + noise).clip(0.0, 1.0)
    return pd.DataFrame(current, columns=FEATURE_NAMES)


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path = None,
) -> dict:
    """
    Génère le rapport de drift Evidently et retourne un résumé JSON.
    """
    output_path = output_path or REPORTS_DIR / "drift_report.html"

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))

    # Extraction du résumé de drift
    result = report.as_dict()
    dataset_drift_result = None
    drifted_features = []

    for metric in result.get("metrics", []):
        metric_id = metric.get("metric", "")

        if "DatasetDriftMetric" in metric_id:
            res = metric.get("result", {})
            dataset_drift_result = {
                "drift_detected" : res.get("dataset_drift", False),
                "drift_share"    : res.get("drift_share", 0.0),
                "n_features"     : res.get("number_of_columns", 60),
                "n_drifted"      : res.get("number_of_drifted_columns", 0),
            }

        if "ColumnDriftMetric" in metric_id:
            res = metric.get("result", {})
            if res.get("drift_detected"):
                drifted_features.append(res.get("column_name", "?"))

    return {
        "dataset": dataset_drift_result,
        "drifted_features": drifted_features,
        "report_path": str(output_path),
    }


def print_summary(summary: dict) -> None:
    """Affiche un résumé lisible du drift détecté."""
    ds = summary.get("dataset") or {}
    print("\n" + "=" * 55)
    print("       RÉSUMÉ — RAPPORT DE DATA DRIFT (Evidently)")
    print("=" * 55)
    drift_detected = ds.get("drift_detected", "N/A")
    print(f"  Drift global détecté : {drift_detected}")
    print(f"  Part de features driftées : {ds.get('drift_share', 0)*100:.1f}%")
    print(f"  Features driftées : {ds.get('n_drifted', '?')} / {ds.get('n_features', 60)}")

    if summary["drifted_features"]:
        print(f"  Liste : {', '.join(summary['drifted_features'][:10])}")
        if len(summary["drifted_features"]) > 10:
            print(f"          ... et {len(summary['drifted_features']) - 10} autres")

    print(f"\n  Rapport HTML : {summary['report_path']}")
    print("=" * 55)


if __name__ == "__main__":
    print("[DRIFT] Chargement des données de référence (train)...")
    reference = load_reference_data()
    print(f"[DRIFT] Référence : {reference.shape}")

    print("[DRIFT] Chargement des données courantes...")
    current = load_current_data()
    print(f"[DRIFT] Courant   : {current.shape}")

    print("[DRIFT] Génération du rapport Evidently...")
    summary = generate_drift_report(reference, current)

    print_summary(summary)

    # Sauvegarde du résumé JSON
    summary_path = REPORTS_DIR / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[DRIFT] Résumé JSON : {summary_path}")
