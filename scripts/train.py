"""Script d'entraînement et de sauvegarde des modèles de détection de Smurfs."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("SMURF_DATA_PATH", PROJECT_ROOT / "data" / "rocket_league_skill_data.csv"))
MODELS_DIR = PROJECT_ROOT / "models"

def main() -> None:
    print("⏳ Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("⚙️ Création des pipelines...")
    log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    gb = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, random_state=42))

    print("🧠 Entraînement des modèles...")
    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    print("💾 Sauvegarde des modèles...")
    MODELS_DIR.mkdir(exist_ok=True) 
    
    joblib.dump(log_reg, MODELS_DIR / "log_reg.joblib")
    joblib.dump(rf, MODELS_DIR / "rf.joblib")
    joblib.dump(gb, MODELS_DIR / "gradient_boosting.joblib")

    print("✅ Terminé ! Les 3 modèles sont dans le dossier models/.")

if __name__ == "__main__":
    main()