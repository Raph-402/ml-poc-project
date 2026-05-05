# 🏎️ Rocket League Smurf Detection : Proof of Concept

## 📖 Contexte Métier
Dans Rocket League, le "Smurfing" (joueurs d'élite utilisant des comptes de bas niveau) détruit l'expérience utilisateur et impacte la rétention. Ce projet propose une solution de Machine Learning comportementale : analyser l'ADN mécanique des joueurs (vitesse, gestion du boost, temps en l'air) pour détecter les anomalies et identifier les Smurfs, sans se baser uniquement sur le score.

## 🏗️ Architecture du Projet
Ce PoC respecte une architecture MLOps stricte :
- **`data/`** : Fichiers CSV contenant les données des matchs (Features mécaniques).
- **`notebooks/`** : Le "Lab". Démarche exploratoire, Feature Engineering et Sélection de modèles (Logistic Regression, Random Forest, Gradient Boosting).
- **`models/`** : Modèles entraînés et exportés au format `.joblib`.
- **`src/`** : Code métier modulaire (`data.py`, `metrics.py`, `app.py`).
- **`scripts/`** : Orchestration. Scripts d'entraînement (`train.py`) et de pipeline complet (`main.py`).

## 🚀 Comment lancer le projet ?

**1. Installation des dépendances :**
```bash
pip install -r requirements.txt