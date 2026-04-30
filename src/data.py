from __future__ import annotations
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation."""
    
    data_path = DATA_DIR / "rocket_league_skill_data.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test