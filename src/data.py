"""Carga e preparação dos datasets de crédito e veículos."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET_DEFAULT = "target_default"
TARGET_PRICE = "preco_real"


def load_credit(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega o dataset de inadimplência."""
    path = Path(path) if path else DATA_DIR / "credito.csv"
    return pd.read_csv(path)


def load_vehicles(path: str | Path | None = None) -> pd.DataFrame:
    """Carrega o dataset de preço de revenda de veículos."""
    path = Path(path) if path else DATA_DIR / "veiculos.csv"
    return pd.read_csv(path)


def vehicle_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Codifica `marca` em variáveis dummy e separa X (features) e y (preço)."""
    encoded = pd.get_dummies(df, columns=["marca"], drop_first=True)
    y = encoded[TARGET_PRICE]
    X = encoded.drop(columns=[TARGET_PRICE])
    return X, y
