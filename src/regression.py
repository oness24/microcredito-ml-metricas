"""Modelo de regressão do preço de revenda de veículos."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .data import load_vehicles, vehicle_feature_matrix

_ESTIMATORS = {"linear": LinearRegression, "gbr": GradientBoostingRegressor}


@dataclass
class PricingModel:
    """Modelo treinado, conjunto de teste e predições."""

    model: object
    kind: str
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: np.ndarray


def train_pricing_model(df: pd.DataFrame | None = None, *, kind: str = "gbr",
                        test_size: float = 0.30, seed: int = 42) -> PricingModel:
    """Treina um modelo de preço.

    Args:
        df: dataset; se None, carrega `data/veiculos.csv`.
        kind: 'linear' (baseline) ou 'gbr' (Gradient Boosting).
        test_size: fração reservada para teste.
        seed: semente para reprodutibilidade.
    """
    if kind not in _ESTIMATORS:
        raise ValueError(f"kind deve ser um de {list(_ESTIMATORS)}, recebido {kind!r}")
    if df is None:
        df = load_vehicles()
    X, y = vehicle_feature_matrix(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    estimator = _ESTIMATORS[kind]
    model = estimator(random_state=seed) if kind == "gbr" else estimator()
    model.fit(X_tr, y_tr)
    return PricingModel(model, kind, X_te, y_te, model.predict(X_te))
