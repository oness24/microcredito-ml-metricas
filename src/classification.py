"""Modelo de classificação de inadimplência (default)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET_DEFAULT, load_credit


@dataclass
class CreditModel:
    """Modelo treinado e o respectivo conjunto de teste."""

    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    scores: np.ndarray  # P(inadimplência) no conjunto de teste


def train_default_model(df: pd.DataFrame | None = None, *,
                        class_weight: str | dict | None = None,
                        test_size: float = 0.30, seed: int = 42) -> CreditModel:
    """Treina uma regressão logística padronizada para prever inadimplência.

    Args:
        df: dataset; se None, carrega `data/credito.csv`.
        class_weight: passado à `LogisticRegression` (ex.: 'balanced').
        test_size: fração reservada para teste.
        seed: semente para reprodutibilidade.
    """
    if df is None:
        df = load_credit()
    X = df.drop(columns=TARGET_DEFAULT)
    y = df[TARGET_DEFAULT]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight,
                                   random_state=seed)),
    ])
    pipeline.fit(X_tr, y_tr)
    scores = pipeline.predict_proba(X_te)[:, 1]
    return CreditModel(pipeline, X_te, y_te, scores)
