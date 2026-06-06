"""Métricas e lógica de custo de negócio.

A classe positiva é a inadimplência (`target_default = 1`). Os custos são definidos pelo evento
de negócio para evitar ambiguidade sobre o que é falso positivo / falso negativo:

    FN (Real 1, Prev. 0) = aprovar um inadimplente  -> perde o principal
    FP (Real 0, Prev. 1) = negar um bom pagador      -> perde a margem
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)


@dataclass(frozen=True)
class CostMatrix:
    """Custos de negócio dos erros de classificação.

    Args:
        cost_fn: custo de aprovar um inadimplente (perda do principal).
        cost_fp: custo de negar um bom pagador (perda da margem).
    """

    cost_fn: float = 10.0
    cost_fp: float = 2.0

    def total(self, y_true, y_pred) -> float:
        """Custo total dado um vetor de predições binárias."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return self.cost_fn * fn + self.cost_fp * fp

    @property
    def bayes_threshold(self) -> float:
        """Limiar ótimo teórico assumindo probabilidades bem calibradas."""
        return self.cost_fp / (self.cost_fp + self.cost_fn)


def expected_cost(y_true, y_score, threshold: float, costs: CostMatrix) -> float:
    """Custo esperado ao aplicar um limiar sobre os scores."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return costs.total(y_true, y_pred)


@dataclass
class ThresholdSweep:
    """Resultado da varredura de limiares."""

    threshold: float
    cost: float
    grid: np.ndarray
    costs: np.ndarray


def optimal_threshold(y_true, y_score, costs: CostMatrix,
                      grid: np.ndarray | None = None) -> ThresholdSweep:
    """Varre limiares e retorna o de menor custo esperado."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    values = np.array([expected_cost(y_true, y_score, t, costs) for t in grid])
    best = int(values.argmin())
    return ThresholdSweep(float(grid[best]), float(values[best]), grid, values)


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Painel de métricas de classificação para um dado limiar."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE e a razão RMSE/MAE (assinatura de erros grandes)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"mae": mae, "rmse": rmse, "rmse_mae_ratio": rmse / mae}
