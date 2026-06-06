"""Testes da lógica de custo e métricas."""
import numpy as np

from src.metrics import (CostMatrix, expected_cost, optimal_threshold,
                         classification_metrics, regression_metrics)


def test_cost_matrix_total_counts_fn_and_fp():
    # Predições: 1 FN (real 1, prev 0) e 1 FP (real 0, prev 1).
    y_true = [1, 0, 1, 0]
    y_pred = [0, 1, 1, 0]
    costs = CostMatrix(cost_fn=10, cost_fp=2)
    assert costs.total(y_true, y_pred) == 10 + 2


def test_bayes_threshold():
    costs = CostMatrix(cost_fn=10, cost_fp=2)
    assert abs(costs.bayes_threshold - 2 / 12) < 1e-9


def test_perfect_prediction_has_zero_cost():
    y = [0, 1, 1, 0, 1]
    assert CostMatrix().total(y, y) == 0


def test_optimal_threshold_drops_when_fn_is_expensive():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    score = np.clip(0.35 * y + rng.normal(0, 0.2, size=100) + 0.2, 0, 1)
    cheap = optimal_threshold(y, score, CostMatrix(cost_fn=2, cost_fp=2)).threshold
    expensive = optimal_threshold(y, score, CostMatrix(cost_fn=50, cost_fp=1)).threshold
    # Quanto mais caro o FN, mais baixo o limiar (recusar mais para evitar calotes).
    assert expensive <= cheap


def test_expected_cost_matches_manual():
    y_true = [1, 1, 0, 0]
    score = [0.9, 0.4, 0.8, 0.1]
    costs = CostMatrix(cost_fn=10, cost_fp=2)
    # limiar 0.5 -> pred [1,0,1,0]; FN=1 (segundo), FP=1 (terceiro)
    assert expected_cost(y_true, score, 0.5, costs) == 12


def test_classification_metrics_keys_and_ranges():
    y = [0, 1, 0, 1, 1]
    s = [0.1, 0.9, 0.2, 0.8, 0.6]
    m = classification_metrics(y, s, 0.5)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_regression_metrics_rmse_ge_mae():
    y_true = [1, 2, 3, 4]
    y_pred = [1.5, 2.5, 2.0, 5.0]
    m = regression_metrics(y_true, y_pred)
    assert m["rmse"] >= m["mae"] >= 0
    assert m["rmse_mae_ratio"] >= 1.0
