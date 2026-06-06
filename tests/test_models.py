"""Testes de fumaça dos modelos: treinam e produzem saídas sãs."""
from src.classification import train_default_model
from src.regression import train_pricing_model
from src.metrics import regression_metrics


def test_default_model_trains_and_scores_in_unit_interval():
    model = train_default_model(seed=42)
    assert len(model.scores) == len(model.y_test)
    assert model.scores.min() >= 0.0 and model.scores.max() <= 1.0


def test_gbr_beats_linear_on_mae():
    linear = train_pricing_model(kind="linear", seed=42)
    gbr = train_pricing_model(kind="gbr", seed=42)
    mae_lin = regression_metrics(linear.y_test, linear.predictions)["mae"]
    mae_gbr = regression_metrics(gbr.y_test, gbr.predictions)["mae"]
    assert mae_gbr < mae_lin
