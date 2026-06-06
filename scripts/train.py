"""Pipeline de treino end-to-end.

Treina os modelos de inadimplência e de preço, otimiza o limiar de decisão pelo custo de negócio,
salva os artefatos (modelos e figuras) e grava um resumo em reports/metrics.json.

Uso:
    python -m scripts.train
    python -m scripts.train --cost-fn 10 --cost-fp 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from src.classification import train_default_model
from src.regression import train_pricing_model
from src.metrics import (CostMatrix, classification_metrics, expected_cost,
                         optimal_threshold, regression_metrics)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


def run(cost_fn: float, cost_fp: float, seed: int) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    costs = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)

    # --- Classificação ---------------------------------------------------------
    credit = train_default_model(seed=seed)
    y_te, scores = credit.y_test, credit.scores

    sweep = optimal_threshold(y_te, scores, costs)
    cost_050 = expected_cost(y_te, scores, 0.50, costs)
    m_050 = classification_metrics(y_te, scores, 0.50)
    m_opt = classification_metrics(y_te, scores, sweep.threshold)

    joblib.dump(credit.pipeline, MODELS_DIR / "credit_default.joblib")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep.grid, sweep.costs, color="#2563eb", lw=2)
    ax.axvline(sweep.threshold, color="#dc2626", ls="--",
               label=f"otimo = {sweep.threshold:.2f}")
    ax.axvline(0.50, color="#6b7280", ls=":", label="padrao 0,50")
    ax.set_xlabel("limiar"); ax.set_ylabel("custo esperado")
    ax.set_title("Custo esperado x limiar"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "custo_vs_limiar.png", dpi=120)
    plt.close(fig)

    # --- Regressão -------------------------------------------------------------
    linear = train_pricing_model(kind="linear", seed=seed)
    gbr = train_pricing_model(kind="gbr", seed=seed)
    reg_lin = regression_metrics(linear.y_test, linear.predictions)
    reg_gbr = regression_metrics(gbr.y_test, gbr.predictions)

    joblib.dump(gbr.model, MODELS_DIR / "vehicle_pricing_gbr.joblib")

    summary = {
        "classification": {
            "cost_matrix": {"cost_fn": cost_fn, "cost_fp": cost_fp,
                            "bayes_threshold": round(costs.bayes_threshold, 4)},
            "threshold_050": {"cost": cost_050, **{k: round(v, 4) for k, v in m_050.items()}},
            "threshold_optimal": {"threshold": round(sweep.threshold, 4),
                                  "cost": sweep.cost,
                                  **{k: round(v, 4) for k, v in m_opt.items()}},
            "cost_reduction": round(1 - sweep.cost / cost_050, 4),
        },
        "regression": {
            "linear": {k: round(v, 2) for k, v in reg_lin.items()},
            "gradient_boosting": {k: round(v, 2) for k, v in reg_gbr.items()},
            "mae_improvement": round(1 - reg_gbr["mae"] / reg_lin["mae"], 4),
            "rmse_improvement": round(1 - reg_gbr["rmse"] / reg_lin["rmse"], 4),
        },
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina os modelos e gera métricas.")
    parser.add_argument("--cost-fn", type=float, default=10.0,
                        help="custo de aprovar um inadimplente (default: 10)")
    parser.add_argument("--cost-fp", type=float, default=2.0,
                        help="custo de negar um bom pagador (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    s = run(args.cost_fn, args.cost_fp, args.seed)
    clf, reg = s["classification"], s["regression"]
    print("== Classificacao ==")
    print(f"  limiar otimo: {clf['threshold_optimal']['threshold']} "
          f"(custo {clf['threshold_optimal']['cost']} vs {clf['threshold_050']['cost']} em 0,50; "
          f"reducao {clf['cost_reduction']:.0%})")
    print(f"  PR-AUC: {clf['threshold_optimal']['pr_auc']} | "
          f"F1@otimo: {clf['threshold_optimal']['f1']}")
    print("== Regressao ==")
    print(f"  linear  -> MAE {reg['linear']['mae']:.0f} | RMSE {reg['linear']['rmse']:.0f}")
    print(f"  GBR     -> MAE {reg['gradient_boosting']['mae']:.0f} | "
          f"RMSE {reg['gradient_boosting']['rmse']:.0f}")
    print(f"  ganho   -> MAE {reg['mae_improvement']:.0%} | RMSE {reg['rmse_improvement']:.0%}")
    print("\nArtefatos: models/*.joblib | reports/figures/*.png | reports/metrics.json")


if __name__ == "__main__":
    main()
