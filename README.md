# Microcrédito: métricas de modelo e decisão de negócio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oness24/microcredito-ml-metricas/blob/main/notebooks/Analise_Microcredito.ipynb)

Estudo de caso de uma fintech de microcrédito on-line, em duas frentes:

- **Classificação de inadimplência** (base desbalanceada). O modelo reporta ~95% de acurácia, mas
  o negócio sofre com falsos positivos e negativos. A análise mostra por que a acurácia engana e
  define um limiar de decisão a partir de uma matriz de custos.
- **Regressão de preço de revenda de veículos**, usada para calcular limite de crédito. A análise
  compara MAE e RMSE e recomenda qual métrica usar no contrato (SLA).

O foco é ligar cada métrica a uma decisão e ao seu impacto financeiro, não apenas treinar modelos.

## Resultados

| Frente | Principais números |
|---|---|
| Classificação | Acurácia de um classificador trivial: 83% (engana). Matriz de custos FN=10 (aprovar inadimplente, perde o principal ~R$ 20 mil) vs FP=2 (negar bom pagador, perde a margem). Limiar de menor custo ≈ 0,11: inadimplentes aprovados caem de 20 para 7; custo esperado −21%. Monitoramento por PR-AUC e F1. |
| Regressão | Baseline linear: MAE ≈ R$ 16,5 mil, RMSE ≈ R$ 20 mil. Gradient Boosting: MAE −78%, RMSE −70%. Contrato em MAE, RMSE como sentinela e revisão manual de alto valor. |

Os números são reproduzidos por `python -m scripts.train` (gera `reports/metrics.json`).

## Estrutura

```
.
├── data/                 datasets (credito.csv, veiculos.csv)
├── notebooks/            Analise_Microcredito.ipynb  (relatório executável, Colab-ready)
├── src/                  biblioteca: data, metrics, classification, regression
├── scripts/train.py      pipeline de treino (CLI) -> modelos, figuras, metrics.json
├── tests/                testes (pytest) da lógica de custo e dos modelos
├── reports/              HTML do notebook, figuras e métricas geradas
├── docs/                 enunciado da atividade
├── build_notebook.py     gerador do notebook a partir de data/
├── requirements.txt · Makefile · LICENSE
└── .github/workflows/ci.yml
```

## Como rodar

### Google Colab (sem instalação)
Abra o badge acima e use *Ambiente de execução > Executar tudo*. Os dados estão embutidos no
notebook (gzip+base64), então não é preciso enviar CSVs nem instalar pacotes.

### Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make test       # roda os testes
make train      # treina os modelos e grava reports/metrics.json
make html       # regenera e executa o notebook e exporta o HTML
```

## Decisões de modelagem

- **Custo definido pelo evento de negócio**, não pela sigla FP/FN, para evitar ambiguidade
  (positivo = inadimplente; aprovar um inadimplente é um falso negativo).
- **Limiar de decisão** escolhido por menor custo esperado, não fixado em 0,50. O limiar ótimo
  teórico (Bayes) é `cost_fp / (cost_fp + cost_fn)`.
- **MAE como métrica de contrato** na regressão por ser interpretável e robusto; RMSE acompanha
  como alarme de erros graves, já que poucos carros premium dominam o erro quadrático.

## Reprodutibilidade
Semente fixa (`seed=42`) em treino e split. CI (GitHub Actions) instala as dependências, roda os
testes e o pipeline de treino a cada push.
