# Microcrédito on-line — Métricas → Decisão de Negócio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oness24/microcredito-ml-metricas/blob/main/Analise_Microcredito.ipynb)

Atividade de Machine Learning (PUCPR): análise de **classificação de inadimplência** (desbalanceada)
e **regressão de preço de revenda de veículos**, conectando cada métrica a uma decisão de negócio.

## Entregável principal
- **`Analise_Microcredito.ipynb`** — notebook Jupyter completo (PT-BR) com sumário executivo, código,
  tabelas, gráficos, cálculo de custo esperado/limiar (Parte A), comparação MAE×RMSE e decisão (Parte B),
  e propostas de melhoria priorizadas.
- **`Analise_Microcredito.html`** — versão exportada (abre em qualquer navegador, sem instalar nada).

## Dados
- `dados_credito.csv`  — classificação (`target_default`, ~17% positivos). *(cópia de `d1788ef9-...csv`)*
- `dados_veiculos.csv` — regressão (`preco_real`).                         *(cópia de `5fa876eb-...csv`)*

## Como rodar no Google Colab (recomendado — zero instalação)
1. Abra [colab.research.google.com](https://colab.research.google.com) → **Arquivo ▸ Fazer upload de notebook** → envie `Analise_Microcredito.ipynb`.
2. **Ambiente de execução ▸ Executar tudo** (*Runtime ▸ Run all*).

Não precisa enviar os CSVs nem instalar bibliotecas: os dados estão **embutidos** no notebook
(gzip+base64) e são gravados em disco na 1ª célula; o Colab já traz pandas/sklearn/matplotlib.

## Como reproduzir localmente
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter nbconvert
jupyter notebook Analise_Microcredito.ipynb      # ou: jupyter lab
```
Para regenerar o notebook do zero a partir do gerador e reexecutá-lo:
```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Analise_Microcredito.ipynb
```

## Principais resultados
| Frente | Resultado |
|---|---|
| A · Classificação | Acurácia "trivial" = 83% (engana). Limiar de menor custo cai p/ **0,11**, calotes aprovados **20→7**, **custo esperado −21%**. Monitorar por **PR-AUC + F1**. |
| B · Regressão | Linear: MAE≈R$16,5k / RMSE≈R$20k. **Gradient Boosting**: MAE −78%, RMSE −70%. Contrato em **MAE**, RMSE de sentinela + revisão manual de alto valor. |

> `build_notebook.py` é apenas o *gerador* do notebook (scaffolding). Toda a análise/código também
> está nas células do próprio `.ipynb`, que serve como anexo de código.
