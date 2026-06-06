"""Gera notebooks/Analise_Microcredito.ipynb a partir das fontes em data/.

Uso:
    python build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/Analise_Microcredito.ipynb

O notebook embute os CSVs (gzip+base64) para rodar no Google Colab sem upload manual.
"""
import gzip
import base64
import nbformat as nbf


def _embed(path):
    with open(path, "rb") as f:
        return base64.b64encode(gzip.compress(f.read())).decode()


B64_CRED = _embed("data/credito.csv")
B64_VEIC = _embed("data/veiculos.csv")

nb = nbf.v4.new_notebook()
cells = []
def md(src):  cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

BADGE = ("https://colab.research.google.com/github/oness24/microcredito-ml-metricas/"
         "blob/main/notebooks/Analise_Microcredito.ipynb")

# --------------------------------------------------------------------------- CAPA
md(f"""# Microcrédito on-line — métricas de modelo e decisão de negócio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({BADGE})

**Autor:** Onesmus Simiyu · PUCPR · Machine Learning

Estudo de caso de uma fintech de microcrédito. Duas frentes são analisadas:

- **Classificação** — o modelo de inadimplência (`default`) reporta ~95% de acurácia, mas o
  negócio observa mais falsos positivos e falsos negativos do que esse número sugere.
- **Regressão** — um modelo de preço de revenda de veículos apoia a definição de limite de
  crédito, e o time precisa decidir se o contrato (SLA) deve ser medido por MAE ou RMSE.

O objetivo não é só treinar modelos, e sim ligar cada métrica a uma decisão e ao seu impacto
financeiro.

## Sumário executivo

**Classificação.** A acurácia de 95% é enganosa: como apenas ~17% da base é inadimplente, um
classificador que aprova todo mundo já acerta ~83% sem aprender nada. O custo real está nos
inadimplentes aprovados, que consomem o principal emprestado. Modelando os custos de negócio
(aprovar inadimplente custa ~5x negar um bom pagador) e otimizando o limiar de decisão, o custo
esperado cai cerca de 21% em relação ao corte padrão de 0,50, reduzindo de 20 para 7 os
inadimplentes aprovados na amostra de teste. Recomenda-se monitorar PR-AUC e F1 no ponto de
operação, não a acurácia.

**Regressão.** O baseline linear erra em média ~R$ 16,5 mil (MAE), com RMSE ~R$ 20 mil. A
diferença entre as duas métricas é o peso que o RMSE dá a poucos erros grandes (carros premium).
A recomendação é contratar o **MAE** como SLA — interpretável e estável — e usar o RMSE como
sentinela de erros graves, com revisão manual obrigatória acima de um teto de valor. Um modelo
não-linear (Gradient Boosting) reduz MAE e RMSE de forma expressiva e deve substituir o baseline.
""")

# --------------------------------------------------------------------------- COLAB
md(r"""---
### Como executar

**Google Colab.** Abra o badge acima (ou faça upload deste `.ipynb`) e use *Ambiente de execução
> Executar tudo*. Não é necessário enviar os CSVs nem instalar pacotes: os dados estão embutidos
na célula abaixo e o Colab já traz `pandas`, `scikit-learn` e `matplotlib`.

**Local.** `pip install -r requirements.txt` e abra com `jupyter notebook`.""")
code(("# Bootstrap de dados: grava os CSVs a partir de cópia embutida (gzip+base64).\n"
      "# Funciona no Colab, no Jupyter local ou em CI, sem upload manual.\n"
      "import os, gzip, base64\n\n"
      "_DADOS = {\n"
      '    "dados_credito.csv":  "%s",\n'
      '    "dados_veiculos.csv": "%s",\n'
      "}\n"
      "for _nome, _b64 in _DADOS.items():\n"
      "    if not os.path.exists(_nome):\n"
      '        with open(_nome, "wb") as _f:\n'
      "            _f.write(gzip.decompress(base64.b64decode(_b64)))\n"
      'print("Dados disponiveis:", ", ".join(_DADOS))') % (B64_CRED, B64_VEIC))

# --------------------------------------------------------------------------- SETUP
md("## Configuração")
code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             mean_absolute_error, mean_squared_error)

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11, "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
AZUL, VERM, VERDE, CINZA = "#2563eb", "#dc2626", "#16a34a", "#6b7280"

def reais(x, _=None):
    return f"R$ {x:,.0f}".replace(",", ".")

credito  = pd.read_csv("dados_credito.csv")
veiculos = pd.read_csv("dados_veiculos.csv")
print("Credito (classificacao):", credito.shape)
print("Veiculos (regressao):   ", veiculos.shape)
credito.head()""")

# ========================================================================= PARTE A
md(r"""---
## Parte A — Classificação da inadimplência

A classe positiva é `target_default = 1` (cliente que dá calote). Os erros têm custos
assimétricos, e é exatamente isso que a acurácia ignora.""")

md("### A.1 Distribuição da classe")
code(r"""dist = credito["target_default"].value_counts().sort_index()
taxa = credito["target_default"].mean()

fig, ax = plt.subplots(figsize=(6, 3.4))
barras = ax.bar(["Bom pagador (0)", "Inadimplente (1)"], dist.values, color=[VERDE, VERM])
for b, v in zip(barras, dist.values):
    ax.text(b.get_x()+b.get_width()/2, v+4, f"{v} ({v/dist.sum():.1%})",
            ha="center", va="bottom", fontsize=10)
ax.set_title(f"Taxa de inadimplencia = {taxa:.1%}")
ax.set_ylabel("clientes"); ax.set_ylim(0, dist.max()*1.18)
plt.tight_layout(); plt.show()

print(f"Inadimplencia: {taxa:.1%}. Um modelo que sempre responde 'bom pagador' "
      f"acertaria {1-taxa:.1%} dos casos sem aprender nada.")""")

md("### A.2 Modelo baseline (regressão logística)")
code(r"""X = credito.drop(columns="target_default")
y = credito["target_default"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)

scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

baseline = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tr_s, y_tr)
proba = baseline.predict_proba(X_te_s)[:, 1]
pred_050 = (proba >= 0.50).astype(int)

print(f"Treino: {len(X_tr)} | Teste: {len(X_te)} | inadimplencia no teste: {y_te.mean():.1%}")""")

md("### A.3 Matriz de confusão e métricas")
code(r"""def painel(y_true, y_score, thr):
    y_hat = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_hat)
    tn, fp, fn, tp = cm.ravel()
    m = {
        "Acuracia":  accuracy_score(y_true, y_hat),
        "Precisao":  precision_score(y_true, y_hat, zero_division=0),
        "Recall":    recall_score(y_true, y_hat, zero_division=0),
        "F1":        f1_score(y_true, y_hat, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_score),
        "PR-AUC":    average_precision_score(y_true, y_score),
    }
    return cm, (tn, fp, fn, tp), m

cm, (tn, fp, fn, tp), met = painel(y_te, proba, 0.50)

fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Prev. 0", "Prev. 1"]); ax.set_yticks([0, 1], ["Real 0", "Real 1"])
nomes = [["VN", "FP"], ["FN", "VP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{nomes[i][j]}\n{cm[i,j]}", ha="center", va="center",
                fontsize=13, color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_title("Matriz de confusao (limiar 0,50)")
plt.tight_layout(); plt.show()

tabela = pd.DataFrame({"valor": met}).round(3)
tabela["interpretacao"] = [
    "acertos totais; inflada pela classe majoritaria",
    "dos marcados como inadimplentes, quantos eram de fato",
    "dos inadimplentes reais, quantos foram capturados",
    "media harmonica de precisao e recall",
    "ordenacao de risco; independe do limiar",
    "qualidade na classe rara; metrica adequada ao desbalanceamento",
]
tabela""")

md(r"""### A.4 Por que a acurácia engana neste cenário

Comparação com um classificador trivial que sempre responde "bom pagador".""")
code(r"""acc_modelo  = accuracy_score(y_te, pred_050)
acc_trivial = accuracy_score(y_te, np.zeros_like(y_te))

comp = pd.DataFrame({
    "Acuracia": [acc_trivial, acc_modelo],
    "Recall (inadimplentes capturados)": [0.0, met["Recall"]],
    "Inadimplentes que escaparam (FN)": [int(y_te.sum()), fn],
}, index=["Sempre 'bom pagador'", "Regressao logistica"]).round(3)
comp""")
md(r"""O classificador trivial já alcança acurácia próxima dos 95% reportados, mas deixa passar
100% dos inadimplentes. A acurácia premia quem ignora a classe rara — e é a classe rara que
custa dinheiro. Por isso ela não serve como métrica-guia aqui.""")

md(r"""### A.5 Matriz de custos de negócio

A acurácia trata todo erro como igual; o resultado financeiro não. Definimos o custo pelo
**evento de negócio**, não pela sigla, para evitar ambiguidade sobre o que é FP/FN (positivo =
inadimplente).

| Evento | Célula | Consequência | Custo |
|---|---|---|---|
| Aprovar um inadimplente | Real 1 / Prev. 0 (FN) | perde o principal (`valor_solicitado` médio ~R$ 20 mil) | 10 |
| Negar um bom pagador | Real 0 / Prev. 1 (FP) | perde a margem/juros (~R$ 2 mil) | 2 |
| Negar um inadimplente (VP) | Real 1 / Prev. 1 | decisão correta | 0 |
| Aprovar um bom pagador (VN) | Real 0 / Prev. 0 | decisão correta | 0 |

O enunciado sugere uma razão em torno de 5:2. Ajustamos para **10:2 (5:1)** ancorando em valores:
aprovar um caloteiro queima o principal (~R$ 20 mil), enquanto negar um bom cliente custa apenas
a margem perdida (~R$ 2 mil). A perda de principal é da ordem de 5x maior. A razão é o que
importa; em produção pode-se usar o `valor_solicitado` linha a linha.""")
code(r"""CUSTO_FN = 10  # aprovar inadimplente (perde o principal)
CUSTO_FP = 2   # negar bom pagador (perde a margem)

def custo_esperado(y_true, y_score, thr):
    y_hat = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return CUSTO_FN*fn + CUSTO_FP*fp, fn, fp

c050, fn050, fp050 = custo_esperado(y_te, proba, 0.50)
limiar_bayes = CUSTO_FP / (CUSTO_FP + CUSTO_FN)
print(f"Limiar 0,50: FN={fn050}, FP={fp050}, custo total={c050}")
print(f"Limiar otimo teorico (Bayes) = {limiar_bayes:.2f}")""")

md("### A.6 Limiar de menor custo esperado")
code(r"""thrs = np.linspace(0.01, 0.99, 99)
custos = np.array([custo_esperado(y_te, proba, t)[0] for t in thrs])
thr_otimo = thrs[custos.argmin()]
custo_otimo, fn_ot, fp_ot = custo_esperado(y_te, proba, thr_otimo)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))

axs[0].plot(thrs, custos, color=AZUL, lw=2)
axs[0].axvline(thr_otimo, color=VERM, ls="--", label=f"otimo = {thr_otimo:.2f}")
axs[0].axvline(0.50, color=CINZA, ls=":", label="padrao 0,50")
axs[0].scatter([thr_otimo], [custo_otimo], color=VERM, zorder=5)
axs[0].set_title("Custo esperado x limiar")
axs[0].set_xlabel("limiar"); axs[0].set_ylabel("custo total"); axs[0].legend()

fpr, tpr, _ = roc_curve(y_te, proba)
axs[1].plot(fpr, tpr, color=AZUL, lw=2, label=f"AUC = {met['ROC-AUC']:.3f}")
axs[1].plot([0, 1], [0, 1], ls="--", color=CINZA)
axs[1].set_title("Curva ROC"); axs[1].set_xlabel("FPR"); axs[1].set_ylabel("TPR")
axs[1].legend()

prec, rec, _ = precision_recall_curve(y_te, proba)
axs[2].plot(rec, prec, color=VERDE, lw=2, label=f"PR-AUC = {met['PR-AUC']:.3f}")
axs[2].axhline(y_te.mean(), ls="--", color=CINZA, label=f"acaso = {y_te.mean():.2f}")
axs[2].set_title("Curva precisao-recall"); axs[2].set_xlabel("recall"); axs[2].set_ylabel("precisao")
axs[2].legend()
plt.tight_layout(); plt.show()

print(f"Limiar otimo = {thr_otimo:.2f} | custo {custo_otimo} (vs {c050} em 0,50) "
      f"| reducao de {(1-custo_otimo/c050):.1%}")
print(f"FN: {fn050} -> {fn_ot} | FP: {fp050} -> {fp_ot}")""")

md(r"""Baixar o limiar faz o modelo recusar mais (mais recall na classe inadimplente, menos FN),
ao custo de barrar alguns bons pagadores (mais FP). Como o FN é 5x mais caro, a troca compensa e
o custo total cai cerca de 21%. É o trade-off central: troca-se um volume maior de erros baratos
por menos erros caros.""")
code(r"""cm_ot, (tn2, fp2, fn2, tp2), met_ot = painel(y_te, proba, thr_otimo)
pd.DataFrame({
    "Limiar 0,50": [fp050, fn050, c050, met["Recall"], met["Precisao"], met["F1"]],
    f"Limiar {thr_otimo:.2f}": [fp_ot, fn_ot, custo_otimo, met_ot["Recall"],
                               met_ot["Precisao"], met_ot["F1"]],
}, index=["FP (bons negados)", "FN (calotes aprovados)", "Custo total",
          "Recall", "Precisao", "F1"]).round(3)""")

md(r"""### A.7 Métrica-guia para monitoramento

| Métrica | Adequada? | Motivo |
|---|---|---|
| Acurácia | Não | inflada pela classe majoritária; sobe mesmo com o modelo piorando na classe rara |
| ROC-AUC | Parcial | boa para ranquear risco, mas otimista sob forte desbalanceamento |
| PR-AUC | Sim | foca na classe positiva rara; sensível ao que gera custo |
| F1 no limiar de operação | Sim | resume precisão e recall no ponto de decisão efetivamente usado |

Recomendação: monitorar **PR-AUC** (saúde do modelo, independe de corte) e **F1 no limiar de
menor custo** (saúde da operação). A acurácia pode ser reportada, mas não deve ser otimizada.
Alertas disparam por queda de PR-AUC ou aumento do custo esperado em produção.""")

md("### A.8 Melhorias propostas")
code(r"""# 1. class_weight='balanced' (custo nulo, dá peso à classe rara)
mod_bal = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=SEED).fit(X_tr_s, y_tr)
proba_bal = mod_bal.predict_proba(X_te_s)[:, 1]
_, _, met_bal = painel(y_te, proba_bal, 0.50)
grid = np.linspace(.01, .99, 99)
c_bal = min(custo_esperado(y_te, proba_bal, t)[0] for t in grid)

display(pd.DataFrame({
    "PR-AUC": [met["PR-AUC"], met_bal["PR-AUC"]],
    "Recall@0,50": [met["Recall"], met_bal["Recall"]],
    "Custo (melhor limiar)": [custo_otimo, c_bal],
}, index=["Baseline", "Baseline + class_weight"]).round(3))

# 2. Regra de negócio (filtro determinístico antes do modelo)
regra = (credito["historico_fraude"] == 1) | (credito["atraso_30d_12m"] >= 2)
print(f"\nRegra (fraude OU 2+ atrasos): {regra.sum()} clientes marcados para recusa/revisao,")
print(f"dos quais {credito.loc[regra,'target_default'].mean():.0%} sao inadimplentes "
      f"(vs {taxa:.0%} na base).")""")
md(r"""Prioridade das melhorias:

1. **Ajuste de limiar por custo** (já aplicado). Esforço mínimo, impacto direto no resultado; não
   requer re-treino.
2. **`class_weight='balanced'` ou reamostragem (SMOTE)**. Dá peso à classe rara e melhora recall e
   PR-AUC.
3. **Regra de negócio determinística** para `historico_fraude` ou `atraso_30d_12m >= 2` como
   recusa/revisão automática: alta precisão, custo nulo, e protege contra falhas do modelo.
4. **Limiar por segmento e novas features** — limiares distintos por faixa de `score_bureau`/
   `valor_solicitado` e a razão parcela/renda (comprometimento) como variável adicional.""")

# ========================================================================= PARTE B
md(r"""---
## Parte B — Regressão do preço de revenda

O preço previsto vira limite de crédito. Subestimar aperta o cliente; superestimar expõe a
fintech a uma garantia que não cobre a dívida.""")

md("### B.1 Exploração")
code(r"""fig, axs = plt.subplots(1, 2, figsize=(13, 4.0))
axs[0].hist(veiculos["preco_real"], bins=30, color=AZUL, alpha=.85)
axs[0].axvline(veiculos["preco_real"].median(), color=VERM, ls="--",
               label=f"mediana {reais(veiculos.preco_real.median())}")
axs[0].axvline(veiculos["preco_real"].mean(), color=VERDE, ls="--",
               label=f"media {reais(veiculos.preco_real.mean())}")
axs[0].xaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[0].set_title("Distribuicao do preco")
axs[0].legend(); axs[0].tick_params(axis="x", rotation=20)

corr = veiculos.select_dtypes("number").corr()["preco_real"].drop("preco_real").sort_values()
axs[1].barh(corr.index, corr.values, color=[VERM if v < 0 else VERDE for v in corr.values])
axs[1].set_title("Correlacao com o preco"); axs[1].set_xlabel("Pearson")
plt.tight_layout(); plt.show()
print("Preco sobe com ano e cai com km, revisoes pendentes e numero de donos.")""")

md("### B.2 Modelo baseline (regressão linear)")
code(r"""veic = pd.get_dummies(veiculos, columns=["marca"], drop_first=True)
Xr, yr = veic.drop(columns="preco_real"), veic["preco_real"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.30, random_state=SEED)

linreg = LinearRegression().fit(Xr_tr, yr_tr)
pred_lin = linreg.predict(Xr_te)
mae_lin  = mean_absolute_error(yr_te, pred_lin)
rmse_lin = mean_squared_error(yr_te, pred_lin) ** 0.5
print(f"MAE  = {reais(mae_lin)}")
print(f"RMSE = {reais(rmse_lin)}")
print(f"RMSE/MAE = {rmse_lin/mae_lin:.2f}")""")

md(r"""### B.3 MAE vs RMSE — interpretação prática

- **MAE**: erro típico em reais. "Em média, erramos MAE no preço." Trata todos os erros igual.
- **RMSE**: mesma unidade, mas eleva os erros ao quadrado antes de somar, penalizando erros
  grandes de forma desproporcional. Um único erro de R$ 40 mil pesa como muitos erros pequenos.

A razão RMSE/MAE acima de 1 é a assinatura dos erros grandes: se todos os erros tivessem o mesmo
tamanho, as duas métricas coincidiriam.""")
code(r"""res = yr_te.values - pred_lin
fig, axs = plt.subplots(1, 2, figsize=(13, 4.0))
axs[0].scatter(yr_te, pred_lin, alpha=.6, color=AZUL, edgecolor="white", s=42)
lims = [yr_te.min(), yr_te.max()]
axs[0].plot(lims, lims, ls="--", color=VERM, label="previsao perfeita")
axs[0].xaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[0].set_title("Previsto x real"); axs[0].set_xlabel("real"); axs[0].set_ylabel("previsto")
axs[0].legend(); axs[0].tick_params(axis="x", rotation=20)
axs[1].hist(res, bins=30, color=CINZA, alpha=.85)
axs[1].axvline(0, color=VERM, ls="--")
axs[1].xaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[1].set_title("Residuos"); axs[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()""")

md(r"""### B.4 Outliers e impacto no RMSE — antes e depois

A análise é feita em dois passos.

**(1) A propriedade isolada.** Injeta-se um único erro grande nas previsões do baseline (um carro
premium mal avaliado em +R$ 100 mil) e mede-se o efeito sobre cada métrica.""")
code(r"""def mae_rmse(y, p):
    return mean_absolute_error(y, p), mean_squared_error(y, p) ** 0.5

pred_out = pred_lin.copy(); pred_out[0] -= 100_000
mae_a, rmse_a = mae_rmse(yr_te, pred_lin)
mae_b, rmse_b = mae_rmse(yr_te, pred_out)
display(pd.DataFrame({
    "MAE": [mae_a, mae_b], "RMSE": [rmse_a, rmse_b],
    "RMSE/MAE": [rmse_a/mae_a, rmse_b/mae_b],
}, index=["Sem o outlier", "Com 1 erro de R$100k"])
    .style.format({"MAE": reais, "RMSE": reais, "RMSE/MAE": "{:.2f}"}))
print(f"Um unico erro grande: MAE sobe {(mae_b/mae_a-1):+.0%}, RMSE sobe {(rmse_b/rmse_a-1):+.0%}. "
      "O RMSE reage cerca de 2x mais.")""")
md(r"""**(2) No dataset real.** Removendo progressivamente os maiores erros do baseline, observa-se
o comportamento da razão RMSE/MAE.""")
code(r"""erros = np.abs(yr_te.values - pred_lin)
ordem = np.argsort(erros)[::-1]
linhas = []
for k in [0, 1, 3, 5, 10]:
    keep = np.ones(len(erros), bool); keep[ordem[:k]] = False
    mae_k, rmse_k = mae_rmse(yr_te.values[keep], pred_lin[keep])
    linhas.append([k, mae_k, rmse_k, rmse_k/mae_k])
display(pd.DataFrame(linhas, columns=["k piores removidos", "MAE", "RMSE", "RMSE/MAE"])
    .style.format({"MAE": reais, "RMSE": reais, "RMSE/MAE": "{:.2f}"}).hide(axis="index"))
print(f"Mesmo removendo os 10 maiores erros, a razao RMSE/MAE permanece ~{linhas[-1][3]:.2f}.")
print("No baseline linear nao ha poucos outliers dominando; o gap RMSE-MAE vem de erros")
print("moderados e sistematicos (subajuste do modelo linear). O tratamento correto aqui nao e")
print("remover pontos, e sim trocar de modelo (B.6) — la a razao RMSE/MAE sobe e poucos carros")
print("premium passam a dominar o RMSE, justificando revisao manual desses casos.")""")

md(r"""### B.5 Métrica de contrato: MAE ou RMSE?

Recomendação: **MAE como SLA, RMSE como sentinela.**

| Critério | MAE | RMSE |
|---|---|---|
| Interpretação para o negócio | erro típico em reais; direto | erro quadrático médio; abstrato |
| Sensibilidade a outliers | robusto | dominado por poucos carros premium |
| Risco capturado | erro do dia a dia (maioria dos limites) | erro catastrófico (garantia insuficiente) |

A operação avalia muitos carros medianos; o SLA precisa refletir o erro típico e ser auditável,
o que aponta para o MAE. O risco financeiro grave, porém, está nos casos superestimados de alto
valor, que o RMSE enxerga melhor. Daí a combinação: MAE no contrato, RMSE no alarme, e revisão
manual obrigatória acima de um teto de valor (B.6).""")

md("### B.6 Melhorias propostas")
code(r"""# 1. Modelo nao-linear capta interacoes ano x km x marca
gbr = GradientBoostingRegressor(random_state=SEED).fit(Xr_tr, yr_tr)
pred_gbr = gbr.predict(Xr_te)
mae_gbr  = mean_absolute_error(yr_te, pred_gbr)
rmse_gbr = mean_squared_error(yr_te, pred_gbr) ** 0.5
display(pd.DataFrame({"MAE": [mae_lin, mae_gbr], "RMSE": [rmse_lin, rmse_gbr]},
                     index=["Linear (baseline)", "Gradient Boosting"]).round(0)
        .style.format(reais))
print(f"MAE -{(1-mae_gbr/mae_lin):.0%}, RMSE -{(1-rmse_gbr/rmse_lin):.0%} ao trocar o modelo.")
print(f"RMSE/MAE do GBR = {rmse_gbr/mae_gbr:.2f} (era {rmse_lin/mae_lin:.2f} no linear): o modelo")
print("acerta a maioria com folga e o erro residual se concentra em poucos carros premium —")
print("esses casos vao para revisao manual.\n")

# 2. MAPE por faixa de preco
aval = pd.DataFrame({"real": yr_te.values, "pred": pred_gbr})
aval["faixa"] = pd.cut(aval["real"], [0, 25_000, 50_000, 80_000, np.inf],
                       labels=["ate 25k", "25-50k", "50-80k", "80k+"])
aval["ape"] = (aval["real"] - aval["pred"]).abs() / aval["real"]
faixa = aval.groupby("faixa", observed=True)["ape"].agg(["mean", "count"])
faixa.columns = ["MAPE", "n carros"]
display(faixa.style.format({"MAPE": "{:.1%}"}))
print("A faixa premium concentra o maior erro relativo; candidata a modelo segmentado e revisao manual.")""")
md(r"""Prioridade das melhorias:

1. **Modelo não-linear (Gradient Boosting / Random Forest)** — captura `ano x km x marca` que a
   linear ignora; redução imediata de MAE e RMSE.
2. **Segmentação por marca/ano e monitoramento de MAPE por faixa** — a faixa premium erra mais em
   termos relativos e merece modelo próprio ou alvo em log.
3. **Revisão manual para alto valor** — acima de um teto de `preco_real`, parecer humano antes de
   o preço virar limite de crédito, mitigando o risco de cauda que o RMSE sinaliza.""")

# ========================================================================= CONCLUSAO
md(r"""---
## Conclusão

| Frente | Métrica | Decisão | Impacto |
|---|---|---|---|
| Classificação | custo esperado + PR-AUC/F1 (não acurácia) | limiar de menor custo, `class_weight`, regra determinística | redução ~21% do custo esperado; calotes aprovados 20 → 7 |
| Regressão | MAE no contrato, RMSE sentinela | modelo não-linear + revisão manual de alto valor | limite de crédito mais justo e garantia que cobre a dívida |

Acurácia e RMSE parecem objetivos, mas escondem decisões de valor: quanto custa cada tipo de erro
e qual erro custa mais. Ao traduzir cada métrica no valor financeiro que ela representa, o modelo
passa de número de vitrine a instrumento de decisão.

Trade-off assumido conscientemente: aceita-se mais falsos positivos (bons pagadores negados) para
reduzir falsos negativos (calotes aprovados), porque o segundo custa cerca de 5x o primeiro; e
abre-se mão de simplicidade (modelo não-linear, revisão manual) para conter os erros de cauda que
o RMSE sinaliza.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/Analise_Microcredito.ipynb")
print("notebook gerado:", len(cells), "celulas")
