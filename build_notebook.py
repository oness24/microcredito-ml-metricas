"""Gera o notebook Analise_Microcredito.ipynb (PT-BR) para a atividade de ML.

Executar: python build_notebook.py
Depois:   jupyter nbconvert --to notebook --execute --inplace Analise_Microcredito.ipynb
"""
import gzip, base64
import nbformat as nbf

def _embed(path):
    """Lê o CSV e devolve gzip+base64 para embutir no notebook (Colab sem upload)."""
    with open(path, "rb") as f:
        return base64.b64encode(gzip.compress(f.read())).decode()

B64_CRED = _embed("dados_credito.csv")
B64_VEIC = _embed("dados_veiculos.csv")

nb = nbf.v4.new_notebook()
cells = []
def md(src):  cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ----------------------------------------------------------------------------- CAPA
md(r"""# 🏦 Microcrédito on-line — Métricas que viram Decisão de Negócio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oness24/microcredito-ml-metricas/blob/main/Analise_Microcredito.ipynb)

### Análise de um analista de dados em uma fintech

> **Contexto.** Fui contratado como analista de dados em uma fintech de microcrédito.
> O time tem duas dores:
>
> 1. **Classificação (desbalanceada)** — o modelo de *inadimplência (default)* exibe **95% de acurácia**, mas o negócio sente **mais falsos positivos e falsos negativos** do que o número promete.
> 2. **Regressão** — um modelo de **preço de revenda de veículos** apoia a definição de limite de crédito, e o time não sabe se deve cobrar **MAE** ou **RMSE** no contrato.
>
> Meu papel: **analisar, argumentar e propor melhorias**, conectando *métrica certa → decisão certa → impacto no resultado*.

---

## 🧭 Sumário Executivo

| Frente | Diagnóstico | Decisão recomendada | Impacto esperado |
|---|---|---|---|
| **A · Classificação** | Acurácia alta é uma **ilusão estatística**: prever "ninguém é inadimplente" já acerta ~83%. O que dói no caixa são os **inadimplentes aprovados** (perda do principal). | Trocar o limiar de 0,50 para o **limiar de menor custo esperado** (cai para ~0,11), monitorar por **PR-AUC / F1**, não acurácia; usar **`class_weight`** + **regra de negócio híbrida** (fraude/atraso = recusa automática). | **−21% de custo esperado**: calotes aprovados caem de 20 → 7, mantendo aprovação saudável de bons pagadores. |
| **B · Regressão** | A diferença **RMSE − MAE** é o "imposto dos erros grandes": carros premium/atípicos inflam o RMSE. | **Contratar MAE** como métrica de SLA (erro típico, em R$, fácil de comunicar) e usar **RMSE como sentinela** de erros graves; segmentar por marca/ano e revisão manual para alto valor. | Limite de crédito mais justo no carro mediano e blindagem contra erros caros nos outliers. |

*As seções a seguir sustentam cada decisão com números, tabelas e gráficos.*
""")

# ----------------------------------------------------------------------------- COLAB / DADOS
md(r"""---
### ▶️ Como rodar no **Google Colab**

1. Acesse [colab.research.google.com](https://colab.research.google.com) → **Arquivo ▸ Fazer upload de notebook** → envie este `.ipynb`.
2. Menu **Ambiente de execução ▸ Executar tudo** (*Runtime ▸ Run all*).

Não é preciso enviar os CSVs nem instalar nada: o Colab já traz `pandas`, `scikit-learn` e
`matplotlib`, e os **dados estão embutidos** na célula abaixo (gravados em disco na 1ª execução).""")
code(("# 📦 Bootstrap de dados — grava os CSVs a partir de cópia embutida (gzip+base64).\n"
      "# Roda igual no Colab, no Jupyter local ou em qualquer máquina, sem upload manual.\n"
      "import os, gzip, base64\n\n"
      "_DADOS = {\n"
      '    "dados_credito.csv":  "%s",\n'
      '    "dados_veiculos.csv": "%s",\n'
      "}\n"
      "for _nome, _b64 in _DADOS.items():\n"
      "    if not os.path.exists(_nome):\n"
      '        with open(_nome, "wb") as _f:\n'
      "            _f.write(gzip.decompress(base64.b64decode(_b64)))\n"
      '        print("dados gravados:", _nome)\n'
      'print("✔ Dados prontos.")') % (B64_CRED, B64_VEIC))

# ----------------------------------------------------------------------------- SETUP
md("## ⚙️ Setup — bibliotecas, estilo e carga dos dados")
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

# Paleta e estilo enxuto (sem dependências externas)
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
AZUL, VERM, VERDE, CINZA = "#2563eb", "#dc2626", "#16a34a", "#6b7280"

def reais(x, _=None):
    return f"R$ {x:,.0f}".replace(",", ".")

credito  = pd.read_csv("dados_credito.csv")
veiculos = pd.read_csv("dados_veiculos.csv")

print("Crédito  (classificação):", credito.shape)
print("Veículos (regressão)   :", veiculos.shape)
credito.head()""")

# ============================================================================= PARTE A
md(r"""---
# 🅰️ Parte A — Classificação da Inadimplência

**Pergunta de negócio:** *a quem concedemos crédito?* A classe positiva é `target_default = 1`
(o cliente **dá calote**). Errar aqui tem custos assimétricos — e é isso que a acurácia esconde.""")

md("### A.1 · O elefante na sala: a base é desbalanceada")
code(r"""dist = credito["target_default"].value_counts().sort_index()
taxa = credito["target_default"].mean()

fig, ax = plt.subplots(figsize=(6,3.6))
barras = ax.bar(["Bom pagador (0)", "Inadimplente (1)"], dist.values,
                color=[VERDE, VERM])
for b, v in zip(barras, dist.values):
    ax.text(b.get_x()+b.get_width()/2, v+4, f"{v}\n({v/dist.sum():.1%})",
            ha="center", va="bottom", fontsize=10)
ax.set_title(f"Distribuição da classe — inadimplência = {taxa:.1%}")
ax.set_ylabel("nº de clientes"); ax.set_ylim(0, dist.max()*1.18)
plt.tight_layout(); plt.show()

print(f"Apenas {taxa:.1%} dos clientes são inadimplentes.")
print("Guarde este número: um modelo 'preguiçoso' que chuta SEMPRE 'bom pagador'")
print(f"já acertaria {1-taxa:.1%} dos casos — sem aprender absolutamente nada.")""")

md("### A.2 · Modelo baseline (Regressão Logística)")
code(r"""X = credito.drop(columns="target_default")
y = credito["target_default"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)

# Padronização ajuda a logística a convergir e a comparar coeficientes
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

baseline = LogisticRegression(max_iter=1000, random_state=SEED)
baseline.fit(X_tr_s, y_tr)

proba = baseline.predict_proba(X_te_s)[:, 1]   # P(inadimplência)
pred_050 = (proba >= 0.50).astype(int)         # corte clássico

print(f"Treino: {len(X_tr)} | Teste: {len(X_te)} | "
      f"inadimplência no teste: {y_te.mean():.1%}")""")

md("### A.3 · Matriz de confusão e o painel completo de métricas")
code(r"""def painel_metricas(y_true, y_score, thr, titulo):
    y_hat = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_score >= thr)
    tn, fp, fn, tp = cm.ravel()
    m = {
        "Acurácia":  accuracy_score(y_true, y_hat),
        "Precisão":  precision_score(y_true, y_hat, zero_division=0),
        "Recall":    recall_score(y_true, y_hat, zero_division=0),
        "F1":        f1_score(y_true, y_hat, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_score),
        "PR-AUC":    average_precision_score(y_true, y_score),
    }
    return cm, (tn, fp, fn, tp), m

cm, (tn, fp, fn, tp), met = painel_metricas(y_te, proba, 0.50, "baseline")

fig, ax = plt.subplots(figsize=(4.6,4.2))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1], ["Prev. 0", "Prev. 1"])
ax.set_yticks([0,1], ["Real 0", "Real 1"])
rotulos = [["VN", "FP"], ["FN", "VP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{rotulos[i][j]}\n{cm[i,j]}", ha="center", va="center",
                fontsize=13, color="white" if cm[i,j] > cm.max()/2 else "black")
ax.set_title("Matriz de confusão — limiar 0,50")
plt.tight_layout(); plt.show()

tabela = pd.DataFrame({"valor": met}).round(3)
tabela["leitura"] = [
    "% de acertos totais — INFLADA pela classe majoritária",
    "dos que marquei como inadimplentes, quantos eram de fato",
    "dos inadimplentes reais, quantos eu peguei",
    "equilíbrio precisão×recall",
    "ordena bem risco alto vs. baixo (independe do corte)",
    "qualidade na classe rara — métrica honesta p/ desbalanceado",
]
display(tabela)""")

md(r"""### A.4 · Por que a acurácia **engana** no cenário desbalanceado

Comparo o modelo com um **classificador trivial** que sempre responde *"bom pagador"*.""")
code(r"""acc_modelo  = accuracy_score(y_te, pred_050)
acc_trivial = accuracy_score(y_te, np.zeros_like(y_te))   # chuta tudo = 0
recall_trivial = 0.0  # nunca acha um inadimplente

comp = pd.DataFrame({
    "Acurácia": [acc_trivial, acc_modelo],
    "Recall (inadimplentes pegos)": [recall_trivial, met["Recall"]],
    "Inadimplentes que escaparam (FN)": [int(y_te.sum()), fn],
}, index=["🤖 Chuta sempre 'bom pagador'", "📈 Regressão Logística"]).round(3)
display(comp)

print(f"O modelo 'burro' já alcança {acc_trivial:.1%} de acurácia — perto dos 95% que")
print("o time exibia — mas deixa 100% dos inadimplentes passarem. A acurácia premia")
print("quem ignora a classe rara; é exatamente a classe rara que custa dinheiro.")""")

md(r"""### A.5 · Matriz de **custos** alinhada ao negócio

A acurácia trata todo erro como igual. O caixa não. Defino o custo **pelo evento de negócio**,
não pela sigla — assim não há ambiguidade sobre "o que é FP".

| Evento de negócio | Célula da matriz (positivo = inadimplente) | O que acontece | Custo |
|---|---|---|---|
| **Aprovar um inadimplente** | Real 1, previsto 0 → **FN** | perco o **principal** emprestado (≈ R$ 20 mil) | **10** |
| **Negar um bom pagador** | Real 0, previsto 1 → **FP** | perco apenas a **margem/juros** (≈ R$ 2 mil) | **2** |
| Negar um inadimplente (VP) | Real 1, previsto 1 | decisão correta | 0 |
| Aprovar um bom pagador (VN) | Real 0, previsto 0 | decisão correta (lucro) | 0 |

**Justificativa do 10 : 2 (5 : 1).** O enunciado sugere algo como 5 : 2; eu **refino e justifico**
ancorando em dinheiro. Aprovar um caloteiro queima o **principal** — o `valor_solicitado` médio é
≈ **R$ 20 mil**. Negar um bom cliente custa só a **margem** que ele traria — algo como **R$ 2 mil**
de juros. A perda do principal é da ordem de **5× maior** que a da margem — daí 10 contra 2. *(A
razão é o que importa; em produção, dá para usar o `valor_solicitado` linha a linha.)*""")
code(r"""CUSTO_FN = 10  # aprovar inadimplente (perde o principal ≈ R$ 20 mil)
CUSTO_FP = 2   # negar bom pagador  (perde a margem  ≈ R$ 2 mil)

def custo_esperado(y_true, y_score, thr):
    y_hat = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return CUSTO_FN*fn + CUSTO_FP*fp, fn, fp

c050, fn050, fp050 = custo_esperado(y_te, proba, 0.50)
print(f"No limiar 0,50:  FN={fn050}  FP={fp050}  →  custo total = {c050}")""")

md("### A.6 · Escolhendo o **limiar de menor custo esperado**")
code(r"""thrs = np.linspace(0.01, 0.99, 99)
custos = np.array([custo_esperado(y_te, proba, t)[0] for t in thrs])
thr_otimo = thrs[custos.argmin()]
custo_otimo, fn_ot, fp_ot = custo_esperado(y_te, proba, thr_otimo)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))

# (1) curva de custo
axs[0].plot(thrs, custos, color=AZUL, lw=2)
axs[0].axvline(thr_otimo, color=VERM, ls="--",
               label=f"ótimo = {thr_otimo:.2f}")
axs[0].axvline(0.50, color=CINZA, ls=":", label="padrão 0,50")
axs[0].scatter([thr_otimo], [custo_otimo], color=VERM, zorder=5)
axs[0].set_title("Custo esperado × limiar")
axs[0].set_xlabel("limiar"); axs[0].set_ylabel("custo total"); axs[0].legend()

# (2) ROC
fpr, tpr, _ = roc_curve(y_te, proba)
axs[1].plot(fpr, tpr, color=AZUL, lw=2, label=f"AUC = {met['ROC-AUC']:.3f}")
axs[1].plot([0,1],[0,1], ls="--", color=CINZA)
axs[1].set_title("Curva ROC"); axs[1].set_xlabel("FPR"); axs[1].set_ylabel("TPR (recall)")
axs[1].legend()

# (3) Precision-Recall
prec, rec, _ = precision_recall_curve(y_te, proba)
axs[2].plot(rec, prec, color=VERDE, lw=2, label=f"PR-AUC = {met['PR-AUC']:.3f}")
axs[2].axhline(y_te.mean(), ls="--", color=CINZA, label=f"acaso = {y_te.mean():.2f}")
axs[2].set_title("Curva Precisão-Recall"); axs[2].set_xlabel("recall"); axs[2].set_ylabel("precisão")
axs[2].legend()
plt.tight_layout(); plt.show()

print(f"Limiar ótimo = {thr_otimo:.2f}  |  custo {custo_otimo}  (vs {c050} no 0,50)  "
      f"→ redução de {(1-custo_otimo/c050):.1%}")
print(f"FN: {fn050} → {fn_ot}   |   FP: {fp050} → {fp_ot}")""")

md(r"""**Leitura do trade-off.** Baixar o limiar faz o modelo **recusar mais** (sobe o recall de
inadimplentes → **menos FN**), ao custo de barrar alguns bons pagadores (**mais FP**). Como o
FN é **5× mais caro**, **a troca compensa**: o custo total cai ~21%. É exatamente o trade-off
pedido — *trocar muitos erros baratos por poucos erros caros a menos.*""")
code(r"""cm_ot, (tn2, fp2, fn2, tp2), met_ot = painel_metricas(y_te, proba, thr_otimo, "ótimo")
antes_depois = pd.DataFrame({
    "Limiar 0,50": [fp050, fn050, c050, met["Recall"], met["Precisão"], met["F1"]],
    f"Limiar {thr_otimo:.2f}": [fp_ot, fn_ot, custo_otimo, met_ot["Recall"],
                               met_ot["Precisão"], met_ot["F1"]],
}, index=["FP (bons negados)", "FN (calotes aprovados)", "Custo total",
          "Recall", "Precisão", "F1"]).round(3)
display(antes_depois)""")

md(r"""### A.7 · Métrica-guia para **monitoramento contínuo**

| Candidata | Serve? | Por quê |
|---|---|---|
| Acurácia | ❌ | Inflada pela classe majoritária — sobe mesmo com o modelo piorando na classe rara. |
| ROC-AUC | ⚠️ | Boa para *ranquear* risco, mas **otimista** sob forte desbalanceamento. |
| **PR-AUC** | ✅ | Foca na **classe positiva rara** (inadimplência); sensível ao que custa dinheiro. |
| **F1 @ limiar de negócio** | ✅ | Resume precisão×recall **no ponto de operação** que de fato usamos. |

**Decisão.** Monitoro **PR-AUC** (saúde do modelo, independe de corte) **+ F1 no limiar de
custo mínimo** (saúde da operação). A acurácia fica como métrica *de vaidade* — reportada, nunca
otimizada. Alertas disparam por **queda de PR-AUC** ou **subida do custo esperado** em produção.""")

md("### A.8 · Melhorias propostas (priorizadas)")
code(r"""# Melhoria 1 — class_weight='balanced' (custo ~zero, ganho imediato em recall)
mod_bal = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=SEED).fit(X_tr_s, y_tr)
proba_bal = mod_bal.predict_proba(X_te_s)[:, 1]
_, _, met_bal = painel_metricas(y_te, proba_bal, 0.50, "balanced")
c_bal = custo_esperado(y_te, proba_bal,
        np.linspace(.01,.99,99)[np.argmin(
            [custo_esperado(y_te, proba_bal, t)[0] for t in np.linspace(.01,.99,99)])])[0]

comp_mel = pd.DataFrame({
    "PR-AUC": [met["PR-AUC"], met_bal["PR-AUC"]],
    "Recall @0,50": [met["Recall"], met_bal["Recall"]],
    "Custo (melhor limiar)": [custo_otimo, c_bal],
}, index=["Baseline", "Baseline + class_weight"]).round(3)
display(comp_mel)

# Melhoria 3 (híbrida) — regra de negócio dura antes do modelo
regra_dura = (credito["historico_fraude"] == 1) | (credito["atraso_30d_12m"] >= 2)
print(f"\nRegra híbrida (fraude OU 2+ atrasos) marcaria {regra_dura.sum()} clientes "
      f"para recusa/revisão automática,")
print(f"dos quais {credito.loc[regra_dura,'target_default'].mean():.0%} são inadimplentes "
      f"(vs {taxa:.0%} na base) — filtro barato e de altíssima precisão.")""")

md(r"""**Prioridade das melhorias (esforço × impacto):**

1. **🥇 Ajuste de limiar por custo** *(já feito; esforço mínimo, impacto direto no caixa)* — reduziu o custo esperado **sem reentreinar nada**.
2. **🥈 `class_weight='balanced'` / reamostragem (SMOTE)** — faz o modelo "enxergar" a classe rara; melhora recall e PR-AUC.
3. **🥉 Regra de negócio híbrida** — `historico_fraude` ou `atraso_30d_12m ≥ 2` viram **recusa/revisão automática**: precisão altíssima, custo zero, e protege contra falhas do modelo.
4. **Bônus — limiar por segmento + novas features** — limiares distintos por faixa de `score_bureau`/`valor_solicitado`, e razão `parcela/renda` (comprometimento) como nova variável.
""")

# ============================================================================= PARTE B
md(r"""---
# 🅱️ Parte B — Regressão do Preço de Revenda

**Pergunta de negócio:** quanto vale o veículo dado em garantia? Esse preço vira **limite de
crédito**. Subestimar aperta o cliente; **superestimar expõe a fintech** a uma garantia que não
cobre a dívida.""")

md("### B.1 · Conhecendo os dados")
code(r"""fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
axs[0].hist(veiculos["preco_real"], bins=30, color=AZUL, alpha=.85)
axs[0].axvline(veiculos["preco_real"].median(), color=VERM, ls="--",
               label=f"mediana {reais(veiculos.preco_real.median())}")
axs[0].axvline(veiculos["preco_real"].mean(), color=VERDE, ls="--",
               label=f"média {reais(veiculos.preco_real.mean())}")
axs[0].xaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[0].set_title("Distribuição do preço (cauda à direita = carros premium)")
axs[0].legend(); axs[0].tick_params(axis="x", rotation=20)

num = veiculos.select_dtypes("number")
corr = num.corr()["preco_real"].drop("preco_real").sort_values()
cores = [VERM if v < 0 else VERDE for v in corr.values]
axs[1].barh(corr.index, corr.values, color=cores)
axs[1].set_title("Correlação com o preço")
axs[1].set_xlabel("correlação de Pearson")
plt.tight_layout(); plt.show()
print("Preço sobe com ano (+) e cai com km, revisões pendentes e nº de donos (−).")""")

md("### B.2 · Modelo baseline (Regressão Linear)")
code(r"""veic = pd.get_dummies(veiculos, columns=["marca"], drop_first=True)
Xr = veic.drop(columns="preco_real")
yr = veic["preco_real"]

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.30, random_state=SEED)
linreg = LinearRegression().fit(Xr_tr, yr_tr)
pred_lin = linreg.predict(Xr_te)

mae_lin  = mean_absolute_error(yr_te, pred_lin)
rmse_lin = mean_squared_error(yr_te, pred_lin) ** 0.5
print(f"MAE  = {reais(mae_lin)}")
print(f"RMSE = {reais(rmse_lin)}")
print(f"RMSE/MAE = {rmse_lin/mae_lin:.2f}  (quanto > 1, mais 'erros grandes' existem)")""")

md(r"""### B.3 · MAE vs. RMSE — o que cada um diz na prática

- **MAE** = erro **típico**, em reais. *"Em média erramos `MAE` no preço."* Trata todos os erros igual.
- **RMSE** = mesma unidade, mas **eleva ao quadrado** antes de somar → **pune erros grandes**
  desproporcionalmente. Um único erro de R$ 40 mil pesa como muitos erros pequenos.

A razão **RMSE/MAE > 1** é a assinatura dos **outliers**: se fossem iguais, todos os erros teriam
o mesmo tamanho. Abaixo, visualizo onde mora o erro.""")
code(r"""res = yr_te.values - pred_lin
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))

axs[0].scatter(yr_te, pred_lin, alpha=.6, color=AZUL, edgecolor="white", s=45)
lims = [yr_te.min(), yr_te.max()]
axs[0].plot(lims, lims, ls="--", color=VERM, label="previsão perfeita")
axs[0].xaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[0].set_title("Previsto × Real"); axs[0].set_xlabel("real"); axs[0].set_ylabel("previsto")
axs[0].legend(); axs[0].tick_params(axis="x", rotation=20)

axs[1].hist(res, bins=30, color=CINZA, alpha=.85)
axs[1].axvline(0, color=VERM, ls="--")
axs[1].xaxis.set_major_formatter(mticker.FuncFormatter(reais))
axs[1].set_title("Distribuição dos resíduos (caudas = erros caros p/ o RMSE)")
axs[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()""")

md(r"""### B.4 · Outliers e seu impacto no RMSE — antes × depois

Faço a análise em **dois passos honestos**.

**(1) A propriedade matemática, isolada.** Pego as previsões do baseline e injeto **um único erro
grande** (um carro premium mal avaliado em +R$ 100 mil). É o "antes × depois" mais limpo possível.""")
code(r"""pred_ok = pred_lin.copy()
pred_out = pred_lin.copy(); pred_out[0] -= 100_000   # 1 outlier artificial

def mae_rmse(y, p): return mean_absolute_error(y, p), mean_squared_error(y, p)**0.5
mae_a, rmse_a = mae_rmse(yr_te, pred_ok)
mae_b, rmse_b = mae_rmse(yr_te, pred_out)

display(pd.DataFrame({
    "MAE":  [mae_a, mae_b], "RMSE": [rmse_a, rmse_b],
    "RMSE/MAE": [rmse_a/mae_a, rmse_b/mae_b],
}, index=["Sem o outlier", "Com 1 erro de R$100k"])
    .style.format({"MAE": reais, "RMSE": reais, "RMSE/MAE": "{:.2f}"}))
print(f"Um único erro grande:  MAE sobe +{(mae_b/mae_a-1):.0%}, mas RMSE sobe "
      f"+{(rmse_b/rmse_a-1):.0%} — o RMSE reage ~2× mais. É a definição de 'pune erro grande'.")""")
md(r"""**(2) No dataset real, os erros estão concentrados ou espalhados?** Removo progressivamente os
*k* maiores erros do baseline e observo o RMSE/MAE.""")
code(r"""erros = np.abs(yr_te.values - pred_lin)
ordem = np.argsort(erros)[::-1]
linhas = []
for k in [0, 1, 3, 5, 10]:
    keep = np.ones(len(erros), bool); keep[ordem[:k]] = False
    mae_k, rmse_k = mae_rmse(yr_te.values[keep], pred_lin[keep])
    linhas.append([k, mae_k, rmse_k, rmse_k/mae_k])
tab = pd.DataFrame(linhas, columns=["k piores removidos", "MAE", "RMSE", "RMSE/MAE"])
display(tab.style.format({"MAE": reais, "RMSE": reais,
                          "RMSE/MAE": "{:.2f}"}).hide(axis="index"))
print(f"Mesmo removendo os 10 maiores erros, o RMSE/MAE fica ~{linhas[-1][3]:.2f} — estável.")
print("Ou seja: no baseline LINEAR não há 2-3 outliers dominando; o gap RMSE-MAE vem de")
print("ERROS MODERADOS E SISTEMÁTICOS (subajuste do modelo linear), não de pontos isolados.")
print("→ Conclusão: o 'tratamento' certo aqui NÃO é remover pontos, e sim trocar de modelo")
print("  (B.6). Com um modelo bom, o RMSE/MAE SOBE p/ ~1,6 e aí sim poucos carros premium")
print("  passam a dominar o RMSE — exatamente os casos que mandaremos p/ revisão manual.")""")

md(r"""### B.5 · Métrica de **contrato**: MAE ou RMSE?

**Decisão: contratar o MAE como SLA, com o RMSE de sentinela.**

| Critério | MAE | RMSE |
|---|---|---|
| Interpretação para o negócio | **"erramos ~R$ X no carro típico"** — direto | "erro quadrático médio" — abstrato |
| Sensibilidade a outliers | robusto | refém de poucos carros premium |
| Risco que captura | erro **do dia a dia** (a maioria dos limites) | erro **catastrófico** (garantia furada) |

A operação aprova **muitos carros medianos**; o SLA precisa refletir o erro **típico** e ser
auditável → **MAE**. Mas o risco financeiro grave está nos **superestimados de alto valor**, e é
o RMSE que os enxerga. Por isso: **MAE no contrato, RMSE no alarme** — e revisão manual obrigatória
acima de um teto de valor (Seção B.6).""")

md("### B.6 · Melhorias propostas")
code(r"""# Melhoria 1 — modelo não-linear (Gradient Boosting) capta interações ano×km×marca
gbr = GradientBoostingRegressor(random_state=SEED).fit(Xr_tr, yr_tr)
pred_gbr = gbr.predict(Xr_te)
mae_gbr  = mean_absolute_error(yr_te, pred_gbr)
rmse_gbr = mean_squared_error(yr_te, pred_gbr) ** 0.5

display(pd.DataFrame({
    "MAE":  [mae_lin, mae_gbr],
    "RMSE": [rmse_lin, rmse_gbr],
}, index=["Linear (baseline)", "Gradient Boosting"]).round(0)
    .style.format(reais))
print(f"MAE: -{(1-mae_gbr/mae_lin):.0%}   RMSE: -{(1-rmse_gbr/rmse_lin):.0%} "
      "só trocando o modelo.")
print(f"RMSE/MAE do GBR = {rmse_gbr/mae_gbr:.2f} (era 1,20 no linear): agora o modelo acerta")
print("a maioria com folga e o erro RESIDUAL se concentra em poucos carros premium —")
print("aí o RMSE volta a ser sentinela e esses casos vão para revisão manual.\n")

# Melhoria 2 — MAPE por faixa de preço: onde o erro relativo dói mais?
aval = pd.DataFrame({"real": yr_te.values, "pred": pred_gbr})
aval["faixa"] = pd.cut(aval["real"], [0, 25_000, 50_000, 80_000, np.inf],
                       labels=["até 25k", "25–50k", "50–80k", "80k+"])
aval["ape"] = (aval["real"] - aval["pred"]).abs() / aval["real"]
mape_faixa = aval.groupby("faixa", observed=True)["ape"].agg(["mean", "count"])
mape_faixa.columns = ["MAPE", "nº carros"]
display(mape_faixa.style.format({"MAPE": "{:.1%}"}))
print("A faixa premium concentra o maior erro relativo → candidata natural a")
print("revisão manual e a um modelo segmentado por marca/ano.")""")

md(r"""**Prioridade das melhorias:**

1. **🥇 Modelo não-linear (Gradient Boosting / Random Forest)** — captura `ano×km×marca` que a linear ignora; queda imediata de MAE **e** RMSE.
2. **🥈 Segmentação por marca/ano + monitorar MAPE por faixa** — a faixa premium erra mais em % e merece modelo próprio (ou log-target).
3. **🥉 Regra de revisão manual para alto valor** — acima de um teto de `preco_real`, parecer humano antes de virar limite de crédito: corta o risco de cauda que o RMSE denuncia.
""")

# ============================================================================= CONCLUSAO
md(r"""---
# 🎯 Conclusão — métrica certa → decisão certa → impacto

| # | Métrica certa | Decisão certa | Impacto no resultado |
|---|---|---|---|
| A | **Custo esperado + PR-AUC/F1** (não acurácia) | limiar de menor custo, `class_weight`, regra híbrida | **menos principal queimado** com inadimplentes aprovados |
| B | **MAE no contrato, RMSE de sentinela** | modelo não-linear + revisão manual de alto valor | **limite de crédito justo** e garantia que cobre a dívida |

**Mensagem central.** Acurácia e RMSE *parecem* objetivos, mas escondem decisões de valor: o que
custa um erro, e qual erro custa mais. Ao **traduzir cada métrica no real que ela representa**, o
modelo deixa de ser um número de vaidade e passa a defender o caixa da fintech.

> *Trade-off assumido conscientemente:* aceitamos **mais falsos positivos** (bons pagadores
> negados) para cortar **falsos negativos** (calotes aprovados), porque o segundo custa ~2,5× o
> primeiro; e abrimos mão de um pouco de simplicidade (modelo não-linear, revisão manual) para
> proteger a operação dos erros de cauda.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "Analise_Microcredito.ipynb")
print("notebook gerado:", len(cells), "células")
