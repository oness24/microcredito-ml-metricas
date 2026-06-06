.PHONY: setup test train notebook html clean

PY := python

setup:
	$(PY) -m pip install -r requirements.txt

test:
	pytest -q

train:
	$(PY) -m scripts.train

notebook:
	$(PY) build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=300 notebooks/Analise_Microcredito.ipynb

html: notebook
	jupyter nbconvert --to html --output-dir reports notebooks/Analise_Microcredito.ipynb

clean:
	rm -rf models/*.joblib reports/figures/*.png reports/metrics.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f notebooks/dados_*.csv dados_*.csv
