PYTHON ?= python3
VENV   := .venv
PY     := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(PYTHON))
export PYTHONPATH := src

.PHONY: setup app extract outputs test eval eval-offline clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -q --upgrade pip
	$(VENV)/bin/python -m pip install -q -r requirements.txt

app:
	$(VENV)/bin/python -m pip install -q --only-binary :all: -r requirements-app.txt 2>/dev/null || true
	$(PY) -m streamlit run app.py

# make extract FILE=path/to/doc.pdf [ARGS=--offline]
extract:
	@test -n "$(FILE)" || (echo "usage: make extract FILE=path/to/doc.pdf" >&2; exit 2)
	@$(PY) -m co_extract "$(FILE)" $(ARGS)

outputs:
	@for f in samples/provided/* samples/real/* samples/synthetic/*; do \
	  case "$$f" in *.pdf|*.PDF|*.txt|*.md|*.png|*.jpg|*.jpeg) ;; *) continue ;; esac; \
	  set=$$(basename $$(dirname "$$f")); name=$$(basename "$$f"); name=$${name%.*}; \
	  mkdir -p outputs/$$set; echo "-> $$f"; \
	  $(PY) -m co_extract "$$f" $(ARGS) > outputs/$$set/$$name.json || exit 1; \
	done

test:
	$(PY) -m pytest -q

eval:
	$(PY) eval/run_eval.py $(ARGS)

eval-offline:
	$(PY) eval/run_eval.py --offline

clean:
	rm -rf .pytest_cache .cache **/__pycache__
