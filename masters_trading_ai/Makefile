.PHONY: install run test clean lint

# ── Setup ──────────────────────────────────────
install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# ── Run ────────────────────────────────────────
run:
	.venv/bin/python webapp/server.py

# ── Test ───────────────────────────────────────
test:
	.venv/bin/python -m pytest tests/ -v

# ── Clean ──────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Lint ───────────────────────────────────────
lint:
	.venv/bin/python -m py_compile webapp/server.py
	.venv/bin/python -m py_compile src/inference/predictor.py
