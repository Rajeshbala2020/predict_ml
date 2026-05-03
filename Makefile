PYTHON := python3.11
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate
DB_URL := postgresql+psycopg2://postgres:postgres@localhost:5432/ml_service

.PHONY: setup up down logs train run test-predict quickstart

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f postgres

train:
	$(ACTIVATE) && python train.py

run:
	$(ACTIVATE) && DATABASE_URL="$(DB_URL)" uvicorn main:app --reload --port 8000

test-predict:
	$(ACTIVATE) && python -c "from service import StockPredictor; print(StockPredictor().predict('AAPL'))"

quickstart: setup up train run
