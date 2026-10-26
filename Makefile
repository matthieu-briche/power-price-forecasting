.PHONY: install lint test backtest real plot docker up down

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest --cov=powerfc

backtest:
	powerfc backtest --source synthetic --test-days 180 --out-dir reports

real:
	powerfc backtest --source energy-charts --start 2023-01-01 --days 1000 --test-days 180 --out-dir reports/real

plot:
	python scripts/plot_report.py reports

docker:
	docker build -t powerfc .

up:
	docker compose up --build --abort-on-container-exit pipeline

down:
	docker compose down -v
