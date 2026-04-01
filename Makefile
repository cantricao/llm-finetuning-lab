.PHONY: install format lint typecheck test clean

install:
	poetry install

format:
	poetry run black src/ scripts/ tests/
	poetry run ruff check --fix src/ scripts/ tests/

lint:
	poetry run ruff check src/ scripts/ tests/
	poetry run black --check src/ scripts/ tests/

typecheck:
	poetry run mypy src/ scripts/

test:
	poetry run pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
