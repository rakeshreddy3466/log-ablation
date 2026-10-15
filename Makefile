.PHONY: scenarios run attribute analyse report test lint

scenarios:
	uv run python -m logablation.scenarios --config configs/experiment.yaml

run:
	uv run python -m logablation.pipeline --config configs/experiment.yaml

attribute:
	@echo "attribute: not implemented yet (milestone: rule-based checker + LLM judge)"

analyse:
	@echo "analyse: not implemented yet (milestone: ablation, costs, profile)"

report:
	@echo "report: not implemented yet (milestone: figures and tables)"

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
