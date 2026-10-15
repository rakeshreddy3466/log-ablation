.PHONY: scenarios run attribute analyse holdout report test lint

# 1. scenarios (seeded, regenerable; data/scenarios is git-ignored)
scenarios:
	uv run python -m logablation.scenarios --config configs/experiment.yaml

# 2. pipeline runs (needs Ollama with qwen2.5:3b; resumable, several hours)
run:
	uv run python -m logablation.pipeline --config configs/experiment.yaml

# 3. rule-based attribution and 64-profile ablation on the design side (no model calls)
attribute:
	uv run python -m logablation.attribution.split
	uv run python -m logablation.attribution.ablation

# 4. dataset results, costs and privacy, minimum profile (no model calls; needs runs/)
analyse:
	uv run python -m logablation.analysis.dataset
	uv run python -m logablation.analysis.costs
	uv run python -m logablation.analysis.profile

# 5. holdout scoring of the frozen results/profile.json (commit the profile first)
holdout:
	uv run python -m logablation.analysis.holdout

# 6. every figure (F1-F9) and table (T1-T5) from the committed result files only
report:
	uv run python -m logablation.report

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
