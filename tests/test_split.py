import json
from pathlib import Path

from logablation.attribution.split import make_split


def test_split_disjoint_deterministic_and_balanced() -> None:
    by = {f"s{i}": [["E1", "R1"], ["E1"], ["R1", "I1"], []][i % 4] for i in range(40)}
    a = make_split(by, 0.7, 42)
    assert a == make_split(by, 0.7, 42)
    assert not set(a["design"]) & set(a["holdout"])
    assert sorted(a["design"] + a["holdout"]) == sorted(by)
    assert 26 <= len(a["design"]) <= 30


def test_committed_split_covers_all_scenarios() -> None:
    sp = json.loads(Path("results/split.json").read_text())
    assert not set(sp["design"]) & set(sp["holdout"])
    assert len(sp["design"]) + len(sp["holdout"]) == 100
