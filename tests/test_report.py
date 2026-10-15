from pathlib import Path

import pandas as pd

from logablation.report.style import md_table

FIGURES = [
    "F1_architecture", "F2_fault_map", "F3_manifested_vs_masked", "F4_ablation_drops",
    "F5_value_vs_cost", "F6_category_variant_heatmap", "F7_profile_frontier",
    "F8_confusion_matrices", "F9_holdout",
]  # fmt: skip
TABLES = [
    "T1a_dataset_by_incident_type", "T1b_dataset_by_variant", "T2_attribution_rule_checker",
    "T2b_llm_judge_dev_set", "T3_category_costs", "T3b_presidio_vs_planted",
    "T4_minimum_profile", "T5_holdout",
]  # fmt: skip


def test_report_regenerates_every_figure_and_table(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    from logablation.report import __main__ as rep

    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    monkeypatch.setattr(sys, "argv", ["report", "--figures", str(figs), "--tables", str(tabs)])
    rep.main()
    for f in FIGURES:
        assert (figs / f"{f}.png").stat().st_size > 5000, f
        assert "<svg" in (figs / f"{f}.svg").read_text()[:400], f
    for t in TABLES:
        assert len(pd.read_csv(tabs / f"{t}.csv")) > 0, t
        assert (tabs / f"{t}.md").read_text().startswith("| "), t


def test_report_output_is_deterministic(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    from logablation.report import __main__ as rep

    out = []
    for k in (1, 2):
        d = tmp_path / str(k)
        monkeypatch.setattr(sys, "argv", ["r", "--figures", str(d / "f"), "--tables", str(d / "t")])
        rep.main()
        out.append(d)
    for name in ("f/F3_manifested_vs_masked.svg", "f/F9_holdout.svg", "t/T5_holdout.md"):
        assert (out[0] / name).read_bytes() == (out[1] / name).read_bytes()


def test_committed_profile_is_the_one_the_holdout_scored() -> None:
    import json

    prof = json.loads(Path("results/profile.json").read_text())["profile"]
    held = pd.read_csv("results/holdout.csv")
    assert set(held[held.condition == "profile"].profile) == {prof}


def test_md_table_formats_numbers_and_blanks() -> None:
    md = md_table([{"a": 1.23456, "b": 18343.0, "c": None, "d": "x"}])
    assert md.splitlines()[2] == "| 1.23 | 18,343 |  | x |"
