"""T1..T5 from the committed result files, as CSV and markdown under results/tables/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .style import TABLES, md_table

CATS = ["C1", "C2", "C3", "C4", "C5", "C6"]
FULL = "+".join(CATS)


def _write(name: str, df: pd.DataFrame, out_dir: Path, fmt: str = "{:.3g}") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    rows: list[dict[str, Any]] = df.to_dict("records")  # type: ignore[assignment]
    (out_dir / f"{name}.md").write_text(md_table(rows, fmt))


def t1(res: Path, out_dir: Path) -> None:
    by_type = pd.read_csv(res / "t1_by_incident_type.csv")
    elig = pd.read_csv(res / "eligibility.csv")[["incident_type", "rate"]]
    by_type = by_type.merge(elig.rename(columns={"rate": "eligibility_rate"}), how="left",
                            on="incident_type")  # fmt: skip
    by_type = by_type.rename(columns={"incident_type": "incident type"})
    _write("T1a_dataset_by_incident_type", by_type, out_dir)
    r = pd.read_csv(res / "manifestation_rates.csv")
    v = r[r.variant != "P0"][["variant", "scenarios", "runs", "events", "rate", "ci_lo", "ci_hi",
                              "diff_vs_placebo", "diff_lo", "diff_hi", "clearly_above_placebo"]]  # fmt: skip
    v = v.rename(columns={"events": "manifested"}).assign(masked=lambda d: d.runs - d.manifested)
    p = r[r.variant == "P0"].iloc[0]
    v = pd.concat([v, pd.DataFrame([{
        "variant": "P0 (placebo flip rate)", "scenarios": p.scenarios, "runs": p.runs,
        "manifested": p.events, "masked": p.runs - p.events, "rate": p.rate,
        "ci_lo": p.ci_lo, "ci_hi": p.ci_hi}])], ignore_index=True)  # fmt: skip
    _write("T1b_dataset_by_variant", v, out_dir)


def t2(res: Path, out_dir: Path) -> None:
    a = pd.read_csv(res / "rules_ablation.csv")
    conds = [("full", FULL)] + [(f"minus {c}", "+".join(x for x in CATS if x != c)) for c in CATS]
    conds += [("skeleton only", "skeleton")] + [(f"skeleton + {c}", c) for c in CATS]
    rows = []
    for label, prof in conds:
        row: dict[str, Any] = {"condition": label}
        for g, tag in (("manifested", "man"), ("masked", "mask")):
            r = a[(a.group == g) & (a.profile == prof)].iloc[0]
            row |= {
                f"{tag} first-error acc": r.first_error_acc,
                f"{tag} root-cause acc": r.root_cause_acc,
                f"{tag} root-cause macro-F1": r.root_cause_macro_f1,
            }
            if tag == "man":
                ci = f"{r.rc_drop:+.3f} [{r.rc_drop_lo:.3f}, {r.rc_drop_hi:.3f}]"
                row["man root-cause drop [95% CI]"] = ci
                row["man measurable"] = "yes" if r.rc_measurable else "no measurable value"
        rows.append(row)
    _write("T2_attribution_rule_checker", pd.DataFrame(rows), out_dir)
    j = []
    for f, label in (("v2_qwen2.5", "v2 qwen2.5:7b"), ("v3_qwen2.5", "v3 qwen2.5:7b"),
                     ("v3_phi4-mini", "v3 phi4-mini")):  # fmt: skip
        d = json.loads((res / f"judge_dev_{f}.json").read_text())
        j.append({"judge": label, "dev runs": d["n"], "first-error acc": d["first_error_acc"],
                  "root-cause acc": d["root_cause_acc"], "top label": d["top_label"],
                  "top label share": d["top_share"], "valid": f"{d['valid']}/{d['n']}"})  # fmt: skip
    _write("T2b_llm_judge_dev_set", pd.DataFrame(j), out_dir)


def t3(res: Path, out_dir: Path) -> None:
    c = pd.read_csv(res / "costs.csv")
    keep = ["key", "bytes_median", "bytes_p95", "gzip_median", "gzip_p95", "share_of_total_bytes",
            "events_with_entity_share", "entities_per_run_mean", "planted_present_per_run_mean",
            "write_ms_median", "write_ms_p95", "write_pct_of_median_run"]  # fmt: skip
    _write("T3_category_costs", c[keep].rename(columns={"key": "category"}), out_dir, "{:.4g}")
    d = json.loads((res / "privacy_detector.json").read_text())
    rows = [{"entity": k, **{x: v[x] for x in ("planted", "found", "detected", "false",
                                                 "recall", "precision")}}
            for k, v in d.items() if not k.startswith("_")]  # fmt: skip
    _write("T3b_presidio_vs_planted", pd.DataFrame(rows), out_dir, "{:.3f}")


def t4(res: Path, out_dir: Path) -> None:
    p = json.loads((res / "profile.json").read_text())
    m = pd.read_csv(res / "profile_candidates_manifested.csv")
    full = m[m.profile == FULL].iloc[0]
    ok = m[(m.first_error_acc >= full.first_error_acc - 0.05)
           & (m.root_cause_acc >= full.root_cause_acc - 0.05)]  # fmt: skip
    top = ok.sort_values(["bytes_median", "planted_per_run"]).head(6)
    rows = []
    for _, r in pd.concat([top, m[m.profile.isin([FULL, "skeleton"])]]).iterrows():
        rows.append({
            "profile": r.profile + ("  (frozen)" if r.profile == p["profile"] else ""),
            "first-error acc": r.first_error_acc, "root-cause acc": r.root_cause_acc,
            "median bytes/run": r.bytes_median, "median gzip/run": r.gzip_median,
            "planted values/run": r.planted_per_run,
        })  # fmt: skip
    _write("T4_minimum_profile", pd.DataFrame(rows), out_dir, "{:.4g}")


def t5(res: Path, out_dir: Path) -> None:
    h = pd.read_csv(res / "holdout.csv")
    cols = ["group", "condition", "profile", "n_runs", "n_scenarios", "first_error_acc",
            "first_error_lo", "first_error_hi", "root_cause_acc", "root_cause_lo", "root_cause_hi",
            "first_error_drop", "root_cause_drop", "median_bytes_per_run", "planted_values_per_run"]  # fmt: skip
    _write("T5_holdout", h[cols], out_dir, "{:.4g}")


def main(res: Path = Path("results"), out_dir: Path = TABLES) -> None:
    for fn in (t1, t2, t3, t4, t5):
        fn(res, out_dir)
