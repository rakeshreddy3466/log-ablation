"""Ablation of the rule-based checker over the design side (no model calls).

Conditions: all 64 subsets of C1..C6 kept (full, remove-one, add-one, skeleton-only and every
profile between). Groups: manifested (main), masked (secondary), plus clean and placebo runs as a
false-positive check. Intervals are cluster bootstrap over whole scenarios, paired with `full`.
Only scenarios on the design side are read; the holdout is not opened here.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..scenarios.taxonomy import FIRST_ERROR_STEP, VARIANTS
from ..tracing.categories import ABLATABLE, CATEGORY_IDS
from ..tracing.filter import filter_trace
from .rules import attribute

ID_OF = {v: k for k, v in CATEGORY_IDS.items()}
FIRST_ERROR_CLASSES = ["evidence_agent", "handover_1", "reasoning_agent", "handover_2"]


def profile_name(kept: tuple[str, ...]) -> str:
    return "+".join(ID_OF[c] for c in kept) if kept else "skeleton"


def all_profiles() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for n in range(len(ABLATABLE) + 1):
        out += [tuple(c) for c in itertools.combinations(ABLATABLE, n)]
    return out


def load_side(
    side: str, split_path: Path = Path("results/split.json"), root: Path = Path("runs")
) -> list[dict[str, Any]]:
    """Run metadata plus events for every run of the scenarios on `side`."""
    ids = set(json.loads(split_path.read_text())[side])
    runs = []
    for line in (root / "outcomes.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["scenario_id"] not in ids:
            continue
        events = [
            json.loads(x) for x in (root / "raw" / f"{r['run_id']}.jsonl").read_text().splitlines()
        ]
        runs.append({"meta": r, "events": events})
    return runs


def group_of(meta: dict[str, Any]) -> str | None:
    if meta["variant"] is None:
        return "clean"
    if meta["variant"] == "P0":
        return "placebo"
    return meta["outcome"]  # manifested | masked


def predict(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for prof in all_profiles():
        drop = [c for c in ABLATABLE if c not in prof]
        name = profile_name(prof)
        for r in runs:
            m = r["meta"]
            g = group_of(m)
            pred = attribute(filter_trace(r["events"], drop))
            v = m["variant"]
            rows.append({
                "run_id": m["run_id"], "scenario_id": m["scenario_id"], "group": g,
                "profile": name, "n_kept": len(prof),
                "true_variant": v if v in VARIANTS else None,
                "true_first_error": FIRST_ERROR_STEP.get(v) if v in VARIANTS else None,
                "pred_variant": pred["root_cause_variant"],
                "pred_first_error": pred["first_error_step"],
                "fired_rules": ",".join(pred["fired_rules"]),
            })  # fmt: skip
    return pd.DataFrame(rows)


def _macro_f1(true: np.ndarray, pred: np.ndarray, classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = int(((true == c) & (pred == c)).sum())
        fp = int(((true != c) & (pred == c)).sum())
        fn = int(((true == c) & (pred != c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def metrics(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for group in ("manifested", "masked"):
        g = pred[pred.group == group]
        scen = sorted(g.scenario_id.unique())
        pos = {s: i for i, s in enumerate(scen)}
        w = np.stack([
            np.bincount(rng.integers(0, len(scen), len(scen)), minlength=len(scen))
            for _ in range(n_boot)
        ]).astype(float)  # fmt: skip

        def per_scen(flag: np.ndarray, idx: np.ndarray, k: int = len(scen)) -> np.ndarray:
            return np.bincount(idx, weights=flag.astype(float), minlength=k)

        n_full = None
        base: dict[str, np.ndarray] = {}
        for prof, p in g.groupby("profile", sort=False):
            p = p.sort_values("run_id")
            idx = p.scenario_id.map(pos).to_numpy()
            n_runs = per_scen(np.ones(len(p)), idx)
            if n_full is None:
                n_full = n_runs
            fe_ok = (p.pred_first_error == p.true_first_error).to_numpy()
            rc_ok = (p.pred_variant == p.true_variant).to_numpy()
            row: dict[str, Any] = {
                "group": group, "profile": prof, "n_kept": int(p.n_kept.iloc[0]), "n_runs": len(p),
                "n_scenarios": len(scen),
                "first_error_acc": fe_ok.mean(), "root_cause_acc": rc_ok.mean(),
                "first_error_macro_f1": _macro_f1(
                    p.true_first_error.to_numpy(), p.pred_first_error.to_numpy(),
                    sorted(p.true_first_error.unique()),
                ),
                "root_cause_macro_f1": _macro_f1(
                    p.true_variant.to_numpy(), p.pred_variant.to_numpy(),
                    sorted(p.true_variant.unique()),
                ),
                "unknown_rate": float((p.pred_variant == "unknown").mean()),
            }  # fmt: skip
            for v in VARIANTS:
                sel = (p.true_variant == v).to_numpy()
                row[f"recall_{v}"] = float(rc_ok[sel].mean()) if sel.any() else np.nan
            boots = {
                "fe": (w @ per_scen(fe_ok, idx)) / (w @ n_runs),
                "rc": (w @ per_scen(rc_ok, idx)) / (w @ n_runs),
            }
            for v in VARIANTS:
                sel = (p.true_variant == v).to_numpy()
                nv = per_scen(sel, idx)
                with np.errstate(invalid="ignore", divide="ignore"):
                    boots[f"r_{v}"] = (w @ per_scen(rc_ok & sel, idx)) / (w @ nv)
            if prof == "C1+C2+C3+C4+C5+C6":
                base = boots
                base_row = row
            out.append({"_boots": boots, **row})
        # drops vs full
        for row in out:
            if row["group"] != group:
                continue
            b = row.pop("_boots")
            for key, col in (("fe", "first_error_acc"), ("rc", "root_cause_acc")):
                d = base[key] - b[key]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                row[f"{key}_drop"] = base_row[col] - row[col]
                row[f"{key}_drop_lo"], row[f"{key}_drop_hi"] = float(lo), float(hi)
                row[f"{key}_measurable"] = bool(lo > 0 or hi < 0)
            for v in VARIANTS:
                d = base[f"r_{v}"] - b[f"r_{v}"]
                if np.isnan(d).all():
                    lo = hi = np.nan
                else:
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                row[f"recall_drop_{v}"] = base_row[f"recall_{v}"] - row[f"recall_{v}"]
                row[f"recall_drop_{v}_lo"], row[f"recall_drop_{v}_hi"] = float(lo), float(hi)
    df = pd.DataFrame(out)
    # false-positive check on runs with no fault / a meaningless change
    fp = []
    for group in ("clean", "placebo"):
        for prof, p in pred[pred.group == group].groupby("profile", sort=False):
            fp.append({
                "group": group, "profile": prof, "n_kept": int(p.n_kept.iloc[0]),
                "n_runs": len(p), "fire_rate": float((p.fired_rules != "").mean()),
            })  # fmt: skip
    return pd.concat([df, pd.DataFrame(fp)], ignore_index=True)


def main() -> None:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())["bootstrap"]
    runs = load_side("design")
    for r in runs:  # truth labels of manifested runs must agree with the taxonomy mapping
        m = r["meta"]
        if m["outcome"] == "manifested":
            assert m["root_cause_variant"] == m["variant"], m["run_id"]
            assert m["first_error_step"] == FIRST_ERROR_STEP[m["variant"]], m["run_id"]
    pred = predict(runs)
    Path("attribution").mkdir(exist_ok=True)
    pred.to_parquet("attribution/predictions_rules.parquet", index=False)
    met = metrics(pred, cfg["resamples"], cfg["seed"])
    met.to_csv("results/rules_ablation.csv", index=False)
    print(f"design runs {len(runs)}; predictions {len(pred)}; metrics rows {len(met)}")


if __name__ == "__main__":
    main()
