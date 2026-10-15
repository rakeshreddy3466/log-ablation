"""Minimum profile on the design side: candidates, scoring, selection, F5 and F7.

A candidate keeps each of C1, C4, C5, C6 or drops it, and takes C2 as {off, raw, masked,
masked_x} and C3 as {off, raw, masked, masked_x, light}: 4 x 5 x 16 = 320 candidates. The rule
checker scores every one on the design side (manifested = main, masked = secondary). Selection is
the CLAUDE.md rule: smallest median trace size whose first-error AND root-cause accuracy are both
within 5 points of full; tie-break on fewer sensitive values. Only design-side rows are read.
"""

from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..attribution.ablation import load_side, metrics
from ..attribution.rules import attribute
from ..scenarios.taxonomy import FIRST_ERROR_STEP, VARIANTS
from ..tracing.categories import CATEGORY_IDS, SKELETON
from .costs import build_alternatives
from .privacy import Detector

IDS = list(CATEGORY_IDS)  # C1..C6
SUFFIX = {"raw": "", "masked": "m", "masked_x": "x", "light": "l"}
C2_MODES = (None, "raw", "masked", "masked_x")
C3_MODES = (None, "raw", "masked", "masked_x", "light")
FULL = {"C1": "raw", "C2": "raw", "C3": "raw", "C4": "raw", "C5": "raw", "C6": "raw"}
FULL_NAME = "C1+C2+C3+C4+C5+C6"
TOLERANCE = 0.05
Choice = dict[str, str | None]


def name_of(choice: Choice) -> str:
    parts = [f"{c}{SUFFIX[m]}" for c, m in choice.items() if m]
    return "+".join(parts) if parts else "skeleton"


def candidates() -> list[Choice]:
    out = []
    for c2, c3 in itertools.product(C2_MODES, C3_MODES):
        for flags in itertools.product((None, "raw"), repeat=4):
            c1, c4, c5, c6 = flags
            out.append({"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6})
    return out


def apply_profile(
    events: list[dict[str, Any]], alt: dict[str, dict[int, dict[str, Any]]], choice: Choice
) -> list[dict[str, Any]]:
    """The trace a profile would have logged: skeleton plus each category in its chosen form."""
    out = []
    for i, ev in enumerate(events):
        cat = ev["category"]
        if cat == SKELETON:
            out.append(ev)
            continue
        cid = next(k for k, v in CATEGORY_IDS.items() if v == cat)
        mode = choice[cid]
        if mode is None:
            continue
        out.append(ev if mode == "raw" else alt[f"{cat}:{mode}"][i])
    return out


def group_of(meta: dict[str, Any]) -> str:
    if meta["variant"] is None:
        return "clean"
    return "placebo" if meta["variant"] == "P0" else meta["outcome"]


def predict_profiles(
    runs: list[dict[str, Any]], choices: list[Choice], stock: Detector, fixed: Detector
) -> pd.DataFrame:
    rows = []
    for r in runs:
        m, ev = r["meta"], r["events"]
        alt = build_alternatives(ev, stock, fixed)
        v = m["variant"] if m["variant"] in VARIANTS else None
        for ch in choices:
            pred = attribute(apply_profile(ev, alt, ch))
            rows.append({
                "run_id": m["run_id"], "scenario_id": m["scenario_id"], "group": group_of(m),
                "profile": name_of(ch), "n_kept": sum(bool(x) for x in ch.values()),
                "true_variant": v, "true_first_error": FIRST_ERROR_STEP.get(v) if v else None,
                "pred_variant": pred["root_cause_variant"],
                "pred_first_error": pred["first_error_step"],
                "fired_rules": ",".join(pred["fired_rules"]),
            })  # fmt: skip
    return pd.DataFrame(rows)


def key_of(cat_id: str, mode: str) -> str:
    cat = CATEGORY_IDS[cat_id]
    return cat if mode == "raw" else f"{cat}:{mode}"


def size_table(choices: list[Choice], per_run: pd.DataFrame) -> pd.DataFrame:
    """Per candidate: median bytes / gzip per run, mean planted values and Presidio entities."""
    piv = {
        col: per_run.pivot(index="run_id", columns="key", values=col).fillna(0)
        for col in ("bytes", "gzip", "planted", "entities")
    }
    rows = []
    for ch in choices:
        keys = [SKELETON] + [key_of(c, m) for c, m in ch.items() if m]
        tot = {col: piv[col][keys].sum(axis=1) for col in piv}
        rows.append({
            "profile": name_of(ch),
            "bytes_median": float(tot["bytes"].median()),
            "gzip_median": float(tot["gzip"].median()),
            "planted_per_run": float(tot["planted"].mean()),
            "entities_per_run": float(tot["entities"].mean()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def select(table: pd.DataFrame) -> pd.Series:
    full = table[table.profile == FULL_NAME].iloc[0]
    ok = table[
        (table.first_error_acc >= full.first_error_acc - TOLERANCE)
        & (table.root_cause_acc >= full.root_cause_acc - TOLERANCE)
    ]
    return ok.sort_values(["bytes_median", "planted_per_run", "entities_per_run", "profile"]).iloc[
        0
    ]


def frontier(df: pd.DataFrame, acc: str) -> pd.DataFrame:
    """Non-dominated points: no other candidate is both smaller and at least as accurate."""
    d = df.sort_values(["bytes_median", acc], ascending=[True, False])
    best, keep = -1.0, []
    for _, r in d.iterrows():
        if r[acc] > best:
            keep.append(r)
            best = r[acc]
    return pd.DataFrame(keep)


def main() -> None:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())["bootstrap"]
    stock, fixed = Detector(False), Detector(True)
    design = load_side("design")
    per_run = pd.read_csv("results/costs_per_run.csv")
    per_run = per_run[per_run.side == "design"]
    choices = candidates()
    pred = predict_profiles(design, choices, stock, fixed)
    met = metrics(pred, cfg["resamples"], cfg["seed"])
    sizes = size_table(choices, per_run)
    man = met[met.group == "manifested"].merge(sizes, on="profile")
    mas = met[met.group == "masked"].merge(sizes, on="profile")
    man.to_csv("results/profile_candidates_manifested.csv", index=False)
    mas.to_csv("results/profile_candidates_masked.csv", index=False)
    best = select(man)
    full = man[man.profile == FULL_NAME].iloc[0]
    sk = man[man.profile == "skeleton"].iloc[0]
    chosen_ch = next(c for c in choices if name_of(c) == best.profile)
    on_masked = mas[mas.profile == best.profile].iloc[0]
    profile = {
        "frozen": date.today().isoformat(),
        "selected_on": "design side only; the holdout has not been read",
        "rule": "smallest median trace bytes with first-error and root-cause accuracy both within "
                "5 points of full (manifested runs); tie-break fewer planted sensitive values",
        "n_candidates": len(choices),
        "profile": best.profile,
        "categories": {k: v for k, v in chosen_ch.items() if v},
        "design_manifested": {
            "first_error_acc": best.first_error_acc,
            "root_cause_acc": best.root_cause_acc,
            "full_first_error_acc": full.first_error_acc,
            "full_root_cause_acc": full.root_cause_acc,
            "first_error_drop": best.fe_drop,
            "first_error_drop_ci": [best.fe_drop_lo, best.fe_drop_hi],
            "root_cause_drop": best.rc_drop,
            "root_cause_drop_ci": [best.rc_drop_lo, best.rc_drop_hi],
        },
        "design_masked": {
            "first_error_acc": on_masked.first_error_acc,
            "root_cause_acc": on_masked.root_cause_acc,
        },
        "median_bytes_per_run": best.bytes_median,
        "median_gzip_bytes_per_run": best.gzip_median,
        "full_median_bytes_per_run": full.bytes_median,
        "bytes_share_of_full": best.bytes_median / full.bytes_median,
        "planted_values_per_run": best.planted_per_run,
        "full_planted_values_per_run": full.planted_per_run,
        "skeleton_accuracy": {"first_error": sk.first_error_acc, "root_cause": sk.root_cause_acc},
    }  # fmt: skip
    Path("results/profile.json").write_text(json.dumps(profile, indent=1, default=float) + "\n")
    stock.save()
    fixed.save()
    print(json.dumps(profile, indent=1, default=float))
    cols = ["profile", "bytes_median", "first_error_acc", "root_cause_acc", "planted_per_run"]
    ok = man[(man.first_error_acc >= full.first_error_acc - TOLERANCE)
             & (man.root_cause_acc >= full.root_cause_acc - TOLERANCE)]  # fmt: skip
    print(ok.sort_values("bytes_median")[cols].head(12).round(3).to_string(index=False))
    # lighter variants substituted for the original category
    rows = []
    for key, name in (("C2m", "C1+C2m+C3+C4+C5+C6"), ("C2x", "C1+C2x+C3+C4+C5+C6"),
                      ("C3m", "C1+C2+C3m+C4+C5+C6"), ("C3x", "C1+C2+C3x+C4+C5+C6"),
                      ("C3l", "C1+C2+C3l+C4+C5+C6")):  # fmt: skip
        for g, t in (("manifested", man), ("masked", mas)):
            r = t[t.profile == name].iloc[0]
            f = t[t.profile == FULL_NAME].iloc[0]
            rows.append({
                "variant": key, "group": g, "first_error_acc": r.first_error_acc,
                "root_cause_acc": r.root_cause_acc, "first_error_drop": r.fe_drop,
                "first_error_drop_lo": r.fe_drop_lo, "first_error_drop_hi": r.fe_drop_hi,
                "root_cause_drop": r.rc_drop, "root_cause_drop_lo": r.rc_drop_lo,
                "root_cause_drop_hi": r.rc_drop_hi, "full_profile_bytes_median": r.bytes_median,
                "full_profile_bytes_saved_vs_full": f.bytes_median - r.bytes_median,
                "planted_per_run_full_profile": r.planted_per_run,
            })  # fmt: skip
    pd.DataFrame(rows).to_csv("results/lighter_variants.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
