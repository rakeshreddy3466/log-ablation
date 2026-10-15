"""Holdout scoring of the frozen profile: full, profile and skeleton, manifested and masked.

Run only after results/profile.json has been committed. The profile is read from that file and
nothing is tuned here. Cluster bootstrap over scenarios (2,000 resamples, seed from the config).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..attribution.ablation import load_side, metrics
from .privacy import Detector
from .profile import FULL, FULL_NAME, TOLERANCE, Choice, name_of, predict_profiles

CONDITIONS = ("full", "profile", "skeleton")


def accuracy_cis(pred: pd.DataFrame, names: dict[str, str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in ("manifested", "masked"):
        g = pred[pred.group == group]
        scen = sorted(g.scenario_id.unique())
        pos = {s: i for i, s in enumerate(scen)}
        w = np.stack([np.bincount(rng.integers(0, len(scen), len(scen)), minlength=len(scen))
                      for _ in range(n_boot)]).astype(float)  # fmt: skip
        for cond, prof in names.items():
            p = g[g.profile == prof].sort_values("run_id")
            idx = p.scenario_id.map(pos).to_numpy()
            n = np.bincount(idx, minlength=len(scen)).astype(float)
            row: dict[str, Any] = {"group": group, "condition": cond, "profile": prof,
                                   "n_runs": len(p), "n_scenarios": len(scen)}  # fmt: skip
            for key, col_ok in (
                ("first_error", p.pred_first_error == p.true_first_error),
                ("root_cause", p.pred_variant == p.true_variant),
            ):
                ok = np.bincount(idx, weights=col_ok.to_numpy().astype(float), minlength=len(scen))
                b = (w @ ok) / (w @ n)
                row[f"{key}_acc"] = float(col_ok.mean())
                row[f"{key}_lo"], row[f"{key}_hi"] = (
                    float(x) for x in np.percentile(b, [2.5, 97.5])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())["bootstrap"]
    prof = json.loads(Path("results/profile.json").read_text())
    chosen: Choice = {c: None for c in FULL} | prof["categories"]
    assert name_of(chosen) == prof["profile"]
    skeleton: Choice = {c: None for c in FULL}
    names = {"full": FULL_NAME, "profile": prof["profile"], "skeleton": "skeleton"}
    holdout = load_side("holdout")
    pred = predict_profiles(holdout, [FULL, chosen, skeleton], Detector(False), Detector(True))
    ci = accuracy_cis(pred, names, cfg["resamples"], cfg["seed"])
    drops = metrics(pred, cfg["resamples"], cfg["seed"])
    rows = []
    for _, r in ci.iterrows():
        d = drops[(drops.group == r.group) & (drops.profile == r.profile)].iloc[0]
        rows.append({
            **r.to_dict(),
            "first_error_drop": d.fe_drop, "first_error_drop_lo": d.fe_drop_lo,
            "first_error_drop_hi": d.fe_drop_hi, "root_cause_drop": d.rc_drop,
            "root_cause_drop_lo": d.rc_drop_lo, "root_cause_drop_hi": d.rc_drop_hi,
            "first_error_macro_f1": d.first_error_macro_f1,
            "root_cause_macro_f1": d.root_cause_macro_f1,
            "unknown_rate": d.unknown_rate,
        })  # fmt: skip
    t5 = pd.DataFrame(rows)
    per_run = pd.read_csv("results/costs_per_run.csv")
    per_run = per_run[per_run.side == "holdout"]
    sizes = {}
    for cond, ch in (("full", FULL), ("profile", chosen), ("skeleton", skeleton)):
        from .profile import size_table

        s = size_table([ch], per_run).iloc[0]
        sizes[cond] = (s.bytes_median, s.planted_per_run)
    t5["median_bytes_per_run"] = t5.condition.map(lambda c: sizes[c][0])
    t5["planted_values_per_run"] = t5.condition.map(lambda c: sizes[c][1])
    man = t5[t5.group == "manifested"].set_index("condition")
    verdict = {
        "profile": prof["profile"],
        "tolerance_points": TOLERANCE * 100,
        "manifested": {
            "first_error_drop": float(man.loc["profile", "first_error_drop"]),
            "root_cause_drop": float(man.loc["profile", "root_cause_drop"]),
        },
    }
    verdict["passes"] = bool(
        verdict["manifested"]["first_error_drop"] <= TOLERANCE
        and verdict["manifested"]["root_cause_drop"] <= TOLERANCE
    )
    t5.to_csv("results/holdout.csv", index=False)
    pred.to_parquet("attribution/predictions_holdout_rules.parquet", index=False)
    Path("results/holdout_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t5.round(3).to_string(index=False))
    print(json.dumps(verdict, indent=1))
    miss = pred[(pred.profile == prof["profile"]) & pred.group.isin(["manifested", "masked"])]
    miss = miss[miss.pred_variant != miss.true_variant]
    print(miss[["group", "true_variant", "pred_variant", "fired_rules"]].value_counts().to_string())


if __name__ == "__main__":
    main()
