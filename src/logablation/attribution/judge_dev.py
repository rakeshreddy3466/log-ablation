"""Judge development on a fixed, design-side dev set: score a prompt version, log it, nothing else.

Dev set: 2 manifested runs per variant, drawn with a fixed seed. A variant with fewer than 2
manifested design runs is topped up with its masked runs (the fault is in the trace either way).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from ..agents import guard
from ..scenarios.taxonomy import FIRST_ERROR_STEP, VARIANTS
from .ablation import load_side
from .judge import JUDGE_MODEL, JUDGE_SEED, judge_events, switch_to_judge

DEV_SET = Path("results/judge_dev_set.json")
DEV_LOG = Path("results/judge_dev_log.md")


def build_dev_set(per_variant: int = 2, seed: int = JUDGE_SEED) -> dict[str, Any]:
    runs = load_side("design")
    rng = random.Random(seed)
    picked: list[dict[str, str]] = []
    for v in VARIANTS:
        for outcome in ("manifested", "masked"):
            have = sum(p["variant"] == v for p in picked)
            pool = sorted(
                r["meta"]["run_id"]
                for r in runs
                if r["meta"]["variant"] == v and r["meta"]["outcome"] == outcome
            )
            rng.shuffle(pool)
            for rid in pool[: max(per_variant - have, 0)]:
                picked.append({"run_id": rid, "variant": v, "outcome": outcome})
    return {"seed": seed, "side": "design", "runs": picked}


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fe = sum(r["pred_first_error"] == r["true_first_error"] for r in rows)
    rc = sum(r["pred_variant"] == r["true_variant"] for r in rows)
    dist = Counter(r["pred_variant"] for r in rows)
    top, top_n = dist.most_common(1)[0]
    return {
        "n": n,
        "first_error_acc": fe / n,
        "root_cause_acc": rc / n,
        "label_distribution": dict(dist.most_common()),
        "top_label": top,
        "top_share": top_n / n,
        "valid": sum(r["valid"] for r in rows),
        "degenerate": top_n / n > 0.6 or rc / n < 0.3,
        "per_variant_correct": {
            v: sum(r["pred_variant"] == v for r in rows if r["true_variant"] == v) for v in VARIANTS
        },
        "mean_s": statistics.fmean(r["wall_s"] for r in rows),
        "tokens_max": max(r["prompt_tokens"] or 0 for r in rows),
    }


def run_dev(version: str, model: str) -> dict[str, Any]:
    dev = json.loads(DEV_SET.read_text())
    runs = {r["meta"]["run_id"]: r for r in load_side("design")}
    client = switch_to_judge(model=model, also_unload=("qwen2.5:7b", "phi4-mini:latest"))
    rows = []
    for d in dev["runs"]:
        r = runs[d["run_id"]]
        out = judge_events(client, r["events"], version=version, order_key=d["run_id"] + "|full")
        foreign = guard.check_after_run(model)
        rows.append({
            **out, "run_id": d["run_id"], "true_variant": d["variant"],
            "true_first_error": FIRST_ERROR_STEP[d["variant"]], "foreign": foreign,
        })  # fmt: skip
        print(d["variant"], "->", out["pred_first_error"], out["pred_variant"], out["wall_s"], "s")
        if foreign:
            raise SystemExit(f"foreign model resident: {foreign}")
    client.unload()
    res = score(rows)
    res |= {"version": version, "model": model, "predictions": [
        {k: r[k] for k in ("run_id", "true_variant", "pred_first_error", "pred_variant",
                           "confidence", "valid", "wall_s", "reason")} for r in rows]}  # fmt: skip
    return res


def append_log(res: dict[str, Any]) -> None:
    lines = [
        f"\n## {res['version']} on {res['model']} ({date.today().isoformat()}, dev set, full)\n",
        f"- first-error accuracy: {res['first_error_acc']:.0%} ({res['n']} runs)",
        f"- root-cause accuracy: {res['root_cause_acc']:.0%} (chance 12.5%)",
        f"- predicted label distribution: {res['label_distribution']}",
        f"- most common label: {res['top_label']} at {res['top_share']:.0%} of predictions",
        f"- correct per true variant: {res['per_variant_correct']}",
        f"- valid answers: {res['valid']}/{res['n']}; {res['mean_s']:.1f} s/call; "
        f"max prompt tokens {res['tokens_max']}",
        f"- degenerate (one label > 60% or root-cause < 30%): {res['degenerate']}",
    ]
    DEV_LOG.parent.mkdir(exist_ok=True)
    with DEV_LOG.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    Path(f"results/judge_dev_{res['version']}_{res['model'].split(':')[0]}.json").write_text(
        json.dumps(res, indent=1) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--model", default=JUDGE_MODEL)
    a = ap.parse_args()
    if a.build:
        d = build_dev_set()
        DEV_SET.write_text(json.dumps(d, indent=1) + "\n")
        print(Counter((r["variant"], r["outcome"]) for r in d["runs"]))
        return
    res = run_dev(a.version, a.model)
    append_log(res)
    print({k: v for k, v in res.items() if k != "predictions"})


if __name__ == "__main__":
    main()
