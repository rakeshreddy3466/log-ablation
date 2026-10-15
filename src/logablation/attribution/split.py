"""70/30 design/holdout split by scenario, stratified by variant, seed from the config.

Scenarios host several variants, so a plain per-variant split is impossible. Scenarios are
shuffled with the seed and assigned one by one to the side that keeps every variant they host
closest to the 70/30 target (greedy balancing). All runs of a scenario land on the same side.
Only scenario ids and variant names are read: no outcome, no trace.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def make_split(
    variants_by_scenario: dict[str, list[str]], design_frac: float, seed: int
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    ids = sorted(variants_by_scenario)
    rng.shuffle(ids)
    # hosts of many variants first: they constrain the balance most
    ids.sort(key=lambda s: -len(set(variants_by_scenario[s])))
    side_n: dict[str, dict[str, int]] = {"design": defaultdict(int), "holdout": defaultdict(int)}
    tot = {"design": 0, "holdout": 0}
    out: dict[str, list[str]] = {"design": [], "holdout": []}
    for s in ids:
        vs = set(variants_by_scenario[s]) or {"__none__"}

        def cost(side: str, vs: set[str] = vs) -> float:
            c = 0.0
            for v in vs:
                d = side_n["design"][v] + (side == "design")
                h = side_n["holdout"][v] + (side == "holdout")
                c += abs(d / (d + h) - design_frac)
            # also keep the overall scenario count at the target
            n = tot["design"] + (side == "design")
            c += abs(n / (tot["design"] + tot["holdout"] + 1) - design_frac) * 0.5
            return c

        side = min(("design", "holdout"), key=lambda x: (cost(x), x))
        out[side].append(s)
        tot[side] += 1
        for v in vs:
            side_n[side][v] += 1
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())["split"]
    by: dict[str, list[str]] = defaultdict(list)
    for line in Path("runs/outcomes.jsonl").read_text().splitlines():
        r = json.loads(line)
        by[r["scenario_id"]].append(r["variant"] or "clean")
    sp = make_split(
        {s: [v for v in vs if v != "clean"] for s, vs in by.items()},
        cfg["design"],
        cfg["seed"],
    )
    counts: dict[str, Any] = {}
    for side, ids in sp.items():
        c: dict[str, int] = defaultdict(int)
        for s in ids:
            for v in by[s]:
                c[v] += 1
        counts[side] = dict(sorted(c.items()))
    doc = {"seed": cfg["seed"], "design_fraction": cfg["design"], **sp, "run_counts": counts}
    Path("results/split.json").write_text(json.dumps(doc, indent=1) + "\n")
    print({k: len(v) for k, v in sp.items()})
    for side in counts:
        print(side, counts[side])


if __name__ == "__main__":
    main()
