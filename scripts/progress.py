"""Progress of a running `make run`: uv run python scripts/progress.py [--watch SECONDS]."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import yaml

OUT = Path("runs/outcomes.jsonl")


def snapshot() -> str:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())["run_mix"]
    recs = []
    for line in OUT.read_text().splitlines() if OUT.exists() else []:
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    clean = [r for r in recs if r["variant"] is None]
    faulted = [r for r in recs if r["variant"]]
    n_var = len(cfg["variants"])
    target_f = cfg["per_variant"] * n_var  # upper bound; the real plan depends on eligibility
    secs = [r["timings"]["total_s"] for r in recs[-30:]]
    mean = statistics.fmean(secs) if secs else 0.0
    left = max(cfg["clean"] - len(clean), 0) + max(target_f - len(faulted), 0)
    ok = sum(r["success"] for r in clean)
    out = [
        f"clean   {len(clean):>4}/{cfg['clean']}   "
        f"clean_success {ok}, natural_failure {len(clean) - ok}",
        f"faulted {len(faulted):>4}/<= {target_f}   "
        + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["outcome"] for r in faulted).items())),
        f"mean of last {len(secs)} runs: {mean:.1f}s   remaining (upper bound): {left} runs"
        f" ~ {left * mean / 3600:.1f} h",
        f"foreign-model runs: {sum(bool(r['foreign_models']) for r in recs)}   "
        f"agent failures: {sum(r['failed_step'] is not None for r in recs)}",
    ]
    if recs:
        last = recs[-1]
        out.append(f"last: {last['run_id']}  {last['outcome']}  {last['timings']['total_s']:.0f}s")
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    out.append(next((x.strip() for x in mp.splitlines() if "free percentage" in x), ""))
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip()
    out.append("ollama ps:\n" + ps)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="refresh every N seconds")
    a = ap.parse_args()
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")
        print(snapshot(), flush=True)
        if not a.watch:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
