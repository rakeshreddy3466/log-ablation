"""`python -m logablation.scenarios` -- write scenarios to data/scenarios/."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .generator import generate_many


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    p.add_argument("--base-seed", type=int)
    p.add_argument("--count", type=int)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    cfg = yaml.safe_load(a.config.read_text())["scenarios"] if a.config.exists() else {}
    base_seed = a.base_seed if a.base_seed is not None else cfg.get("base_seed", 1000)
    count = a.count if a.count is not None else cfg.get("count", 300)
    out = a.out or Path(cfg.get("out_dir", "data/scenarios"))
    out.mkdir(parents=True, exist_ok=True)

    scenarios = generate_many(base_seed, count)
    for sc in scenarios:
        (out / f"{sc.scenario_id}.json").write_text(sc.model_dump_json() + "\n")
    print(f"wrote {len(scenarios)} scenarios to {out}")


if __name__ == "__main__":
    main()
