"""Dataset results from runs/outcomes.jsonl: T1, manifestation rates, placebo baseline, figure F3.

No model calls. Every interval is a cluster bootstrap: whole scenarios are resampled, because the
runs of one scenario (clean twin and its faulted variants) are not independent.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VARIANTS = ["E1", "E2", "E3", "R1", "R2", "I1", "I2", "I3"]
PLACEBO = "P0"
TYPES = [
    "db_pool_exhausted", "bad_deploy", "memory_leak",
    "cross_zone_latency", "expired_certificate", "cache_eviction_storm",
]  # fmt: skip
N_BOOT = 2000


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    keep = ["run_id", "scenario_id", "incident_type", "variant", "outcome", "success"]
    df = pd.DataFrame([{k: r[k] for k in keep} for r in rows])
    # Placebo records written before the graph.py fix carry success=False whatever the answer;
    # the three component checks are the truth, so success is recomputed from them.
    df["success"] = [
        bool(r["root_cause_correct"] and r["action_correct"] and r["target_correct"]) for r in rows
    ]
    return df


def t1_by_type(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df.variant.isna()]
    fault = df[df.variant.isin(VARIANTS)]
    out = []
    for t in [*TYPES, "ALL"]:
        c = clean if t == "ALL" else clean[clean.incident_type == t]
        f = fault if t == "ALL" else fault[fault.incident_type == t]
        out.append({
            "incident_type": t,
            "clean_runs": len(c),
            "eligible": int((c.outcome == "clean_success").sum()),
            "faulted_runs": len(f),
            "manifested": int((f.outcome == "manifested").sum()),
            "masked": int((f.outcome == "masked").sum()),
        })  # fmt: skip
    return pd.DataFrame(out)


def _rate(rows: dict[str, tuple[int, int]], scen: list[str]) -> float:
    n = sum(rows[s][1] for s in scen if s in rows)
    return sum(rows[s][0] for s in scen if s in rows) / n if n else float("nan")


def variant_rates(df: pd.DataFrame, seed: int = 0, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Manifestation rate per variant (manifested / faulted runs) and P0 flip rate.

    The CI of the difference to the placebo uses the same scenario resample for both, so it
    respects that variants and placebo share scenarios. 'clearly above' = difference CI > 0.
    """
    rng = np.random.default_rng(seed)
    pool = sorted(df[df.variant.notna()].scenario_id.unique())
    per: dict[str, dict[str, tuple[int, int]]] = {}
    for v in [*VARIANTS, PLACEBO]:
        sub = df[df.variant == v]
        flag = (~sub.success) if v == PLACEBO else (sub.outcome == "manifested")
        g = defaultdict(lambda: [0, 0])
        for s, f in zip(sub.scenario_id, flag, strict=True):
            g[s][0] += int(f)
            g[s][1] += 1
        per[v] = {s: (a, b) for s, (a, b) in g.items()}
    idx = rng.integers(0, len(pool), size=(n_boot, len(pool)))
    # counts per scenario draw multiplicity: resample = weights per scenario
    w = np.stack([np.bincount(i, minlength=len(pool)) for i in idx])
    pos = {s: k for k, s in enumerate(pool)}

    def boot(v: str) -> np.ndarray:
        a = np.zeros(len(pool))
        b = np.zeros(len(pool))
        for s, (x, n) in per[v].items():
            a[pos[s]], b[pos[s]] = x, n
        num, den = w @ a, w @ b
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    pb = boot(PLACEBO)
    p_hat = _rate(per[PLACEBO], pool)
    rows = []
    for v in [*VARIANTS, PLACEBO]:
        bs = boot(v)
        hat = _rate(per[v], pool)
        row: dict[str, Any] = {
            "variant": v,
            "runs": sum(n for _, n in per[v].values()),
            "scenarios": len(per[v]),
            "events": sum(x for x, _ in per[v].values()),
            "rate": hat,
            "ci_lo": float(np.nanpercentile(bs, 2.5)),
            "ci_hi": float(np.nanpercentile(bs, 97.5)),
        }
        if v != PLACEBO:
            d = bs - pb
            row |= {
                "diff_vs_placebo": hat - p_hat,
                "diff_lo": float(np.nanpercentile(d, 2.5)),
                "diff_hi": float(np.nanpercentile(d, 97.5)),
            }
            row["clearly_above_placebo"] = bool(row["diff_lo"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def figure_f3(rates: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1f2328", "#59636e"
    c_man, c_mask = "#2a6fbb", "#c9d1d9"
    r = rates.set_index("variant")
    order = [*VARIANTS, PLACEBO]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    x = np.arange(len(order))
    man = np.array([r.loc[v, "rate"] for v in order])
    ax.bar(x, man, 0.62, color=c_man, label="manifested (P0: answer flipped)")
    ax.bar(x, 1 - man, 0.62, bottom=man, color=c_mask, label="masked (P0: unchanged)")
    err = np.array([[r.loc[v, "rate"] - r.loc[v, "ci_lo"], r.loc[v, "ci_hi"] - r.loc[v, "rate"]]
                    for v in order]).T  # fmt: skip
    ax.errorbar(x, man, yerr=err, fmt="none", ecolor=ink, elinewidth=1, capsize=3)
    p = r.loc[PLACEBO, "rate"]
    ax.axhline(p, color=ink, ls="--", lw=1)
    ax.text(len(order) - 0.45, p + 0.015, f"placebo flip rate {p:.0%}", ha="right", color=ink,
            fontsize=8)  # fmt: skip
    for k, v in enumerate(order):
        n, e = int(r.loc[v, "runs"]), int(r.loc[v, "events"])
        ax.text(k, 1.02, f"{e}/{n}", ha="center", fontsize=8, color=muted)
        if v != PLACEBO and r.loc[v, "clearly_above_placebo"]:
            ax.text(k, r.loc[v, "ci_hi"] + 0.03, "*", ha="center", fontsize=12, color=ink)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share of runs", color=ink)
    ax.set_title("F3  Manifested vs masked per fault variant", loc="left", color=ink)
    ax.text(0, -0.2, "Whiskers: 95% cluster-bootstrap CI (scenarios resampled). "
            "* = difference to placebo CI above zero. Top: events/runs.",
            transform=ax.transAxes, fontsize=8, color=muted)  # fmt: skip
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncols=2)  # fmt: skip
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"F3_manifested_vs_masked.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load(Path("runs/outcomes.jsonl"))
    res = Path("results")
    t1 = t1_by_type(df)
    t1.to_csv(res / "t1_by_incident_type.csv", index=False)
    rates = variant_rates(df)
    rates.to_csv(res / "manifestation_rates.csv", index=False)
    f = df[df.variant.isin(VARIANTS)]
    tv = rates[rates.variant.isin(VARIANTS)].assign(
        manifested=lambda d: d.events, masked=lambda d: d.runs - d.events
    )[["variant", "scenarios", "runs", "manifested", "masked"]]  # fmt: skip
    tv.to_csv(res / "t1_by_variant.csv", index=False)
    figure_f3(rates, Path("figures"))
    pd.set_option("display.width", 200)
    print(t1.to_string(index=False))
    print(tv.to_string(index=False))
    print(rates.round(3).to_string(index=False))
    print("natural_failure:", int((df.outcome == "natural_failure").sum()), "faulted:", len(f))


if __name__ == "__main__":
    main()
