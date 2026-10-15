"""F5 (value vs cost) and F7 (profile frontier) from the design-side csv files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..analysis.profile import FULL_NAME, frontier  # noqa: E402
from ..tracing.categories import CATEGORY_IDS  # noqa: E402

INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#59636e", "#2a6fbb", "#d98b2b", "#b8c1cc"
IDS = list(CATEGORY_IDS)


def _save(fig: plt.Figure, name: str) -> None:
    Path("figures").mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def figure_f5() -> None:
    abl = pd.read_csv("results/rules_ablation.csv")
    abl = abl[abl.group == "manifested"]
    per_run = pd.read_csv("results/costs_per_run.csv")
    per_run = per_run[per_run.side == "design"]
    cost = per_run.groupby("key").agg(bytes=("bytes", "median"), planted=("planted", "mean"))
    light = pd.read_csv("results/lighter_variants.csv")
    light = light[light.group == "manifested"].set_index("variant")
    vkey = {"C2m": "tool_calls:masked", "C2x": "tool_calls:masked_x", "C3m": "prompt_shown:masked",
            "C3x": "prompt_shown:masked_x", "C3l": "prompt_shown:light"}  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharex=True)
    for ax, (k, title) in zip(
        axes, (("fe", "First-error accuracy drop"), ("rc", "Root-cause accuracy drop")), strict=True
    ):
        for cid in IDS:
            cat = CATEGORY_IDS[cid]
            kept = "+".join(c for c in IDS if c != cid)
            drop = float(abl[abl.profile == kept][f"{k}_drop"].iloc[0])
            x, size = float(cost.loc[cat, "bytes"]), float(cost.loc[cat, "planted"])
            ax.scatter(x, drop, s=40 + size * 60, color=BLUE, alpha=0.85, zorder=3)
            ax.annotate(cid, (x, drop), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=INK)  # fmt: skip
        pts: dict[tuple[float, float], list[str]] = {}
        for name, key in vkey.items():
            r = light.loc[name]
            drop = float(r["first_error_drop" if k == "fe" else "root_cause_drop"])
            x, size = float(cost.loc[key, "bytes"]), float(cost.loc[key, "planted"])
            ax.scatter(x, drop, s=40 + size * 60, facecolor="none", edgecolor=ORANGE,
                       linewidth=1.6, zorder=3)  # fmt: skip
            pts.setdefault((round(x, -2), round(drop, 3)), []).append(name)
        for (x, drop), names in pts.items():
            ax.annotate(" ".join(names), (x, drop), xytext=(6, -13), textcoords="offset points",
                        fontsize=8, color=ORANGE)  # fmt: skip
        ax.set_xscale("log")
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("median bytes per run (log scale)", color=INK, fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylim(-0.06, 0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy drop", color=INK, fontsize=9)
    fig.suptitle("F5  Value vs cost per log category (rule checker, manifested, design)",
                 x=0.01, ha="left", color=INK, fontsize=11)  # fmt: skip
    fig.text(0.01, -0.03, "Filled = category removed. Hollow orange = lighter variant used "
             "instead of the raw category (drop = what the lighter form costs). Bubble area = "
             "planted sensitive values per run.", fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout()
    _save(fig, "F5_value_vs_cost")


def figure_f7() -> None:
    df = pd.read_csv("results/profile_candidates_manifested.csv")
    prof = __import__("json").loads(Path("results/profile.json").read_text())
    full = df[df.profile == FULL_NAME].iloc[0]
    chosen = df[df.profile == prof["profile"]].iloc[0]
    sk = df[df.profile == "skeleton"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, (acc, title) in zip(
        axes,
        (("first_error_acc", "First-error accuracy"), ("root_cause_acc", "Root-cause accuracy")),
        strict=True,
    ):
        raw = df[~df.profile.str.contains(r"C[23][mxl]")]
        lighter = df[df.profile.str.contains(r"C[23][mxl]")]
        ax.scatter(
            raw.bytes_median, raw[acc], s=22, color=BLUE, alpha=0.8, label="raw categories (64)"
        )
        ax.scatter(lighter.bytes_median, lighter[acc], s=22, facecolor="none", edgecolor=ORANGE,
                   alpha=0.9, label="with a lighter C2/C3 (256)")  # fmt: skip
        f = frontier(df, acc)
        ax.step(f.bytes_median, f[acc], where="post", color=INK, lw=1.2, label="frontier")
        ax.axhline(full[acc] - 0.05, color=MUTED, ls="--", lw=1)
        ax.text(df.bytes_median.max(), full[acc] - 0.05 - 0.04, "full minus 5 points",
                ha="right", fontsize=8, color=MUTED)  # fmt: skip
        for r, lab, col in (
            (full, "full", INK),
            (chosen, prof["profile"], "#b3261e"),
            (sk, "skeleton", INK),
        ):
            ax.scatter(r.bytes_median, r[acc], s=90, facecolor="none", edgecolor=col, linewidth=2)
            ax.annotate(lab, (r.bytes_median, r[acc]), xytext=(6, 8), textcoords="offset points",
                        fontsize=8, color=col)  # fmt: skip
        ax.set_xlabel("median bytes per run", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("F7  All 320 candidate profiles: size vs accuracy (rule checker, design)",
                 x=0.01, ha="left", color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout()
    _save(fig, "F7_profile_frontier")


def main() -> None:
    figure_f5()
    figure_f7()
    _ = np


if __name__ == "__main__":
    main()
