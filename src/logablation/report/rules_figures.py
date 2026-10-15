"""F4 and F6 for the rule-based checker, drawn from results/rules_ablation.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..scenarios.taxonomy import VARIANTS  # noqa: E402
from ..tracing.categories import ABLATABLE, CATEGORY_IDS  # noqa: E402
from .style import save  # noqa: E402

INK, MUTED = "#1f2328", "#59636e"
BLUE, ORANGE = "#2a6fbb", "#d98b2b"
IDS = list(CATEGORY_IDS)  # C1..C6
LABEL = {k: f"{k} {v}" for k, v in CATEGORY_IDS.items()}


def _remove_one(df: pd.DataFrame, group: str, cat_id: str) -> pd.Series:
    kept = "+".join(c for c in IDS if c != cat_id)
    return df[(df.group == group) & (df.profile == kept)].iloc[0]


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    save(fig, name, out_dir)


def figure_f4(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150, sharey=True)
    y = np.arange(len(IDS))
    h = 0.36
    for ax, (key, title) in zip(
        axes, (("fe", "First-error accuracy"), ("rc", "Root-cause accuracy")), strict=True
    ):
        for off, group, col in ((-h / 2, "manifested", BLUE), (h / 2, "masked", ORANGE)):
            rows = [_remove_one(df, group, c) for c in IDS]
            d = np.array([r[f"{key}_drop"] for r in rows])
            lo = np.array([r[f"{key}_drop_lo"] for r in rows])
            hi = np.array([r[f"{key}_drop_hi"] for r in rows])
            ax.barh(y + off, d, h * 0.92, color=col, label=group)
            ax.errorbar(d, y + off, xerr=[np.maximum(d - lo, 0), np.maximum(hi - d, 0)],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)  # fmt: skip
            for yi, lo_i, hi_i in zip(y + off, lo, hi, strict=True):
                if lo_i <= 0 <= hi_i:
                    ax.text(max(hi_i, 0) + 0.01, yi, "no measurable value", va="center",
                            fontsize=6.5, color=MUTED)  # fmt: skip
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlim(0, 0.75)
        ax.set_xlabel("accuracy drop when the category is removed", color=INK, fontsize=9)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y, [LABEL[c] for c in IDS])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("F4  Rule-based checker: accuracy drop per removed category (design side)",
                 x=0.01, ha="left", color=INK, fontsize=11)  # fmt: skip
    fig.text(0.01, -0.03, "Whiskers: 95% paired cluster-bootstrap CI over scenarios. "
             "C1 and C6 substitute for each other (interface rules): removing one costs nothing.",
             fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout()
    _save(fig, out_dir, "F4_ablation_drops")


def figure_f6(df: pd.DataFrame, out_dir: Path) -> None:
    drop = np.full((len(IDS), len(VARIANTS)), np.nan)
    sig = np.zeros_like(drop, dtype=bool)
    for i, c in enumerate(IDS):
        r = _remove_one(df, "manifested", c)
        for j, v in enumerate(VARIANTS):
            drop[i, j] = r[f"recall_drop_{v}"]
            lo, hi = r[f"recall_drop_{v}_lo"], r[f"recall_drop_{v}_hi"]
            sig[i, j] = bool(lo > 0 or hi < 0)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    im = ax.imshow(drop, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(IDS)):
        for j in range(len(VARIANTS)):
            v = drop[i, j]
            if np.isnan(v):
                txt, col = "n/a", MUTED
            else:
                txt = f"{v:.0%}" + ("*" if sig[i, j] else "")
                col = "white" if v > 0.55 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
    ax.set_xticks(range(len(VARIANTS)), VARIANTS)
    ax.set_yticks(range(len(IDS)), [LABEL[c] for c in IDS])
    ax.set_title("F6  Recall drop when a category is removed, per fault variant "
                 "(rule-based, manifested, design)",
                 loc="left", color=INK, fontsize=10)  # fmt: skip
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("recall drop vs full", color=INK, fontsize=8)
    fig.text(0.01, -0.02, "* = 95% cluster-bootstrap CI of the drop excludes zero. "
             "Few runs per variant (1-14): direction, not size. Negative = removing C2 also "
             "removes false E3 firings that outrank the true rule (colour clipped at 0).",
             fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout()
    _save(fig, out_dir, "F6_category_variant_heatmap")


def main() -> None:
    df = pd.read_csv("results/rules_ablation.csv")
    assert set(IDS) == {k for k in CATEGORY_IDS} and len(ABLATABLE) == 6
    figure_f4(df, Path("figures"))
    figure_f6(df, Path("figures"))


if __name__ == "__main__":
    main()
