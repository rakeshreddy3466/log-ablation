"""F9: holdout accuracy of full / frozen profile / skeleton with 95% cluster-bootstrap CIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED = "#1f2328", "#59636e"
COL = {"full": "#2a6fbb", "profile": "#d98b2b", "skeleton": "#b8c1cc"}


def main() -> None:
    df = pd.read_csv("results/holdout.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150, sharey=True)
    for ax, group in zip(axes, ("manifested", "masked"), strict=True):
        g = df[df.group == group].set_index("condition")
        x = np.arange(2)
        w = 0.26
        for k, cond in enumerate(("full", "profile", "skeleton")):
            vals = [g.loc[cond, "first_error_acc"], g.loc[cond, "root_cause_acc"]]
            lo = [g.loc[cond, "first_error_lo"], g.loc[cond, "root_cause_lo"]]
            hi = [g.loc[cond, "first_error_hi"], g.loc[cond, "root_cause_hi"]]
            xs = x + (k - 1) * w
            ax.bar(xs, vals, w * 0.92, color=COL[cond], label=cond)
            ax.errorbar(xs, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none",
                        ecolor=INK, elinewidth=1, capsize=3)  # fmt: skip
            for xi, v in zip(xs, vals, strict=True):
                ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=INK)
        n = int(g.loc["full", "n_runs"])
        ax.set_xticks(x, ["first-error", "root-cause"])
        ax.set_title(f"{group} runs (n={n}, holdout)", loc="left", color=INK, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", ncols=3)
    fig.suptitle("F9  Holdout: full vs frozen profile vs skeleton (rule checker)",
                 x=0.01, ha="left", color=INK, fontsize=11)  # fmt: skip
    fig.text(0.01, -0.03, "Whiskers: 95% cluster-bootstrap CI over scenarios.", fontsize=8,
             color=MUTED)  # fmt: skip
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"figures/F9_holdout.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    main()
