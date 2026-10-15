"""F8: confusion matrices, full logging vs the frozen profile (design side, manifested runs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .style import INK, MUTED, plt, save

VARIANT_ORDER = ["E1", "E2", "E3", "R1", "R2", "I1", "I2", "I3", "unknown"]
STEP_ORDER = ["evidence_agent", "handover_1", "reasoning_agent", "handover_2", "unknown"]
FULL = "C1+C2+C3+C4+C5+C6"


def matrix(df: pd.DataFrame, true_col: str, pred_col: str, labels: list[str]) -> np.ndarray:
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(df[true_col], df[pred_col], strict=True):
        if t in labels and p in labels:
            m[labels.index(t), labels.index(p)] += 1
    return m


def figure_f8(out_dir: Path) -> None:
    pred = pd.read_parquet("attribution/predictions_profile_design.parquet")
    prof = json.loads(Path("results/profile.json").read_text())["profile"]
    man = pred[pred.group == "manifested"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    spec = [("true_variant", "pred_variant", VARIANT_ORDER, "root-cause variant"),
            ("true_first_error", "pred_first_error", STEP_ORDER, "first error step")]  # fmt: skip
    for r, (tcol, pcol, labels, title) in enumerate(spec):
        for c, (profile, label) in enumerate(((FULL, "full logging"), (prof, f"profile {prof}"))):
            ax = axes[r][c]
            sub = man[man.profile == profile]
            m = matrix(sub, tcol, pcol, labels)
            ax.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1))
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if m[i, j]:
                        ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                                color="white" if m[i, j] > m.max() * 0.6 else INK)  # fmt: skip
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ax.set_xlabel("predicted", fontsize=9, color=INK)
            ax.set_ylabel("true", fontsize=9, color=INK)
            acc = float((sub[tcol] == sub[pcol]).mean())
            ax.set_title(f"{title}: {label}\naccuracy {acc:.1%}, n={len(sub)}", loc="left",
                         fontsize=9, color=INK)  # fmt: skip
    fig.suptitle(
        "F8  Confusion matrices, full logging vs frozen profile (rule checker, "
        "design side, manifested runs)",
        x=0.01, ha="left", color=INK, fontsize=11,
    )  # fmt: skip
    fig.text(0.01, -0.01, "Counts of runs. 'unknown' = no rule fired.", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "F8_confusion_matrices", out_dir)
