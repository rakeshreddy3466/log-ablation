"""`make report`: regenerate every figure (F1-F9) and table (T1-T5) from the committed result
files. No run records, no scenarios and no model are needed."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analysis.dataset import figure_f3
from . import confusion, diagrams, holdout_figure, profile_figures, rules_figures, tables
from .style import FIGURES, TABLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=Path, default=FIGURES)
    ap.add_argument("--tables", type=Path, default=TABLES)
    a = ap.parse_args()
    diagrams.figure_f1(a.figures)
    diagrams.figure_f2(a.figures)
    figure_f3(pd.read_csv("results/manifestation_rates.csv"), a.figures)
    df = pd.read_csv("results/rules_ablation.csv")
    rules_figures.figure_f4(df, a.figures)
    profile_figures.figure_f5(a.figures)
    rules_figures.figure_f6(df, a.figures)
    profile_figures.figure_f7(a.figures)
    confusion.figure_f8(a.figures)
    holdout_figure.main(a.figures)
    tables.main(Path("results"), a.tables)
    print(f"figures -> {a.figures}, tables -> {a.tables}")


if __name__ == "__main__":
    main()
