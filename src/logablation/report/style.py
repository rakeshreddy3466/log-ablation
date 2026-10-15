"""Shared figure and table output, so that regenerating gives byte-identical files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "logablation"  # stable element ids in SVG

INK, MUTED, BLUE, ORANGE, GREY, RED = (
    "#1f2328", "#59636e", "#2a6fbb", "#d98b2b", "#b8c1cc", "#b3261e",
)  # fmt: skip
FIGURES = Path("figures")
TABLES = Path("results/tables")


def save(fig: plt.Figure, name: str, out_dir: Path = FIGURES) -> None:
    """PNG and SVG without volatile metadata (no date, no software string)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def md_table(rows: list[dict[str, object]], float_fmt: str = "{:.3g}") -> str:
    """A GitHub-flavoured markdown table from a list of dicts (same keys in every row)."""
    if not rows:
        return ""
    cols = list(rows[0])

    def cell(v: object) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float):
            return f"{v:,.0f}" if abs(v) >= 1000 else float_fmt.format(v)
        return str(v)

    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(cell(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"
