"""F1 architecture and F2 fault map, drawn with matplotlib patches (no external tool)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .style import BLUE, INK, MUTED, RED, save


def _box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, size=8.5, bold=False):  # type: ignore[no-untyped-def]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal")  # fmt: skip


def _arrow(ax, x0, y0, x1, y1, color=INK, style="-|>", ls="-"):  # type: ignore[no-untyped-def]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": style, "color": color, "lw": 1.1, "ls": ls})  # fmt: skip


def figure_f1(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7), dpi=150)
    ax.set_xlim(0, 16.6)
    ax.set_ylim(-0.2, 9.1)
    ax.axis("off")
    names = [
        "Scenario\n(alerts, metrics,\ndeploys, logs,\ntopology)",
        "Mock tools\nget_alerts\nget_metrics\nget_deploys\nget_logs\nget_topology",
        "Evidence\nagent",
        "Handover 1",
        "Reasoning\nagent",
        "Handover 2",
        "Action\nagent",
        "Outcome check\n(deterministic,\nnever traced)",
    ]
    kinds = ["data", "data", "agent", "hand", "agent", "hand", "agent", "check"]
    fills = {"data": "#eef2f6", "agent": "#dbe8f7", "hand": "#fdebd2", "check": "#e6e6e6"}
    xs = [0.3 + i * 2.05 for i in range(8)]
    y0, h = 4.7, 1.7
    for x, n, k in zip(xs, names, kinds, strict=True):
        _box(ax, x, y0, 1.75, h, n, fc=fills[k], bold=k == "agent")
    for a_, b_ in zip(xs[:-1], xs[1:], strict=True):
        _arrow(ax, a_ + 1.75, y0 + h / 2, b_, y0 + h / 2)
    inject = {
        1: "E1 stale_window\nE2 wrong_service\nE3 missing_alert\nP0 placebo\n(tool layer)",
        3: "I1 dropped_field\nI2 truncated_findings\n(handover 1 hook)",
        4: "R1 misleading_prior\nR2 unstable_sampling\n(prompt / model settings)",
        5: "I3 dropped_target\n(handover 2 hook)",
    }
    for i, txt in inject.items():
        ax.text(xs[i] + 0.87, 8.35, txt, ha="center", va="top", fontsize=7.6, color=RED)
        top = 7.55 if i in (3, 5) else 7.2
        _arrow(ax, xs[i] + 0.87, top, xs[i] + 0.87, y0 + h + 0.03, color=RED)
    ax.text(0.3, 9.0, "Fault injection (at most one per run, via hooks; agent code never changes)",
            fontsize=9, color=RED, fontweight="bold", va="top")  # fmt: skip
    # tracer and categories
    _box(ax, 2.3, 2.95, 12.3, 0.8,
         "Tracer: every event written once as JSON (run_id, step, span_id, category, timestamp, "
         "payload); categories switched off by filtering afterwards", fc="#eaf5ea")  # fmt: skip
    for i in (1, 2, 3, 4, 5, 6):
        _arrow(ax, xs[i] + 0.87, y0, xs[i] + 0.87, 3.75, color=MUTED, ls="--")
    cats = ["skeleton\n(never removed)", "C1 step_io", "C2 tool_calls", "C3 prompt_shown",
            "C4 confidence", "C5 timings", "C6 handover_state"]  # fmt: skip
    cx = [0.3 + i * 2.3 for i in range(7)]
    for x, c in zip(cx, cats, strict=True):
        _box(ax, x, 1.95, 2.05, 0.85, c, fc="#ffffff", ec=BLUE)
        _arrow(ax, x + 1.02, 2.95, x + 1.02, 2.8, color=BLUE)
        ax.plot([x + 1.02, x + 1.02], [1.95, 1.7], color=INK, lw=1.1)
    ax.plot([cx[0] + 1.02, cx[-1] + 1.02], [1.7, 1.7], color=INK, lw=1.1)
    # attribution
    _box(ax, 0.3, 0.1, 3.8, 1.0, "Ablation filter\nfull, minus one, add one,\nall 64 profiles, "
         "skeleton", fc="#f3ecfa")  # fmt: skip
    _box(ax, 4.9, 0.1, 4.4, 1.0, "Rule checker (reference ceiling)\nused for all results",
         fc="#dbe8f7", bold=True)  # fmt: skip
    _box(ax, 10.0, 0.1, 3.0, 1.0, "LLM judge (qwen2.5:7b)\nnegative result", fc="#eeeeee")
    _box(ax, 13.5, 0.1, 2.8, 1.0, "first_error_step\nroot_cause_variant", fc="#fdebd2")
    for x0, x1 in ((4.1, 4.9), (9.3, 10.0), (13.0, 13.5)):
        _arrow(ax, x0, 0.6, x1, 0.6)
    _arrow(ax, 2.2, 1.7, 2.2, 1.1)
    ax.set_title("F1  Architecture: LangGraph pipeline (3 agents, qwen2.5:3b, temperature 0, explicit "
                 "handovers), fault injection points, tracer, attribution",
                 loc="left", color=INK, fontsize=11)  # fmt: skip
    save(fig, "F1_architecture", out_dir)


FAULTS = [
    ("E1", "stale_window", "evidence", "tool layer", "get_metrics returns a time window 30 min earlier", "evidence_agent", "C2"),
    ("E2", "wrong_service", "evidence", "tool layer", "metrics and logs of the root-cause service come back as a neighbour's", "evidence_agent", "C2"),
    ("E3", "missing_alert", "evidence", "tool layer", "the decisive alert is removed from get_alerts", "evidence_agent", "C2"),
    ("R1", "misleading_prior", "reasoning", "reasoning prompt", "a note that most past incidents were another type is appended", "reasoning_agent", "C3"),
    ("R2", "unstable_sampling", "reasoning", "model settings", "temperature 1.3 on the reasoning call", "reasoning_agent", "C3 (options)"),
    ("I1", "dropped_field", "interface", "handover 1", "affected_service removed from the evidence bundle", "handover_1", "C6 or C1"),
    ("I2", "truncated_findings", "interface", "handover 1", "only the first two findings are passed on", "handover_1", "C6 or C1"),
    ("I3", "dropped_target", "interface", "handover 2", "suspected_service removed from the diagnosis", "handover_2", "C6 or C1"),
    ("P0", "placebo", "none", "tool layer", "one routine INFO line added to a bystander service's logs", "none (no fault)", "none"),
]  # fmt: skip


def figure_f2(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=150)
    ax.axis("off")
    cols = ["ID", "variant", "class", "injected at", "effect", "first error step", "revealed by"]
    xs = [0.0, 0.05, 0.19, 0.30, 0.43, 0.80, 0.91]
    colour = {
        "evidence": "#dbe8f7",
        "reasoning": "#fdebd2",
        "interface": "#e8f3e4",
        "none": "#eeeeee",
    }
    top = 0.92
    for x, c in zip(xs, cols, strict=True):
        ax.text(x, top, c, fontsize=9, fontweight="bold", color=INK, transform=ax.transAxes)
    row_h = 0.095
    for i, row in enumerate(FAULTS):
        y = top - 0.03 - (i + 1) * row_h
        ax.add_patch(plt.Rectangle((-0.01, y - 0.015), 1.02, row_h * 0.92,
                                   transform=ax.transAxes, fc=colour[row[2]], ec="none"))  # fmt: skip
        for x, v in zip(xs, row, strict=True):
            ax.text(x, y + row_h * 0.3, v, fontsize=8.3, color=INK, transform=ax.transAxes)
    ax.set_title("F2  Fault map: where each fault enters the pipeline and which log shows it",
                 loc="left", color=INK, fontsize=11)  # fmt: skip
    ax.text(0, -0.02, "'Revealed by' is the category the rule checker needs on the design side "
            "(F6). C1 and C6 are substitutes for the interface faults.", fontsize=8, color=MUTED,
            transform=ax.transAxes)  # fmt: skip
    save(fig, "F2_fault_map", out_dir)
