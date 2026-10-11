"""Render the README chart (light + dark) and the GitHub social preview from results/workflow/results.json.

    pip install matplotlib
    python docs/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
summary = json.loads((ROOT / "results/workflow/results.json").read_text(encoding="utf-8"))["summary"]

THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984",
              "grid": "#e6e5e1", "clef": "#2a78d6", "agent": "#eb6834"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#8f8e86",
             "grid": "#33332f", "clef": "#3987e5", "agent": "#d95926"},
}
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]

ROWS = [("clef", "clef-routed graph"), ("agent", "tool-calling agent")]
correct = {k: int(summary[k]["correct"].split("/")[0]) for k, _ in ROWS}
total = int(summary["clef"]["correct"].split("/")[1])
PANELS = [
    ("Tool requests · median latency", {k: summary[k]["tool_median_s"] for k, _ in ROWS}, "{:.2f} s", None),
    ("Open-ended requests · median latency", {k: summary[k]["open_median_s"] for k, _ in ROWS}, "{:.2f} s", None),
    (f"Correct runs (of {total})", correct, "{:d}", total),
]


def rounded_bar(ax, y, width, height, color, surface):
    """Horizontal bar, square at the baseline, 4px-ish rounded data end."""
    radius = min(0.12, height / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), max(width, 1e-6), height,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                mutation_aspect=1, linewidth=0, facecolor=color))
    if width > radius:  # square off the baseline end
        ax.add_patch(plt.Rectangle((0, y - height / 2), radius, height, linewidth=0, facecolor=color))


def chart(theme_name):
    t = THEMES[theme_name]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=200, gridspec_kw={"wspace": 0.55})
    fig.patch.set_facecolor(t["surface"])
    fig.text(0.012, 0.95, "Clef-routed LangGraph workflow vs. tool-calling agent",
             fontsize=15, fontweight="bold", color=t["primary"], va="top")
    fig.text(0.012, 0.86, "15 Turkish brokerage requests × 3 runs · Qwen3.5-9B for both LLM steps · "
             "clef-flash router · 2× RTX A5000, end-to-end over LAN",
             fontsize=9.5, color=t["secondary"], va="top")
    for ax, (title, values, fmt, xmax) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        limit = xmax or max(values.values()) * 1.32
        for index, (key, label) in enumerate(ROWS):
            y = 1 - index
            rounded_bar(ax, y, values[key], 0.5, t[key], t["surface"])
            ax.text(values[key] + limit * 0.02, y, fmt.format(values[key]),
                    va="center", ha="left", fontsize=11, fontweight="bold", color=t["primary"])
        ax.set_xlim(0, limit * 1.18 if xmax else limit)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([1, 0], [label for _, label in ROWS], fontsize=9.5, color=t["secondary"])
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", colors=t["muted"], labelsize=8, length=0)
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.set_title(title, loc="left", fontsize=10.5, color=t["primary"], pad=8)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.70, bottom=0.12)
    out = ROOT / f"docs/results_{theme_name}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def social_preview():
    t = THEMES["light"]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    speedup = summary["agent"]["tool_median_s"] / summary["clef"]["tool_median_s"]
    fig.text(0.06, 0.86, "clef-langchain-bench", fontsize=40, fontweight="bold", color=t["primary"], va="top")
    fig.text(0.06, 0.72, "A local decision model as the router of a LangGraph workflow,\n"
             "compared with a tool-calling agent", fontsize=20, color=t["secondary"], va="top", linespacing=1.4)
    stats = [
        (f"{speedup:.0f}×", "faster on tool requests", f"{summary['clef']['tool_median_s']:.2f} s vs "
         f"{summary['agent']['tool_median_s']:.2f} s median"),
        (f"{correct['clef']}/{total}", "correct, clef route", f"agent: {correct['agent']}/{total}, "
         "invented 2 prices"),
        ("2× A5000", "fully local", "clef-flash + Qwen3.5-9B"),
    ]
    for index, (big, label, note) in enumerate(stats):
        x = 0.06 + index * 0.31
        fig.text(x, 0.40, big, fontsize=46, fontweight="bold", color=t["clef"] if index < 2 else t["primary"])
        fig.text(x, 0.31, label, fontsize=16, color=t["primary"])
        fig.text(x, 0.25, note, fontsize=13, color=t["secondary"])
    fig.text(0.06, 0.08, "LangChain · LangGraph · Cloudflare Clef · Jev/SystemOne API", fontsize=13, color=t["muted"])
    out = ROOT / "docs/social_preview.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (chart("light"), chart("dark"), social_preview()):
        print(path.relative_to(ROOT))
