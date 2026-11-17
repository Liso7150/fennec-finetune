"""Plot the frozen holdout retention verdict, never overwrite a historical graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

BG = "#15120F"
PANEL = "#1B1714"
BORDER = "#3C322A"
TEXT = "#F2EEE8"
MUTED = "#A99D91"
ACCENT = "#E9865D"
GREEN = "#5EC98F"
RED = "#EF615C"


def loss_curve(results):
    steps, losses, avg = [], [], []
    path = results / "airllm_train.log"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"step (\d+) epoch \d+ loss=([\d.]+) avg\d+=([\d.]+)", line)
            if match:
                steps.append(int(match[1]))
                losses.append(float(match[2]))
                avg.append(float(match[3]))
    return steps, losses, avg


def retention_title(verdict):
    if verdict["regressions"]:
        return "RETENTION REGRESSION - keep original", RED
    if verdict["recommendation"] == "keep_original":
        return "TIE - no proven improvement; no change", ACCENT
    return "RETENTION SCORE INCREASE ONLY - no promotion; keep original", GREEN


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True,
                    help="run directory holding holdout_{qwen36,finetune}.json and finetune_verdict.json")
    args = ap.parse_args(argv)
    results = Path(args.results_dir).resolve()
    out = results / "bench_finetune_comparison.png"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing graph: {out}")
    verdict = json.loads((results / "finetune_verdict.json").read_text(encoding="utf-8"))
    o, c = verdict["original"]["aggregate"], verdict["candidate"]["aggregate"]
    title, color = retention_title(verdict)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2A231D", "font.size": 10.5, "axes.titlesize": 12,
    })
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                          left=0.06, right=0.97, top=0.88, bottom=0.09)
    fig.suptitle("Qwen3.6-35B-A3B: AirLLM LoRA holdout retention",
                 fontsize=13.5, color=TEXT, y=0.97)
    fig.text(0.5, 0.925, title, ha="center", fontsize=12.5, color=color, weight="bold")

    ax = fig.add_subplot(gs[0, 0])
    x, width = range(2), 0.36
    bars = [ax.bar([i - width / 2 for i in x], [o["passed"], o["accuracy"] * 100], width,
                   color=MUTED, label="original"),
            ax.bar([i + width / 2 for i in x], [c["passed"], c["accuracy"] * 100], width,
                   color=ACCENT, label="fine-tuned candidate")]
    for group, fmt in zip(bars, ("{:.0f}", "{:.1f}%")):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    fmt.format(bar.get_height()), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"holdout ({o['total']} tasks)", "holdout accuracy (%)"])
    ax.set_ylim(0, max(o["total"], c["total"], 100) + 2)
    ax.set_ylabel("exact-match passes")
    ax.set_title("Frozen held-out suite: retention only")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    families = sorted(set(verdict["original"]["per_family"]) | set(verdict["candidate"]["per_family"]))
    opf, cpf = verdict["original"]["per_family"], verdict["candidate"]["per_family"]
    positions = range(len(families))
    orig = [opf.get(f, {}).get("accuracy", 0.0) * 100 for f in families]
    cand = [cpf.get(f, {}).get("accuracy", 0.0) * 100 for f in families]
    ax.bar([p - width / 2 for p in positions], orig, width, color=MUTED, label="original")
    ax.bar([p + width / 2 for p in positions], cand, width, color=ACCENT, label="candidate")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(families, rotation=38, ha="right", fontsize=8)
    ax.set_ylim(0, 104)
    ax.set_ylabel("accuracy %")
    ax.set_title("Per-family holdout accuracy")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    steps, losses, avg = loss_curve(results)
    if steps:
        ax.plot(steps, losses, "o-", color=ACCENT, ms=4, label="step loss")
        ax.plot(steps, avg, "-", color=GREEN, lw=1.6, label="logged running mean")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.set_title("Streamed LoRA training loss")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.grid(alpha=0.3)
    else:
        note = (
            f"Original: {o['passed']}/{o['total']} exact matches; candidate: {c['passed']}/{c['total']}.\n"
            f"Gains: {len(verdict['gains']) or 'none'}; regressions: {', '.join(verdict['regressions']) or 'none'}; "
            f"ties: {verdict['ties']}.\n"
            "Frozen fixtures measure structured reasoning, not general model quality; "
            "prior exposure cannot be ruled out.\n"
            "No automatic promotion. Ties show no proven improvement; keep the original unchanged.\n"
            "Independent evaluation is still required; historic runs may differ in configuration/backend."
        )
        ax.text(0.01, 0.95, note, va="top", fontsize=10, color=MUTED, linespacing=1.7, wrap=True)
    try:
        with out.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150, facecolor=BG)
    finally:
        plt.close(fig)
    print(f"saved {out}\n{title}")


if __name__ == "__main__":
    main()
