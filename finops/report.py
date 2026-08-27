"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None,
                 extra_sections: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    `unit_economics` optionally carries the $/1M-token headline (the unit that
    actually matters); `extra_sections` is a list of (heading, markdown-body)
    pairs appended after the sustainability block.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
    ]
    if unit_economics:
        lines += [
            f"**Unit economics:** ${unit_economics.get('baseline_per_m', 0):.3f} -> "
            f"${unit_economics.get('optimized_per_m', 0):.3f} per 1M tokens "
            f"(**-{unit_economics.get('savings_pct', 0):.1f}%**)  ",
        ]
    lines += [
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | Share of savings |",
        "|---|---|---|",
    ]
    total_lever = sum(levers.values()) or 1.0
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} | {amount / total_lever * 100:.1f}% |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
        for extra in sustainability.get("notes", []):
            lines.append(f"- {extra}")
    for heading, body in (extra_sections or []):
        lines += ["", f"## {heading}", "", body]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str,
                      baseline_usd: float | None = None,
                      optimized_usd: float | None = None) -> str:
    """Write the savings chart PNG. Returns the path. No-op if matplotlib absent.

    With `baseline_usd` it draws a real waterfall (baseline -> one step down per
    lever -> optimized); without it, a plain bar chart of the levers.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]

    if baseline_usd is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
        ax.set_title("GPU cost savings by FinOps lever")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    optimized_usd = baseline_usd - sum(vals) if optimized_usd is None else optimized_usd
    labels = ["Baseline"] + [n.split(" (")[0] for n in names] + ["Optimized"]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(0, baseline_usd, color="#7a2e2e", width=0.6)
    ax.text(0, baseline_usd, f"${baseline_usd:,.0f}", ha="center", va="bottom", fontsize=9)
    running = baseline_usd
    for i, (n, v) in enumerate(zip(names, vals), start=1):
        ax.bar(i, -v, bottom=running, color="#2e548a", width=0.6)
        ax.plot([i - 0.7, i + 0.3], [running, running], color="#999", lw=0.8, ls="--")
        label = f"-${v:,.0f} ({v / baseline_usd * 100:.1f}%)"
        if v / baseline_usd >= 0.12:   # thick enough to hold the label inside
            ax.text(i, running - v / 2, label.replace(" (", "\n("),
                    ha="center", va="center", fontsize=8, color="white")
        else:
            ax.text(i, running + baseline_usd * 0.012, label,
                    ha="center", va="bottom", fontsize=8, color="#2e548a")
        running -= v
    ax.bar(len(names) + 1, running, color="#2e7a4f", width=0.6)
    ax.text(len(names) + 1, running, f"${running:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("GPU spend (USD / month)")
    saved_pct = (baseline_usd - running) / baseline_usd * 100 if baseline_usd else 0.0
    ax.set_ylim(0, baseline_usd * 1.14)   # headroom so labels clear the title
    ax.set_title(f"NimbusAI GPU spend: baseline -> optimized  (-{saved_pct:.0f}%)", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
