"""M5 - Optimization Report: combine M1-M4 + M6 into baseline-vs-optimized (deck 1/11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import (m1_efficiency_audit, m2_inference_levers, m3_purchasing,
                      m4_allocation, m6_carbon_scheduling)

DAYS = 30
MEDIAN_QUERY_TOKENS = 800


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    r6 = m6_carbon_scheduling.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets -----------------------------------------------------------
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    # Extension 1: the risk-/term-aware policy is the one we would actually sign.
    purchasing_savings = r3["on_demand_monthly"] - r3["v2_monthly"]
    idle_savings = r1["idle_waste_daily"] * DAYS
    # Extension 2: roofline right-sizing replaces the old "one tier down" guess.
    rightsize_savings = r1["rightsizing"]["monthly_savings"]

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot -------------------------------------------
    wh = sustainability.wh_per_query(MEDIAN_QUERY_TOKENS)
    best_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    reasoning = r2["reasoning"]
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": best_region,
        "notes": [
            f"Reasoning traffic: {reasoning['traffic_pct']:.1f}% of requests but "
            f"{reasoning['cost_pct']:.1f}% of the inference bill and "
            f"{reasoning['energy_pct']:.1f}% of the energy "
            f"({reasoning['avg_wh_reasoning']:.1f} Wh vs {reasoning['avg_wh_plain']:.2f} Wh per request).",
            f"Moving every interruptible job to {r6['cleanest']} cuts "
            f"{r6['carbon_saved_kg_month']:,.0f} kgCO2e/month ({r6['carbon_saved_pct']:.0f}% of their footprint) "
            f"and ${r6['energy_cost_saved_month']:,.2f}/month of electricity.",
            f"Cheapest power is {r6['cheapest']} (${sustainability.REGION_PRICE_KWH[r6['cheapest']]:.3f}/kWh, "
            f"{sustainability.REGION_CARBON[r6['cheapest']]} gCO2/kWh) - the balanced pick when latency matters.",
        ],
    }

    # --- narrative sections -------------------------------------------------
    lie_rows = []
    for l in r1["lies"]:
        billed = num(cat[l["gpu_type"]]["on_demand_hr"]) * 24 * DAYS
        lie_rows.append(
            f"| `{l['gpu_id']}` | {l['gpu_type']} | {l['gpu_util_pct']}% | {l['mfu']:.2f} | {l['mbu']:.2f} | "
            f"${billed:,.0f} | ${billed * (1 - l['mfu']):,.0f} |")
    lies_md = "\n".join([
        "| GPU | Type | GPU-Util | MFU | MBU | Billed / month | Burned / month |",
        "|---|---|---|---|---|---|---|", *lie_rows,
        "",
        "`nvidia-smi` GPU-Util answers one question only: *was at least one kernel resident on "
        "the device during the sampling window?* It is a duty-cycle counter, not a throughput "
        "counter. A kernel that spends its life stalled on HBM reads, or a stream of tiny kernels "
        "whose launch overhead dominates their math, keeps that counter pinned at ~100% while the "
        "tensor cores idle. That is exactly `gpu-h100-4`: 98% util, MFU 0.19 - the roofline says it "
        f"runs at {r1['rightsizing']['rows'][4]['intensity']:.0f} FLOP/byte against an H100 ridge of "
        f"{r1['rightsizing']['rows'][4]['ridge']:.0f}, i.e. memory-bound. You rent 990 TFLOP/s and "
        "collect ~190. **The billing consequence:** util-based dashboards mark this GPU as healthy "
        "and *fully used*, so nobody right-sizes it and capacity planning asks for more of the same "
        "SKU. Measure MFU/MBU per job, alert when util > 90% and MFU < 0.30, and the waste becomes "
        "visible the day it starts.",
    ])

    lever2 = r2["levers_daily"]
    inference_md = "\n".join([
        "Leave-one-out marginal value of each inference lever (against the fully optimized bill):",
        "",
        "| Lever | $/day | $/month | Why |",
        "|---|---|---|---|",
        f"| Cascade (small-model routing) | ${lever2['cascade']:.2f} | ${lever2['cascade']*DAYS:,.0f} | "
        "80% of traffic is easy; the small tier is ~15x cheaper per token |",
        f"| Batch API | ${lever2['batch']:.2f} | ${lever2['batch']*DAYS:,.0f} | "
        "-50% on everything that tolerates a queue (eval traffic) |",
        f"| Prompt caching | ${lever2['caching']:.2f} | ${lever2['caching']*DAYS:,.0f} | "
        "-90% on the cached share of input; only chat/RAG carry a big shared prefix |",
        "",
        f"Baseline is the naive deployment - every request on the large model, no cache, no batch: "
        f"**${r2['baseline_per_m']:.3f}/1M-token**. The optimized mix lands at "
        f"**${r2['optimized_per_m']:.3f}/1M-token ({r2['savings_pct']:.1f}% cheaper)**. Note the ordering: "
        "routing beats discounts by an order of magnitude. Discounts scale the price of a token; "
        "cascading changes which token you buy.",
    ])

    actions_md = "\n".join([
        "Priority is ROI per week of engineering, not raw dollars:",
        "",
        "| # | Action | Monthly value | Effort | Why this order |",
        "|---|---|---|---|---|",
        f"| 1 | Enforce cascade routing (small tier by default, escalate on failure) | "
        f"${lever2['cascade']*DAYS:,.0f} | days | Largest single lever, no vendor negotiation, reversible per-route |",
        f"| 2 | Move interruptible jobs to spot + checkpointing, commit only the 24/7 serving fleet | "
        f"${purchasing_savings:,.0f} | 1-2 weeks | Second-largest lever; needs checkpoint plumbing before it is safe |",
        f"| 3 | Kill idle GPUs (auto-stop after 30 min under 10% util) | ${idle_savings:,.0f} | hours | "
        "Cheapest fix in the report - a cron job and an alert |",
        f"| 4 | Right-size the memory-bound / util-lie GPUs | ${rightsize_savings:,.0f} | 1 week | "
        "Requires a re-benchmark per job, but the roofline already tells us where to look |",
        f"| 5 | Reasoning budget: gate the reasoning path behind a complexity check | "
        f"${reasoning['cap_curve'][1]['savings_monthly']:,.0f} + "
        f"{reasoning['cap_curve'][1]['wh_saved_daily']:,.0f} Wh/day | days | "
        "Small in dollars, dominant in energy - it is 94% of our Wh |",
        f"| 6 | Turn on chargeback (tag coverage is {r4['tag_coverage']:.0%}, gate is 80%) | "
        "indirect | days | Teams only optimize what lands on their own budget line |",
    ])

    ext_md = "\n".join([
        "| # | Extension | Where | Measured result |",
        "|---|---|---|---|",
        f"| 1 | Risk-/term-aware `recommend_tier_v2()` | `finops/pricing.py`, `missions/m3_purchasing.py` | "
        f"v1 claimed {r3['savings_pct']:.1f}% savings, v2 {r3['v2_savings_pct']:.1f}% - v1 was billing reserved "
        f"on *used* hours; the honest number is ${r3['v2_monthly']:,.0f}/mo |",
        f"| 2 | Roofline right-sizing on $/TB-s and $/GB-VRAM | `missions/m1_efficiency_audit.py` | "
        f"${r1['rightsizing']['monthly_savings']:,.0f}/mo ({r1['rightsizing']['savings_pct']:.1f}% of the fleet bill) "
        "by moving memory-bound, low-MFU GPUs to the cheapest SKU that still clears measured BW+VRAM |",
        f"| 3 | `cache_is_worth_it()` break-even gate | `finops/pricing.py`, `missions/m2_inference_levers.py` | "
        f"small tier needs {r2['cache_economics']['small']['break_even_reads']:.1f} reads, sees "
        f"{r2['cache_economics']['small']['avg_reads']:.0f} ({r2['cache_economics']['small']['headroom_x']:.0f}x headroom); "
        f"large tier needs {r2['cache_economics']['large']['break_even_reads']:.1f}, sees "
        f"{r2['cache_economics']['large']['avg_reads']:.0f} - caching is applied on both |",
        f"| 4 | Reasoning budget ($ and Wh) | `missions/m2_inference_levers.py` | "
        f"{reasoning['traffic_pct']:.1f}% of traffic = {reasoning['cost_pct']:.1f}% of cost and "
        f"{reasoning['energy_pct']:.1f}% of energy; a 5% cap saves "
        f"${reasoning['cap_curve'][1]['savings_monthly']:,.0f}/mo + "
        f"{reasoning['cap_curve'][1]['wh_saved_daily']:,.0f} Wh/day |",
        f"| 5 | Carbon-aware scheduling | `missions/m6_carbon_scheduling.py` | "
        f"{r6['carbon_saved_kg_month']:,.0f} kgCO2e/mo ({r6['carbon_saved_pct']:.0f}%) and "
        f"${r6['energy_cost_saved_month']:,.2f}/mo by moving interruptible jobs to {r6['cleanest']} |",
    ])

    regions_md = "\n".join([
        "| Region | $/kWh | gCO2/kWh | Added latency | Verdict |",
        "|---|---|---|---|---|",
        *[f"| {r['region']} | {r['usd_per_kwh']:.3f} | {r['gco2_per_kwh']} | +{r['added_latency_ms']}ms | "
          + ("cleanest - park interruptible training here" if r["region"] == r6["cleanest"]
             else "cheapest power + low carbon - best all-round" if r["region"] == r6["balanced"]
             else "where we run today" if r["region"] == "us-east-1"
             else "avoid - dirtiest and most expensive" if r["region"] == "europe-central2"
             else "viable US fallback") + " |"
          for r in r6["regions"]],
        "",
        f"Carbon and cost are not in conflict here: `{r6['cheapest']}` is both the cheapest power "
        f"(${sustainability.REGION_PRICE_KWH[r6['cheapest']]:.3f}/kWh vs $0.120 at home) and "
        f"{sustainability.REGION_CARBON['us-east-1'] // sustainability.REGION_CARBON[r6['cheapest']]}x cleaner "
        "than us-east-1. The real constraint is latency: the three serving jobs "
        f"({', '.join(r6['fixed'])}) stay put, the five interruptible jobs move.",
    ])

    method_md = "\n".join([
        f"- Baseline = naive inference bill (${r2['baseline_daily']:.2f}/day x {DAYS}) + "
        f"100% on-demand purchasing for the 8 workloads (${r3['on_demand_monthly']:,}/mo).",
        "- Inference savings come from the 2,400-request log; purchasing from `workloads.csv`; "
        "idle and right-size from the 11-GPU telemetry fleet. The telemetry fleet and the workload "
        "list overlap only partially, so the two efficiency levers are conservative floors "
        "rather than a second bite of the same dollar.",
        f"- Tag coverage {r4['tag_coverage']:.0%} clears the 80% chargeback gate; "
        f"top spender is `{max(r4['by_team'], key=r4['by_team'].get)}` at "
        f"${max(r4['by_team'].values()):.2f}/day of the ${sum(r4['by_team'].values()):.2f}/day inference bill.",
        "- Prices are June-2026 snapshots. Spot interruption rates are modelled per GPU type "
        "(H100 3%/h ... L4 15%/h); reserved is billed on the full 720h commitment, not on usage.",
    ])

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        unit_economics={"baseline_per_m": r2["baseline_per_m"],
                        "optimized_per_m": r2["optimized_per_m"],
                        "savings_pct": r2["savings_pct"]},
        extra_sections=[
            ("Inference levers in detail", inference_md),
            ("The GPU-Util lie", lies_md),
            ("Region choice: cost vs carbon vs latency", regions_md),
            ("Recommended actions (in order)", actions_md),
            ('"Your Turn" extensions - measured', ext_md),
            ("Method and caveats", method_md),
        ])
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                   baseline_usd=baseline, optimized_usd=optimized)

    if verbose:
        print("== M5 Optimization Report ==")
        enc = _sys.stdout.encoding or "utf-8"
        print(md.encode(enc, "replace").decode(enc))
        print("\nWritten: outputs/report.md" + (" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
