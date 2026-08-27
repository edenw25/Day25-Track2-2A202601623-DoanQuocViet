"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Extension 1 ("Your Turn") lives here too: the baseline duty-cycle policy
(`pricing.recommend_tier`) is scored side by side against a risk- and
term-aware policy (`pricing.recommend_tier_v2`) so the delta is measurable.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def _v1_cost(job, cat) -> tuple[str, float, float]:
    """Baseline policy: duty cycle + interruptible flag, reserved billed on usage."""
    gtype = job["gpu_type"]
    ngpu = int(num(job["num_gpus"]))
    hpd = num(job["hours_per_day"])
    interruptible = bool(int(num(job["interruptible"])))
    c = cat[gtype]
    gpu_hours = hpd * DAYS * ngpu
    od = num(c["on_demand_hr"])
    on_demand_cost = gpu_hours * od

    tier = pricing.recommend_tier(hpd, interruptible)
    if tier == "spot":
        opt = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    elif tier == "reserved":
        opt = gpu_hours * num(c["reserved_3yr_hr"])
    else:
        opt = on_demand_cost
    return tier, on_demand_cost, opt


def _v2_cost(job, cat) -> tuple[dict, float]:
    """Extension 1 policy: per-GPU-type interruption risk + honest commitment billing."""
    gtype = job["gpu_type"]
    ngpu = int(num(job["num_gpus"]))
    rec = pricing.recommend_tier_v2(
        hours_per_day=num(job["hours_per_day"]),
        interruptible=bool(int(num(job["interruptible"]))),
        prices=cat[gtype],
        gpu_type=gtype,
        job_days=int(num(job["days"])),
        days_per_month=DAYS,
    )
    return rec, rec["monthly_cost_per_gpu"] * ngpu


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = v2_monthly = 0.0
    recs = []
    for j in jobs:
        tier, od_cost, opt_cost = _v1_cost(j, cat)
        rec2, cost2 = _v2_cost(j, cat)

        on_demand_monthly += od_cost
        optimized_monthly += opt_cost
        v2_monthly += cost2
        recs.append({
            "job_id": j["job_id"], "gpu_type": j["gpu_type"], "tier": tier,
            "on_demand": round(od_cost), "optimized": round(opt_cost),
            "v2_tier": rec2["tier"] + (f"-{rec2['reserved_term']}" if rec2["reserved_term"] else ""),
            "v2_cost": round(cost2), "v2_reason": rec2["reason"],
            "interrupt_rate": rec2["interrupt_rate"], "v2_menu": rec2["menu"],
        })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    v2_savings_pct = (on_demand_monthly - v2_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- Extension 1: risk- & term-aware policy (v2) --")
        print(f"{'job':18}{'gpu':7}{'int/h':>7}  {'v1 tier':11}{'v2 tier':16}{'v1 $':>10}{'v2 $':>10}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['interrupt_rate']:>6.0%}   {r['tier']:11}{r['v2_tier']:16}"
                  f"${r['optimized']:>9,}${r['v2_cost']:>9,}")
        print(f"\nv1 policy: ${optimized_monthly:,.0f}/mo  ({savings_pct:.1f}% vs on-demand)")
        print(f"v2 policy: ${v2_monthly:,.0f}/mo  ({v2_savings_pct:.1f}% vs on-demand)")
        print(f"delta    : ${v2_monthly - optimized_monthly:+,.0f}/mo "
              f"({v2_savings_pct - savings_pct:+.1f} pp) -- v2 bills reserved on the commitment, not on usage")
        print("\nwhy v2 moved:")
        for r in recs:
            if r["v2_tier"].split("-")[0] != r["tier"] or abs(r["v2_cost"] - r["optimized"]) > 1:
                print(f"  {r['job_id']:18} {r['v2_reason']}  menu={r['v2_menu']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "v2_monthly": round(v2_monthly), "v2_savings_pct": round(v2_savings_pct, 1)}


if __name__ == "__main__":
    run()
