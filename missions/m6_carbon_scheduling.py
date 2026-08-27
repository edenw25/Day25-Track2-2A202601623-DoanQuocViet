"""M6 (Extension 5) - Carbon-aware scheduling for interruptible workloads.

Run: python missions/m6_carbon_scheduling.py

Interruptible jobs are the ones that can *move*: they already tolerate being
stopped and restarted, so they can follow the cheapest + cleanest grid instead of
sitting next to the users. Latency-bound serving cannot, and we say so.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS_PER_MONTH = 30
HOME_REGION = "us-east-1"
# Rough one-way network distance penalty from us-east users, for the latency
# trade-off discussion (illustrative ms of added round-trip).
REGION_LATENCY_MS = {
    "us-east-1": 5, "us-east-wa": 55, "us-west-2": 70,
    "europe-north1": 110, "europe-central2": 120,
}


def region_table() -> list[dict]:
    """All five regions side by side: price, carbon, latency."""
    rows = []
    for region, gco2 in sustainability.REGION_CARBON.items():
        rows.append({
            "region": region,
            "usd_per_kwh": sustainability.REGION_PRICE_KWH.get(region, 0.12),
            "gco2_per_kwh": gco2,
            "added_latency_ms": REGION_LATENCY_MS.get(region, 0),
        })
    return sorted(rows, key=lambda r: r["gco2_per_kwh"])


def job_energy_kwh(job: dict, cat: dict) -> float:
    """Monthly energy of a job: GPU-hours x board power (85% average draw)."""
    watts = num(cat[job["gpu_type"]]["watts"]) * 0.85
    gpu_hours = num(job["hours_per_day"]) * min(int(num(job["days"])), DAYS_PER_MONTH) * int(num(job["num_gpus"]))
    return gpu_hours * watts / 1000.0


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    regions = region_table()

    cleanest = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    cheapest = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)
    # "Balanced" = lowest normalised (cost rank + carbon rank), latency-tolerant only.
    def _score(r):
        c = r["usd_per_kwh"] / max(x["usd_per_kwh"] for x in regions)
        g = r["gco2_per_kwh"] / max(x["gco2_per_kwh"] for x in regions)
        return c + g
    balanced = min(regions, key=_score)["region"]

    movable, fixed = [], []
    for j in jobs:
        kwh = job_energy_kwh(j, cat)
        wh = kwh * 1000.0
        row = {
            "job_id": j["job_id"], "gpu_type": j["gpu_type"], "kwh": round(kwh, 1),
            "home_cost": round(sustainability.energy_cost_usd(wh, HOME_REGION), 2),
            "home_carbon_kg": round(sustainability.carbon_g(wh, HOME_REGION) / 1000.0, 1),
            "clean_cost": round(sustainability.energy_cost_usd(wh, cleanest), 2),
            "clean_carbon_kg": round(sustainability.carbon_g(wh, cleanest) / 1000.0, 1),
        }
        row["carbon_saved_kg"] = round(row["home_carbon_kg"] - row["clean_carbon_kg"], 1)
        row["cost_saved"] = round(row["home_cost"] - row["clean_cost"], 2)
        (movable if bool(int(num(j["interruptible"]))) else fixed).append(row)

    tot_carbon = round(sum(r["carbon_saved_kg"] for r in movable), 1)
    tot_cost = round(sum(r["cost_saved"] for r in movable), 2)
    home_carbon = round(sum(r["home_carbon_kg"] for r in movable), 1)

    if verbose:
        print("== M6 Carbon-aware Scheduling (Extension 5) ==")
        print(f"{'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'+latency':>10}   note")
        for r in regions:
            note = []
            if r["region"] == cleanest:
                note.append("cleanest")
            if r["region"] == cheapest:
                note.append("cheapest")
            if r["region"] == balanced:
                note.append("balanced pick")
            if r["region"] == HOME_REGION:
                note.append("today")
            print(f"{r['region']:16}{r['usd_per_kwh']:>8.3f}{r['gco2_per_kwh']:>10}{r['added_latency_ms']:>9}ms   "
                  + ", ".join(note))

        print(f"\nmovable (interruptible) jobs - {HOME_REGION} vs {cleanest}:")
        print(f"{'job':18}{'gpu':7}{'kWh/mo':>9}{'$ home':>9}{'$ clean':>9}{'kgCO2 home':>12}{'kgCO2 clean':>13}{'saved':>9}")
        for r in movable:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['kwh']:>9,.0f}{r['home_cost']:>9,.2f}{r['clean_cost']:>9,.2f}"
                  f"{r['home_carbon_kg']:>12,.0f}{r['clean_carbon_kg']:>13,.0f}{r['carbon_saved_kg']:>9,.0f}")
        pct = tot_carbon / home_carbon * 100 if home_carbon else 0.0
        print(f"\nmove all interruptible jobs to {cleanest}: -{tot_carbon:,.0f} kgCO2e/month "
              f"({pct:.0f}% of their footprint) and ${tot_cost:,.2f}/month of electricity")
        print(f"jobs that must stay near users (latency-bound serving): "
              f"{[r['job_id'] for r in fixed]} (+{REGION_LATENCY_MS[cleanest]}ms would be user-visible)")

    return {"regions": regions, "cleanest": cleanest, "cheapest": cheapest, "balanced": balanced,
            "movable": movable, "fixed": [r["job_id"] for r in fixed],
            "carbon_saved_kg_month": tot_carbon, "energy_cost_saved_month": tot_cost,
            "carbon_saved_pct": round(tot_carbon / home_carbon * 100, 1) if home_carbon else 0.0}


if __name__ == "__main__":
    run()
