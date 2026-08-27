"""M1 - Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck 5).

Run: python missions/m1_efficiency_audit.py

Extension 2 ("Your Turn") lives here: capability unit-economics ($/GB-VRAM,
$/TB-s) plus roofline-driven right-sizing for memory-bound serving GPUs.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
HEADROOM = 1.15            # keep 15% of peak BW / VRAM free before calling a SKU "big enough"
MFU_RIGHTSIZE_MAX = 0.35   # above this the GPU is earning its FLOPs - leave it alone


def unit_economics(cat: dict) -> list[dict]:
    """Per-GPU capability pricing: $/GB-VRAM, $/TB-s of HBM bandwidth, TFLOP/$.

    A memory-bound serving job buys *bandwidth and VRAM*, not FLOPs - so ranking
    the catalog by $/GPU-hr alone systematically picks the wrong GPU.
    """
    out = []
    for gtype, c in cat.items():
        od, hbm, bw = num(c["on_demand_hr"]), num(c["hbm_gb"]), num(c["peak_bw_tbs"])
        out.append({
            "gpu_type": gtype, "on_demand_hr": od, "hbm_gb": hbm, "peak_bw_tbs": bw,
            "usd_per_gb_vram": od / hbm if hbm else 0.0,
            "usd_per_tbs": od / bw if bw else 0.0,
            "tflops_per_usd": num(c["peak_tflops_fp16"]) / od if od else 0.0,
        })
    return sorted(out, key=lambda x: x["usd_per_tbs"])


def mbu_rightsizing(summary: list[dict], cat: dict) -> dict:
    """Extension 2 - size memory-bound GPUs on bandwidth + VRAM, not on FLOPs.

    Regime comes from the roofline: a GPU whose measured arithmetic intensity
    (achieved FLOP / achieved byte) sits below its own ridge point is limited by
    HBM traffic, so the tensor cores you are renting sit idle. For those boxes we
    look for the cheapest SKU that still clears measured bandwidth and VRAM
    demand (+15% headroom). GPUs already converting FLOPs (MFU >= 0.35) are left
    alone - downsizing them would just move the bottleneck.
    """
    rows = []
    total_now = total_fit = 0.0
    for s in summary:
        cur = cat[s["gpu_type"]]
        ridge = metrics.arithmetic_intensity(num(cur["peak_tflops_fp16"]), num(cur["peak_bw_tbs"]))
        intensity = metrics.arithmetic_intensity(s["achieved_tflops"], s["achieved_bw_tbs"])
        regime = metrics.roofline_regime(intensity, ridge)

        need_bw = s["peak_bw_tbs"] * HEADROOM
        need_gb = s["peak_mem_gb"] * HEADROOM
        monthly_now = num(cur["on_demand_hr"]) * 24 * DAYS

        best, best_hr = s["gpu_type"], num(cur["on_demand_hr"])
        eligible = regime == "memory-bound" and s["mfu"] < MFU_RIGHTSIZE_MAX
        if eligible:
            for cand in sorted(cat.values(), key=lambda c: num(c["on_demand_hr"])):
                if (num(cand["peak_bw_tbs"]) >= need_bw and num(cand["hbm_gb"]) >= need_gb
                        and num(cand["on_demand_hr"]) < best_hr):
                    best, best_hr = cand["gpu_type"], num(cand["on_demand_hr"])
                    break
        monthly_fit = best_hr * 24 * DAYS
        total_now += monthly_now
        total_fit += monthly_fit
        rows.append({
            "gpu_id": s["gpu_id"], "gpu_type": s["gpu_type"], "workload": s["workload"],
            "regime": regime, "intensity": round(intensity, 1), "ridge": round(ridge, 1),
            "mfu": s["mfu"], "mbu": s["mbu"],
            "need_bw_tbs": round(need_bw, 2), "need_gb": round(need_gb, 1),
            "recommended": best, "monthly_now": round(monthly_now),
            "monthly_fit": round(monthly_fit), "monthly_savings": round(monthly_now - monthly_fit),
        })
    savings = total_now - total_fit
    return {"rows": rows, "fleet_monthly_now": round(total_now), "fleet_monthly_fit": round(total_fit),
            "monthly_savings": round(savings),
            "savings_pct": round(savings / total_now * 100, 1) if total_now else 0.0}


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "tflops": [], "bw": [], "mem": [], "workload": None})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["workload"] = r["workload"]
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["tflops"].append(num(r["achieved_tflops"]))
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["mem"].append(num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        # size on the BUSY hours only - idle hours would understate real demand
        busy = [t for t, u in zip(a["tflops"], a["util"]) if u >= 10] or a["tflops"]
        busy_bw = [b for b, u in zip(a["bw"], a["util"]) if u >= 10] or a["bw"]
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"], "workload": a["workload"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "achieved_tflops": round(sum(busy) / len(busy), 1),
            "achieved_bw_tbs": round(sum(busy_bw) / len(busy_bw), 3),
            "peak_bw_tbs": round(max(a["bw"]), 3),
            "peak_mem_gb": round(max(a["mem"]), 1),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(cat[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    rightsizing = mbu_rightsizing(summary, cat)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        for l in lies:
            billed = num(cat[l["gpu_type"]]["on_demand_hr"]) * 24 * DAYS
            print(f"  {l['gpu_id']}: util {l['gpu_util_pct']}% but MFU {l['mfu']:.0%} -> "
                  f"${billed:,.0f}/mo rented, ~${billed * l['mfu']:,.0f} of it converted to FLOPs "
                  f"(${billed * (1 - l['mfu']):,.0f} burned)")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste * DAYS:,.0f}/month")

        print("\n-- Extension 2: capability unit economics (cheapest bandwidth first) --")
        print(f"{'gpu_type':9}{'$/hr':>7}{'HBM_GB':>8}{'$/GB-VRAM':>11}{'BW_TB/s':>9}{'$/TB-s':>9}{'TFLOP/$':>9}")
        for c in unit_economics(cat):
            print(f"{c['gpu_type']:9}{c['on_demand_hr']:>7.2f}{c['hbm_gb']:>8.0f}{c['usd_per_gb_vram']:>11.4f}"
                  f"{c['peak_bw_tbs']:>9.2f}{c['usd_per_tbs']:>9.3f}{c['tflops_per_usd']:>9.0f}")

        print("\n-- Extension 2: roofline right-sizing --")
        print(f"{'GPU':13}{'now':7}{'load':7}{'FLOP/B':>8}{'ridge':>7}{'regime':>15}"
              f"{'needBW':>8}{'needGB':>8}{'fit':>8}{'$/mo now':>10}{'$/mo fit':>10}{'save':>8}")
        for r in rightsizing["rows"]:
            print(f"{r['gpu_id']:13}{r['gpu_type']:7}{r['workload']:7}{r['intensity']:>8.1f}{r['ridge']:>7.0f}"
                  f"{r['regime']:>15}{r['need_bw_tbs']:>8.2f}{r['need_gb']:>8.1f}{r['recommended']:>8}"
                  f"{r['monthly_now']:>10,}{r['monthly_fit']:>10,}{r['monthly_savings']:>8,}")
        print(f"right-size the memory-bound + low-MFU GPUs -> ${rightsizing['monthly_savings']:,}/month "
              f"({rightsizing['savings_pct']:.1f}% of the ${rightsizing['fleet_monthly_now']:,}/mo fleet bill)")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsizing": rightsizing}


if __name__ == "__main__":
    run()
