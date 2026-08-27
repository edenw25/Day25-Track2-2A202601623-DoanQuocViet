"""M2 - Inference Cost Levers: $/1M-token, batch x cache x cascade (deck 7).

Run: python missions/m2_inference_levers.py

Two "Your Turn" extensions live here:
  * Extension 3 - `pricing.cache_is_worth_it()` gates the caching lever per tier.
  * Extension 4 - a reasoning budget: $ and Wh split by `is_reasoning`.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) - illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
# Cache economics per tier: write premium (Anthropic ~1.25x input) and the
# hourly storage rent a Gemini-style implicit cache charges per 1M cached tokens.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_STORAGE_PER_M_HOUR = 1.00
CACHE_TTL_HOURS = 1.0
REASONING_TRAFFIC_CAP = 0.10   # Extension 4: proposed routing budget


def cache_economics(rows) -> dict:
    """Extension 3 - is the cached prefix re-read often enough to pay for itself?

    A cached prefix is written once and then re-read by every later request that
    shares it. We proxy a prefix by (team, project, route_tier): one shared
    system prompt per surface. Break-even is computed per model tier, because the
    per-read saving scales with the token price while storage rent does not -
    caching is *hardest* to justify on the cheap small model.
    """
    prefixes = defaultdict(int)
    reads_by_tier = defaultdict(int)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            key = (r["team"], r["project"], r["route_tier"])
            prefixes[key] += 1
            reads_by_tier[r["route_tier"]] += 1

    tiers = {}
    for tier, (price_in, _) in MODEL_PRICES.items():
        keys = [k for k in prefixes if k[2] == tier]
        n_prefix = len(keys) or 1
        avg_reads = reads_by_tier[tier] / n_prefix          # writes once, reads the rest
        be = pricing.cache_break_even_reads(
            base_price_per_m=price_in,
            write_cost_per_m=price_in * CACHE_WRITE_MULTIPLIER,
            storage_cost_per_m_hour=CACHE_STORAGE_PER_M_HOUR,
            ttl_hours=CACHE_TTL_HOURS,
        )
        worth = pricing.cache_is_worth_it(
            avg_reads, write_cost_per_m=price_in * CACHE_WRITE_MULTIPLIER,
            base_price_per_m=price_in,
            storage_cost_per_m_hour=CACHE_STORAGE_PER_M_HOUR, ttl_hours=CACHE_TTL_HOURS,
        )
        tiers[tier] = {"prefixes": len(keys), "cached_requests": reads_by_tier[tier],
                       "avg_reads": round(avg_reads, 1), "break_even_reads": round(be, 1),
                       "worth_it": worth, "headroom_x": round(avg_reads / be, 1) if be > 0 else float("inf")}
    return tiers


def reasoning_budget(rows, per_request_cost, cap_frac: float = REASONING_TRAFFIC_CAP) -> dict:
    """Extension 4 - what does reasoning traffic really cost, in $ and in Wh?

    Reasoning requests emit ~6x the tokens here AND run on an inference path the
    deck prices at ~80x the energy of a small-model answer. Capping them to
    `cap_frac` of traffic is modelled as re-routing the excess to the non-reasoning
    mix (same team, average cost/energy of a non-reasoning request).
    """
    buckets = {0: {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0},
               1: {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0}}
    for r, cost in zip(rows, per_request_cost):
        flag = int(num(r["is_reasoning"]))
        tok = int(num(r["input_tokens"])) + int(num(r["output_tokens"]))
        b = buckets[flag]
        b["n"] += 1
        b["cost"] += cost
        b["tokens"] += tok
        b["wh"] += sustainability.wh_per_query(tok, is_reasoning=bool(flag))

    total_n = buckets[0]["n"] + buckets[1]["n"]
    total_cost = buckets[0]["cost"] + buckets[1]["cost"]
    total_wh = buckets[0]["wh"] + buckets[1]["wh"]
    rn = buckets[1]["n"]

    avg_reason_cost = buckets[1]["cost"] / rn if rn else 0.0
    avg_reason_wh = buckets[1]["wh"] / rn if rn else 0.0
    avg_plain_cost = buckets[0]["cost"] / buckets[0]["n"] if buckets[0]["n"] else 0.0
    avg_plain_wh = buckets[0]["wh"] / buckets[0]["n"] if buckets[0]["n"] else 0.0

    def scenario(frac: float) -> dict:
        moved = max(0, rn - int(total_n * frac))
        return {
            "cap_frac": frac, "requests_rerouted": moved,
            "savings_daily": round(moved * (avg_reason_cost - avg_plain_cost), 2),
            "savings_monthly": round(moved * (avg_reason_cost - avg_plain_cost) * 30, 2),
            "wh_saved_daily": round(moved * (avg_reason_wh - avg_plain_wh), 1),
            "binding": moved > 0,
        }

    curve = [scenario(f) for f in (0.10, 0.05, 0.02)]
    chosen = scenario(cap_frac)
    moved = chosen["requests_rerouted"]

    return {
        "cap_curve": curve,
        "reasoning_requests": rn,
        "traffic_pct": round(rn / total_n * 100, 1) if total_n else 0.0,
        "cost_pct": round(buckets[1]["cost"] / total_cost * 100, 1) if total_cost else 0.0,
        "energy_pct": round(buckets[1]["wh"] / total_wh * 100, 1) if total_wh else 0.0,
        "reasoning_cost_daily": round(buckets[1]["cost"], 2),
        "reasoning_wh_daily": round(buckets[1]["wh"], 1),
        "avg_cost_reasoning": round(avg_reason_cost, 5),
        "avg_cost_plain": round(avg_plain_cost, 5),
        "avg_wh_reasoning": round(avg_reason_wh, 3),
        "avg_wh_plain": round(avg_plain_wh, 3),
        "cap_frac": cap_frac,
        "requests_rerouted": moved,
        "cap_savings_daily": round(moved * (avg_reason_cost - avg_plain_cost), 2),
        "cap_savings_monthly": round(moved * (avg_reason_cost - avg_plain_cost) * 30, 2),
        "cap_wh_saved_daily": round(moved * (avg_reason_wh - avg_plain_wh), 1),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache = cache_economics(rows)

    base_cost = opt_cost = 0.0
    total_tokens = 0
    opt_per_request = []
    lever_no_cache = lever_no_batch = lever_no_cascade = 0.0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        tier = r["route_tier"]
        total_tokens += inp + out
        # BASELINE: naive deployment - everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        # Extension 3: only claim the caching discount where the prefix clears break-even.
        use_cache = cached if cache[tier]["worth_it"] else 0
        pin, pout = MODEL_PRICES[tier]
        c = pricing.request_cost(inp, out, pin, pout, cached_in=use_cache, batch=is_batch)
        opt_cost += c
        opt_per_request.append(c)
        # single-lever attribution (leave-one-out against the fully optimized bill)
        lever_no_cache += pricing.request_cost(inp, out, pin, pout, cached_in=0, batch=is_batch)
        lever_no_batch += pricing.request_cost(inp, out, pin, pout, cached_in=use_cache, batch=False)
        lever_no_cascade += pricing.request_cost(inp, out, lin, lout, cached_in=use_cache, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    levers = {
        "cascade": round(lever_no_cascade - opt_cost, 2),
        "caching": round(lever_no_cache - opt_cost, 2),
        "batch": round(lever_no_batch - opt_cost, 2),
    }
    reasoning = reasoning_budget(rows, opt_per_request)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("marginal value of each lever (leave-one-out, $/day):")
        for k, v in sorted(levers.items(), key=lambda x: -x[1]):
            print(f"  {k:9} ${v:7.2f}/day   ${v*30:9,.0f}/month")

        print("\n-- Extension 3: is prompt caching worth it? --")
        print(f"{'tier':7}{'prefixes':>10}{'cached_req':>12}{'avg_reads':>11}{'break_even':>12}{'headroom':>10}  worth_it")
        for tier, c in cache.items():
            print(f"{tier:7}{c['prefixes']:>10}{c['cached_requests']:>12}{c['avg_reads']:>11.1f}"
                  f"{c['break_even_reads']:>12.1f}{c['headroom_x']:>9.1f}x  {c['worth_it']}")
        print(f"  (write premium {CACHE_WRITE_MULTIPLIER}x input + ${CACHE_STORAGE_PER_M_HOUR:.2f}/1M-token-hour "
              f"storage for {CACHE_TTL_HOURS:.0f}h; each read saves 90% of the input price)")

        print("\n-- Extension 4: reasoning budget --")
        print(f"reasoning traffic : {reasoning['traffic_pct']:.1f}% of requests")
        print(f"reasoning cost    : {reasoning['cost_pct']:.1f}% of the optimized bill "
              f"(${reasoning['reasoning_cost_daily']:,.2f}/day)")
        print(f"reasoning energy  : {reasoning['energy_pct']:.1f}% of Wh ({reasoning['reasoning_wh_daily']:,.0f} Wh/day)")
        print(f"avg request       : ${reasoning['avg_cost_reasoning']:.5f} / {reasoning['avg_wh_reasoning']:.2f} Wh  vs "
              f"${reasoning['avg_cost_plain']:.5f} / {reasoning['avg_wh_plain']:.2f} Wh (non-reasoning)")
        print("cap scenarios (reroute the excess to the non-reasoning mix):")
        for sc in reasoning["cap_curve"]:
            note = "" if sc["binding"] else "  <- not binding: traffic is already below this cap"
            print(f"  cap {sc['cap_frac']:>4.0%}: reroute {sc['requests_rerouted']:>4} req/day  "
                  f"${sc['savings_daily']:>6.2f}/day  ${sc['savings_monthly']:>8,.0f}/mo  "
                  f"{sc['wh_saved_daily']:>8,.0f} Wh/day{note}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "levers_daily": levers, "cache_economics": cache, "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
