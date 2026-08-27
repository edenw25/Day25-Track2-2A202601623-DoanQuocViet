"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# ---------------------------------------------------------------------------
# YOUR TURN — Extension 1: a purchasing policy that knows about interruption
# risk and commitment term, not just duty cycle.
# ---------------------------------------------------------------------------

# Per-hour spot reclaim probability. Scarce, in-demand accelerators are actually
# *safer* on spot in 2026 neoclouds: capacity pools are large and the reclaim
# pressure sits on the cheap long-tail SKUs that everybody oversubscribes.
SPOT_INTERRUPT_RATE = {
    "B200": 0.02, "H100": 0.03, "H200": 0.03,
    "A100": 0.05, "MI300X": 0.06, "A10G": 0.12, "L4": 0.15,
}
DEFAULT_INTERRUPT_RATE = 0.08

# A commitment is only honest if the workload outlives it. `days` in
# workloads.csv is the run length inside a 30-day month.
MIN_DAYS_FOR_1YR = 14
MIN_DAYS_FOR_3YR = 28


def spot_interrupt_rate(gpu_type: str | None) -> float:
    """Per-hour reclaim probability for a GPU type (falls back to a fleet average)."""
    return SPOT_INTERRUPT_RATE.get(gpu_type or "", DEFAULT_INTERRUPT_RATE)


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    prices: dict,
    gpu_type: str | None = None,
    job_days: int = 30,
    days_per_month: int = 30,
    ckpt_overhead_frac: float = 0.03,
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Cost-minimising tier choice per GPU per month (Extension 1).

    Three upgrades over the simple duty-cycle policy:

    1. **Interruption rate is per GPU type.** An A10G reclaimed 12%/hour pays a
       much bigger rework tax than an H100 at 3%/hour, so "interruptible -> spot"
       is not automatically the cheapest answer.
    2. **Reserved is billed on the commitment, not on usage.** You pay 24x30
       hours whether you use them or not — which is the *real* break-even test
       and something the v1 policy silently ignored.
    3. **1yr vs 3yr is chosen by workload durability.** A 14-day training run has
       no business signing a 3-year commit.

    `prices` is a price_catalog row (dict of strings is fine). Returns the chosen
    tier plus the full cost menu, so a report can show why it chose what it chose.
    """
    def _f(key, default=0.0):
        try:
            return float(prices.get(key, default))
        except (TypeError, ValueError):
            return default

    # Billing window is the month, so v1 and v2 are compared over identical
    # runtime; `job_days` only gates which commitment terms are honest to sign.
    used_hours = max(0.0, hours_per_day) * days_per_month
    committed_hours = 24.0 * days_per_month
    od_hr = _f("on_demand_hr")

    menu: dict = {"on_demand": used_hours * od_hr}

    if interruptible:
        sim = spot_checkpoint_cost(
            used_hours, _f("spot_hr"), od_hr,
            interrupt_rate=spot_interrupt_rate(gpu_type),
            ckpt_overhead_frac=ckpt_overhead_frac,
            rework_hours_per_interrupt=rework_hours_per_interrupt,
        )
        menu["spot"] = sim["spot_cost"]

    if job_days >= MIN_DAYS_FOR_1YR and _f("reserved_1yr_hr") > 0:
        menu["reserved_1yr"] = committed_hours * _f("reserved_1yr_hr")
    if job_days >= MIN_DAYS_FOR_3YR and _f("reserved_3yr_hr") > 0:
        menu["reserved_3yr"] = committed_hours * _f("reserved_3yr_hr")

    best = min(menu, key=menu.get)
    tier = "reserved" if best.startswith("reserved") else best
    term = best.split("_", 1)[1] if best.startswith("reserved") else None

    reasons = {
        "spot": f"interrupt rate {spot_interrupt_rate(gpu_type):.0%}/h keeps the rework tax below the on-demand premium",
        "reserved": f"duty cycle {hours_per_day/24:.0%} amortises a {term} commitment billed on all {committed_hours:.0f} h",
        "on_demand": "duty cycle too low to amortise a commitment and the job cannot be interrupted",
    }
    return {
        "tier": tier,
        "reserved_term": term,
        "monthly_cost_per_gpu": round(menu[best], 2),
        "menu": {k: round(v, 2) for k, v in menu.items()},
        "used_hours": round(used_hours, 1),
        "interrupt_rate": spot_interrupt_rate(gpu_type),
        "reason": reasons[tier],
    }


# ---------------------------------------------------------------------------
# YOUR TURN — Extension 3: prompt caching only pays above a break-even read count
# ---------------------------------------------------------------------------

def cache_break_even_reads(
    base_price_per_m: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    storage_cost_per_m_hour: float = 0.0,
    ttl_hours: float = 0.0,
) -> float:
    """How many cache READS a written prefix needs before caching is cheaper.

    Writing a prefix costs a premium over a normal input token (Anthropic 5m
    writes ~1.25x); Gemini-style implicit caches also rent storage per hour.
    Each read then saves `(1 - read_discount)` of the base input price.

        break_even = (write premium + storage rent) / savings per read
    """
    save_per_read = base_price_per_m * (1.0 - read_discount)
    if save_per_read <= 0:
        return float("inf")
    extra_write = max(0.0, write_cost_per_m - base_price_per_m)
    rent = max(0.0, storage_cost_per_m_hour) * max(0.0, ttl_hours)
    return (extra_write + rent) / save_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_price_per_m: float | None = None,
    storage_cost_per_m_hour: float = 0.0,
    ttl_hours: float = 0.0,
) -> bool:
    """True when a cached prefix is re-read often enough to beat its write cost.

    If `base_price_per_m` is omitted we assume the Anthropic 1.25x write premium.
    Cheap models have the *harder* time clearing the bar: the per-read saving
    shrinks with the token price while storage rent does not.
    """
    base = base_price_per_m if base_price_per_m is not None else write_cost_per_m / 1.25
    be = cache_break_even_reads(base, write_cost_per_m, read_discount,
                                storage_cost_per_m_hour, ttl_hours)
    return avg_cache_reads >= be
