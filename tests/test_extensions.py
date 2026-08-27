"""Student-written tests for the five "Your Turn" extensions.

The shipped 15 tests are untouched; these cover only the new logic.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling
from missions._common import catalog_by_type

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4,
        "hbm_gb": 80, "peak_tflops_fp16": 990, "peak_bw_tbs": 3.35, "watts": 700}


# --- Extension 1: recommend_tier_v2 ----------------------------------------

def test_v1_policy_is_unchanged():
    """The new policy must not break the documented v1 contract."""
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_v2_prefers_spot_for_low_risk_interruptible():
    rec = pricing.recommend_tier_v2(20, True, H100, gpu_type="H100", job_days=14)
    assert rec["tier"] == "spot"
    assert rec["monthly_cost_per_gpu"] < rec["menu"]["on_demand"]


def test_v2_interruption_rate_is_per_gpu_type():
    assert pricing.spot_interrupt_rate("H100") < pricing.spot_interrupt_rate("L4")
    cheap = pricing.recommend_tier_v2(8, True, H100, gpu_type="H100", job_days=22)
    risky = pricing.recommend_tier_v2(8, True, H100, gpu_type="L4", job_days=22)
    # same prices, different reclaim risk -> the risky pool costs more rework
    assert risky["menu"]["spot"] > cheap["menu"]["spot"]


def test_v2_bills_reserved_on_the_commitment_not_on_usage():
    """18h/day still pays for 720h of commitment - this is what v1 got wrong."""
    rec = pricing.recommend_tier_v2(18, False, H100, gpu_type="H100", job_days=30)
    assert abs(rec["menu"]["reserved_3yr"] - 24 * 30 * H100["reserved_3yr_hr"]) < 0.01


def test_v2_refuses_a_3yr_commit_for_a_short_job():
    rec = pricing.recommend_tier_v2(24, False, H100, gpu_type="H100", job_days=10)
    assert "reserved_3yr" not in rec["menu"] and "reserved_1yr" not in rec["menu"]
    assert rec["tier"] == "on_demand"


def test_v2_picks_the_longer_term_when_the_workload_is_durable():
    rec = pricing.recommend_tier_v2(24, False, H100, gpu_type="H100", job_days=30)
    assert rec["tier"] == "reserved" and rec["reserved_term"] == "3yr"


# --- Extension 2: MBU / roofline right-sizing -------------------------------

def test_unit_economics_ranks_bandwidth_not_sticker_price():
    cat = catalog_by_type()
    econ = m1_efficiency_audit.unit_economics(cat)
    by_type = {e["gpu_type"]: e for e in econ}
    # L4 is the cheapest GPU-hour but the most expensive bandwidth
    assert by_type["L4"]["on_demand_hr"] < by_type["H100"]["on_demand_hr"]
    assert by_type["L4"]["usd_per_tbs"] > by_type["H100"]["usd_per_tbs"]
    assert econ[0]["usd_per_tbs"] == min(e["usd_per_tbs"] for e in econ)


def test_rightsizing_never_recommends_a_gpu_that_does_not_fit():
    cat = catalog_by_type()
    res = m1_efficiency_audit.run(verbose=False)["rightsizing"]
    for row in res["rows"]:
        rec = cat[row["recommended"]]
        if row["recommended"] != row["gpu_type"]:
            assert float(rec["peak_bw_tbs"]) >= row["need_bw_tbs"]
            assert float(rec["hbm_gb"]) >= row["need_gb"]
            assert float(rec["on_demand_hr"]) < float(cat[row["gpu_type"]]["on_demand_hr"])
    assert res["monthly_savings"] > 0


def test_rightsizing_leaves_healthy_high_mfu_gpus_alone():
    res = m1_efficiency_audit.run(verbose=False)["rightsizing"]
    for row in res["rows"]:
        if row["mfu"] >= m1_efficiency_audit.MFU_RIGHTSIZE_MAX:
            assert row["recommended"] == row["gpu_type"]


# --- Extension 3: cache_is_worth_it ----------------------------------------

def test_cache_break_even_scales_with_storage_rent():
    no_rent = pricing.cache_break_even_reads(3.0, 3.75)
    rented = pricing.cache_break_even_reads(3.0, 3.75, storage_cost_per_m_hour=1.0, ttl_hours=1.0)
    assert rented > no_rent


def test_cheap_models_need_more_reads_to_justify_a_cache():
    small = pricing.cache_break_even_reads(0.20, 0.25, storage_cost_per_m_hour=1.0, ttl_hours=1.0)
    large = pricing.cache_break_even_reads(3.00, 3.75, storage_cost_per_m_hour=1.0, ttl_hours=1.0)
    assert small > large  # the per-read saving shrinks, the rent does not


def test_cache_is_worth_it_gate():
    assert pricing.cache_is_worth_it(50, write_cost_per_m=0.25, base_price_per_m=0.20,
                                     storage_cost_per_m_hour=1.0, ttl_hours=1.0) is True
    assert pricing.cache_is_worth_it(1, write_cost_per_m=0.25, base_price_per_m=0.20,
                                     storage_cost_per_m_hour=1.0, ttl_hours=1.0) is False


def test_dataset_clears_the_cache_break_even():
    econ = m2_inference_levers.run(verbose=False)["cache_economics"]
    for tier, c in econ.items():
        assert c["avg_reads"] > c["break_even_reads"], tier
        assert c["worth_it"] is True


# --- Extension 4: reasoning budget -----------------------------------------

def test_reasoning_is_a_small_share_of_traffic_but_dominates_energy():
    r = m2_inference_levers.run(verbose=False)["reasoning"]
    assert r["traffic_pct"] < 15
    assert r["cost_pct"] > r["traffic_pct"]       # costs more per request
    assert r["energy_pct"] > 90                   # ~80x energy multiplier dominates
    assert r["avg_wh_reasoning"] > r["avg_wh_plain"] * 50


def test_reasoning_cap_curve_is_monotonic():
    curve = m2_inference_levers.run(verbose=False)["reasoning"]["cap_curve"]
    saves = [c["savings_daily"] for c in curve]   # caps get tighter: 10% -> 5% -> 2%
    assert saves == sorted(saves)
    assert curve[0]["binding"] is False           # already below a 10% cap


# --- Extension 5: carbon-aware scheduling ----------------------------------

def test_carbon_scheduling_moves_only_interruptible_jobs():
    res = m6_carbon_scheduling.run(verbose=False)
    moved = {r["job_id"] for r in res["movable"]}
    assert "job-infer-chat" not in moved and "job-infer-chat" in res["fixed"]
    assert "job-train-llm" in moved


def test_cleanest_region_beats_home_on_carbon_and_cost():
    res = m6_carbon_scheduling.run(verbose=False)
    assert res["cleanest"] == "europe-north1"
    assert res["carbon_saved_kg_month"] > 0
    assert res["energy_cost_saved_month"] > 0
    assert 0 < res["carbon_saved_pct"] <= 100


# --- cross-mission consistency (the report must match the missions) ---------

def test_report_numbers_match_mission_outputs():
    from missions import m5_report
    r5 = m5_report.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "report.md")
    md = open(path, encoding="utf-8").read()
    assert f"${r5['baseline_monthly']:,.0f}" in md
    assert f"${r5['optimized_monthly']:,.0f}" in md
    # M5 books the honest (v2) purchasing number, not the optimistic v1 one
    assert r5["levers"]["Purchasing (spot/reserved)"] == r3["on_demand_monthly"] - r3["v2_monthly"]
