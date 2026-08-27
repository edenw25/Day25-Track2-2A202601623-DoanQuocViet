# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,223  
**Projected savings:** $12,910  (**48%**)
**Unit economics:** $6.488 -> $1.126 per 1M tokens (**-82.6%**)  

## Savings by lever

| Lever | Savings (USD) | Share of savings |
|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9.4% |
| Purchasing (spot/reserved) | $9,788 | 75.8% |
| Right-size util-lies | $1,310 | 10.1% |
| Kill idle GPUs | $600 | 4.6% |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1
- Reasoning traffic: 8.4% of requests but 16.5% of the inference bill and 94.0% of the energy (148.2 Wh vs 0.86 Wh per request).
- Moving every interruptible job to europe-north1 cuts 532 kgCO2e/month (92% of their footprint) and $45.63/month of electricity.
- Cheapest power is us-east-wa ($0.055/kWh, 90 gCO2/kWh) - the balanced pick when latency matters.

## Inference levers in detail

Leave-one-out marginal value of each inference lever (against the fully optimized bill):

| Lever | $/day | $/month | Why |
|---|---|---|---|
| Cascade (small-model routing) | $27.64 | $829 | 80% of traffic is easy; the small tier is ~15x cheaper per token |
| Batch API | $1.79 | $54 | -50% on everything that tolerates a queue (eval traffic) |
| Prompt caching | $1.17 | $35 | -90% on the cached share of input; only chat/RAG carry a big shared prefix |

Baseline is the naive deployment - every request on the large model, no cache, no batch: **$6.488/1M-token**. The optimized mix lands at **$1.126/1M-token (82.6% cheaper)**. Note the ordering: routing beats discounts by an order of magnitude. Discounts scale the price of a token; cascading changes which token you buy.

## The GPU-Util lie

| GPU | Type | GPU-Util | MFU | MBU | Billed / month | Burned / month |
|---|---|---|---|---|---|---|
| `gpu-h100-4` | H100 | 98.2% | 0.19 | 0.21 | $1,800 | $1,451 |
| `gpu-a10g-1` | A10G | 96.9% | 0.27 | 0.30 | $720 | $527 |

`nvidia-smi` GPU-Util answers one question only: *was at least one kernel resident on the device during the sampling window?* It is a duty-cycle counter, not a throughput counter. A kernel that spends its life stalled on HBM reads, or a stream of tiny kernels whose launch overhead dominates their math, keeps that counter pinned at ~100% while the tensor cores idle. That is exactly `gpu-h100-4`: 98% util, MFU 0.19 - the roofline says it runs at 278 FLOP/byte against an H100 ridge of 296, i.e. memory-bound. You rent 990 TFLOP/s and collect ~190. **The billing consequence:** util-based dashboards mark this GPU as healthy and *fully used*, so nobody right-sizes it and capacity planning asks for more of the same SKU. Measure MFU/MBU per job, alert when util > 90% and MFU < 0.30, and the waste becomes visible the day it starts.

## Region choice: cost vs carbon vs latency

| Region | $/kWh | gCO2/kWh | Added latency | Verdict |
|---|---|---|---|---|
| europe-north1 | 0.090 | 30 | +110ms | cleanest - park interruptible training here |
| us-east-wa | 0.055 | 90 | +55ms | cheapest power + low carbon - best all-round |
| us-west-2 | 0.070 | 120 | +70ms | viable US fallback |
| us-east-1 | 0.120 | 380 | +5ms | where we run today |
| europe-central2 | 0.180 | 660 | +120ms | avoid - dirtiest and most expensive |

Carbon and cost are not in conflict here: `us-east-wa` is both the cheapest power ($0.055/kWh vs $0.120 at home) and 4x cleaner than us-east-1. The real constraint is latency: the three serving jobs (job-infer-chat, job-infer-rag, job-infer-search) stay put, the five interruptible jobs move.

## Recommended actions (in order)

Priority is ROI per week of engineering, not raw dollars:

| # | Action | Monthly value | Effort | Why this order |
|---|---|---|---|---|
| 1 | Enforce cascade routing (small tier by default, escalate on failure) | $829 | days | Largest single lever, no vendor negotiation, reversible per-route |
| 2 | Move interruptible jobs to spot + checkpointing, commit only the 24/7 serving fleet | $9,788 | 1-2 weeks | Second-largest lever; needs checkpoint plumbing before it is safe |
| 3 | Kill idle GPUs (auto-stop after 30 min under 10% util) | $600 | hours | Cheapest fix in the report - a cron job and an alert |
| 4 | Right-size the memory-bound / util-lie GPUs | $1,310 | 1 week | Requires a re-benchmark per job, but the roofline already tells us where to look |
| 5 | Reasoning budget: gate the reasoning path behind a complexity check | $9 + 11,934 Wh/day | days | Small in dollars, dominant in energy - it is 94% of our Wh |
| 6 | Turn on chargeback (tag coverage is 92%, gate is 80%) | indirect | days | Teams only optimize what lands on their own budget line |

## "Your Turn" extensions - measured

| # | Extension | Where | Measured result |
|---|---|---|---|
| 1 | Risk-/term-aware `recommend_tier_v2()` | `finops/pricing.py`, `missions/m3_purchasing.py` | v1 claimed 39.1% savings, v2 38.1% - v1 was billing reserved on *used* hours; the honest number is $15,879/mo |
| 2 | Roofline right-sizing on $/TB-s and $/GB-VRAM | `missions/m1_efficiency_audit.py` | $1,310/mo (8.5% of the fleet bill) by moving memory-bound, low-MFU GPUs to the cheapest SKU that still clears measured BW+VRAM |
| 3 | `cache_is_worth_it()` break-even gate | `finops/pricing.py`, `missions/m2_inference_levers.py` | small tier needs 5.8 reads, sees 238 (41x headroom); large tier needs 0.6, sees 62 - caching is applied on both |
| 4 | Reasoning budget ($ and Wh) | `missions/m2_inference_levers.py` | 8.4% of traffic = 16.5% of cost and 94.0% of energy; a 5% cap saves $9/mo + 11,934 Wh/day |
| 5 | Carbon-aware scheduling | `missions/m6_carbon_scheduling.py` | 532 kgCO2e/mo (92%) and $45.63/mo by moving interruptible jobs to europe-north1 |

## Method and caveats

- Baseline = naive inference bill ($48.87/day x 30) + 100% on-demand purchasing for the 8 workloads ($25,667/mo).
- Inference savings come from the 2,400-request log; purchasing from `workloads.csv`; idle and right-size from the 11-GPU telemetry fleet. The telemetry fleet and the workload list overlap only partially, so the two efficiency levers are conservative floors rather than a second bite of the same dollar.
- Tag coverage 92% clears the 80% chargeback gate; top spender is `assistant` at $2.59/day of the $8.47/day inference bill.
- Prices are June-2026 snapshots. Spot interruption rates are modelled per GPU type (H100 3%/h ... L4 15%/h); reserved is billed on the full 720h commitment, not on usage.

_Figures are June-2026 as-of snapshots; re-baseline before acting._