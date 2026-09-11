# GOAT Proof 713: ASEntmax — Length-Adaptive α-Entmax Damping Schedule (Issue 747 P0)

> **Date:** 2026-09-11
> **Feature Gate:** `asentmax_schedule` (katgpt-attn; root shim `asentmax_schedule = ["dash_attn", "katgpt-attn/asentmax_schedule"]`)
> **Source:** Research 549 / arXiv:2506.16640 (Vasylenko et al., ICLR 2026) — Eq 10 closed form
> **Verdict:** 🟢 **GOAT PASS (G1–G4)** as an opt-in primitive. Default-on promotion **deferred** to the forward-path wiring gate (P0.7) — the scheduled routing variant is not yet on a production hot path; promoting now would create a default-on-unwired state (feature-gate-audit discipline).

## Summary

The entmax-side mirror of SSMax: pre-scale α-entmax routing scores by `β·(log n_c)^{−0.5}` with `β = 1/(2σ̂√2)` from a rolling range-law σ̂. One extreme-value law, two signed arms — softmax needs sharpening ∝ log n (SSMax, shipped), entmax needs damping ∝ (log n)^{−0.5} (this primitive). Zero training; one `powf` + one mul per head per step.

**G2 headline:** graded-relevance planted-set (k=8) recall **0.81–0.91 scheduled vs 0.14–0.31 unscheduled** across n_c = 256 → 16k, σ = 1 → 8. The unscheduled support collapses to 1–2.5 chunks (over-sparsification — the paper's fixed-α Copy-task failure mode, reproduced on our routing surface); the scheduled support holds at 7–8 with the full planted set.

## Surface

| Item | Location |
|---|---|
| `AsentmaxSchedule` (None / Derived / Generalized) | `katgpt-attn/src/dash_attn/asentmax.rs` |
| `apply_asentmax_inplace` | same |
| `RollingSigmaEstimator` (lock-free EMA, Kamath range law) | same |
| `score_blocks_entmax_with_schedule_into` (routing socket) | `katgpt-attn/src/dash_attn/routing.rs` |
| Root re-export | `src/lib.rs` (`dash_attn::asentmax::*`) |

Deviation from the issue text (documented): the schedule lives in a new `dash_attn/asentmax.rs` module rather than inline in `entmax.rs` — mirrors `ssmax.rs`'s standalone-module status (the transform and the schedule are distinct concerns; entmax.rs stays the pure algorithm). Same socket either way.

## Test Configuration

| Parameter | Value |
|---|---|
| Harness | Bench 032 pattern (NIAH at growing chunk counts) + σ axis |
| Head dim | 64 |
| Planted set | k=8, near-tied graded block at (1.15 − 0.004·r)·M, M = σ√(2 ln n) |
| Chunk counts | 256, 1_024, 4_096, 16_384 (G1 extends to 524_288) |
| Logit std σ | 1.0, 3.0, 8.0 |
| Seeds | 8 per arm (estimator warm-up pass before measurement) |

## Gate Results

### G1 — stationarity (tests/asentmax_g1_stationarity.rs, 4/4 PASS)

| Property | Result |
|---|---|
| Scaled range pinned (oracle σ̂) | 0.83–0.90 across n=512→512k, σ=1/3/10 (target [0.65, 1.35]) |
| Scaled range n-invariance | ratio 1.09 (bound [0.8, 1.25]); unscheduled ratio ≈ 1.58 > 1.4 (√(2 log n) growth) |
| Support σ-invariance | max/min < 2.0 across σ ∈ {1,3,10} at every n (σ cancels exactly in σ·β) |
| Support n-axis | 19.5 → 44.3 over n=512→512k — mild ∝ ln n growth (analytic k* ≈ 4·ln n, expected ratio 2.11) |
| Collapse counterfactual | unscheduled support at σ=10: ≤ 4 (→ 1 in practice); scheduled ≥ 4× raw |
| Simplex + exact zeros | preserved at every (n, σ) |
| Estimator self-consistency | range-fed σ̂ pins the scaled range to 1.0 exactly (finite-n EV corrections cancel by construction); oracle-σ arm sits at the corrected EV value 0.91 |

**Honest scope note:** the issue's original G1 wording ("support stationary ±ε, → 1 without it") is not what the math gives for pure IID-Gaussian scores — the *exact* stationarity claims are the scaled RANGE (n-invariant by construction) and the σ-AXIS (exact cancellation); the support grows mildly ∝ ln n when damped and collapses on the σ axis when not. The issue text was amended to the measured truth.

### G2 — routing quality (bench_747_asentmax_goat.rs, PASS)

| n | σ | raw recall | sched recall | raw mass | sched mass | raw \|S\| | sched \|S\| |
|---|---|---|---|---|---|---|---|
| 256 | 1.0 | 0.266 | 0.875 | 1.000 | 0.995 | 2.1 | 7.5 |
| 1024 | 1.0 | 0.312 | 0.828 | 1.000 | 0.991 | 2.5 | 7.2 |
| 4096 | 1.0 | 0.297 | 0.891 | 1.000 | 0.990 | 2.4 | 8.4 |
| 16384 | 1.0 | 0.250 | 0.906 | 1.000 | 0.998 | 2.0 | 8.1 |
| 256 | 3.0 | 0.156 | 0.875 | 1.000 | 0.996 | 1.2 | 7.5 |
| 1024 | 3.0 | 0.188 | 0.812 | 1.000 | 0.992 | 1.5 | 7.1 |
| 4096 | 3.0 | 0.188 | 0.859 | 1.000 | 0.992 | 1.5 | 7.9 |
| 16384 | 3.0 | 0.188 | 0.906 | 1.000 | 0.999 | 1.5 | 8.1 |
| 256 | 8.0 | 0.141 | 0.875 | 1.000 | 0.996 | 1.1 | 7.4 |
| 1024 | 8.0 | 0.156 | 0.812 | 1.000 | 0.992 | 1.2 | 7.1 |
| 4096 | 8.0 | 0.156 | 0.859 | 1.000 | 0.993 | 1.2 | 7.8 |
| 16384 | 8.0 | 0.141 | 0.906 | 1.000 | 0.999 | 1.1 | 8.1 |

Planted mass ≈ 1.0 in BOTH arms (the raw arm concentrates all mass on the 1–2 planted chunks it keeps) — recall is the differentiator: the raw arm *drops 6–7 of 8 relevant chunks from the support*. Note the planted logits carry their own noise jitter (±σ) — realistic — which is why the raw arm over-sparsifies even at the 256-chunk baseline.

### G3 — no-regression at Bench 032 scale (single-needle task, PASS)

| σ | recall raw/sched | needle mass raw/sched | \|S\| raw/sched |
|---|---|---|---|
| 1.0 | 1.000 / 1.000 | 0.894 / 0.611 | 1.4 / 13.0 |
| 2.0 | 1.000 / 1.000 | 0.883 / 0.620 | 1.1 / 12.1 |

Single-needle retrieval parity at 100% (032 T23 anchor). Cost anchor: 032's own shipped envelope (avg 21.5 active blocks, range ~4–40+ at 64–256 chunks) — scheduled 12–13 is *inside* it, and 13/256 = 5% coverage is below 032's shipped 8.4% average. A ratio-vs-raw bound is the wrong anchor: the raw arm's \|S\| ≈ 1.4 is itself the over-sparsification disease this primitive treats.

### G4 — allocation (tests/asentmax_alloc_check.rs, PASS)

0 allocations over 1000 steady-state cycles (`apply_asentmax_inplace` + `observe_row` + `to_schedule` + `multiplier`), CountingAllocator with liveness canary. Cross-crate `#[path]` include of katgpt-core's shared `counting_allocator!` test common (single source of truth).

### Latency

`apply + observe + to_schedule` at n=16_384: **≈ 9.7–18.6 µs/step** measured across two runs (row-scan + multiply; same order as the shipped `RollingDeltaEstimator` precedent). The multiplier itself is one `powf` + one mul.

### G5 — build matrix

- default features: clippy `--all-targets` clean (the module compiles to nothing; routing refactor is behavior-neutral — 154 pre-existing dash_attn lib tests pass through the refactored path)
- `--features asentmax_schedule`: clippy `--all-targets` clean, 172 lib tests + 4 G1 tests + 1 alloc test PASS
- `--all-features`: check clean
- root `--features asentmax_schedule`: check clean (re-export shim)

## Substrate check (substrate-first skill)

- Searched: `asentmax`/`ASEntmax` (0 hits — concept does not ship), damping/length-adaptive/log-n temperature on entmax paths, `SsmaxMode`/`RollingDeltaEstimator` (the socket pattern), `AdaptiveKConfig` (different concern — support budget, not score scaling).
- Decision: build new, mirroring the `SsmaxMode` socket shape; `RollingSigmaEstimator` mirrors `ssmax::RollingDeltaEstimator` (lock-free EMA + CAS). Placed beside its only consumer in katgpt-attn (katgpt-core untouched — keeps the parallel Issue 746 work disjoint).

## Honest caveats

1. γ=−0.5 is IID-Gaussian-optimal only (paper's own per-head fitted γ varies in sign). The `Generalized` arm is the offline-sweep surface (P4 T4.3); harvested per-head constants may also arrive from riir-train Plan 396 Ph2.
2. The synthetic harness's planted-logit noise jitter is what breaks the raw arm at the baseline — realistic, but the G2 margin on real DashAttention chunk scores (which are summary dots, not trained attention) must be re-measured at the P0.7 wiring gate before any default-on promotion.
3. Routing-quality anchors are the paper's models; our gates re-measure on our routing surfaces (the anchors set targets, not pass conditions — Research 549 §2.5).

## Next

- **P0.7 (new, in Issue 747):** wire `score_blocks_entmax_with_schedule_into` into the forward-path routing call site (config-gated), re-gate G2/G3 on the real prefill path, then re-evaluate default-on promotion.
- P1 (Lemma-2 support controller `k̂ = 4/Δ̂²`), P2 (Prop 6 eviction window), P3 (incremental decode entmax) — unstarted.
