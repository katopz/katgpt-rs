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
- P1 (Lemma-2 support controller `k̂ = 4/Δ̂²`) — landed 2026-09-11, see the P1 section below.
- P2 (Prop 6 eviction window), P3 (incremental decode entmax) — landed 2026-09-11, see the P2/P3 addendum below.

## P1 addendum — derived-k budget (Lemma-2 support controller, 2026-09-11)

**T1.1** `compute_derived_k(Δ̂)` (exact law `k̂ = 4/Δ̂²`, the α=1.5 instantiation of `((α−1)Δ̂)^{−1/(α−1)}`) + `compute_derived_k_from_scores` (one-pass max−mean proxy, 4-way unrolled) in `adaptive_k.rs`, gated `asentmax_schedule`. Unit tests pin the exact map, monotonicity, clamps (incl. NaN → k_max, the safe coverage direction), and the proxy.

**T1.2 G1 PASS** (tests/asentmax_p1_derived_k_g1.rs, 4/4): two-level rows planted AT the Lemma-2 boundary (k = k̂(Δ) needles, gap Δ ∈ {0.5, 0.7, 1.0, 1.4}) — **realized entmax support == k̂ at every n ∈ {1k, 8k, 65k, 512k, 1M}**. The condition has no n term; the length-independence law is exact. The sigmoid arm's input (row variance) provably moves with n at fixed structure — no such law exists for it (reported in-test).

**T1.3 head-to-head — SPLIT VERDICT (recorded honestly):**

| Arm | budget k | recall@k (planted k=8) | recall/block |
|---|---|---|---|
| sigmoid `w·var+b` (w=5, b=0) | 32 (saturated k_max at every n, σ) | 1.000 | 0.031 |
| derived `4/Δ̂²` (max−mean) | 4 (k_min floor at every n, σ) | 0.500 | **0.125** |

- **Absolute coverage: sigmoid wins.** The max−mean proxy measures top-to-center (`Δ + bulk-mean-offset`), not the Lemma-2 top-block-to-bulk-level gap — on multi-level rows it saturates k_min and behaves as a binary concentration detector, not a calibrated coverage budget (mechanism pinned by `g1_max_mean_proxy_calibration_pinned`).
- **Cost efficiency: derived wins 4×** (0.125 vs 0.031 recall/block) — in the concentration regime (single needle) it delivers recall 1.0 at k=4 vs the sigmoid arm's k=32 (8× less attention work).
- Notable: the sigmoid arm's `w=5, b=0` defaults are saturated across the whole sweep (variance ≥ 1 → sigmoid(5·var) ≈ 1) — it is effectively a CONSTANT k=32 here, i.e. not adaptive at all at realistic logit-variance scales.

**Disposition:** the derived budget does NOT replace the sigmoid arm for coverage budgeting (honest negative on the replacement claim). It ships as (a) the exact length-independent law (`compute_derived_k(Δ_level)` — caller-supplied level gap, e.g. from a change-point detector or frozen per-head table) and (b) the concentration-regime convenience (`compute_derived_k_from_scores`). Both stay opt-in under `asentmax_schedule`. A true level-gap estimator could revisit the coverage claim.

## P2 addendum — ALiBi×entmax eviction window (Prop E.2, 2026-09-11)

**T2.1** `alibi_entmax_window_1p5(z_min, z_max, slope)` + `kv_within_window` (the KV-retention predicate) + `evicted_kv_fraction` in new `dash_attn/eviction_window.rs`, gated `asentmax_schedule`. **Formula correction vs the research note:** the paper's Eq. 110 is `d_max = ⌊(z_max − z_min + 1/(α−1))/m_h + 1⌋` — the `+1` is INSIDE the floor and the numerator term is `1/(α−1)` (= 2 for α=1.5), not the note's transcribed `⌊(z_range+1)/m_h⌋ + 1`. Verified against the fetched paper (arXiv:2506.16640v4 App. E.2).

**T2.2 G1 PASS — bit-identity, the cleanest gate in the batch** (`tests/asentmax_p2_eviction_g1.rs`, 4/4): over 432 adversarial configurations (slope × z-range × n ∈ {64,256,1024} × 4 seeds × 3 row builders — random, max-at-farthest-distance, tied-clusters):
- support ⊆ window at every config (every supported index within d_max);
- evicted probabilities are EXACTLY 0.0 (never ε);
- **windowed-vs-full entmax_1p5 bit-identical** (`f32::to_bits` equality at every kept index).

The gate is provable, not just measured: our `entmax_1p5`'s threshold floor `τ_c ≥ s_max − 1` (single-support minimum of the normalized quadratic variant) + the monotone-failure property of the Peters scan imply non-support ⟺ `s ≤ τ_c`; removing exact-zero terms from the index-ordered normalization sum is bit-neutral (`x + 0.0 = x`). Cross-checked with a 20k-trial f64 simulation (0 violations) before the Rust gate. The paper's margin (2) is CONSERVATIVE for our implementation (our true floor is 1) — the gate pins the citable theorem bound.

**T2.3 G2 PASS (model + measured):**

| axis | result |
|---|---|
| KV bytes model @ n=1M (32h × 128d × f16 × KV = 16 GiB/layer) | min evicted **98.9%** (flattest head 2⁻⁸ @ σ=4), steepest **99.99%** — **15.8–16.0 GiB/layer** provably evictable; Kamath range law `2σ√(2 ln n)` feeds the bounds |
| measured entmax row cost @ 1M | full 130,028 µs vs windowed(48) **1.0 µs** — the compute beyond the window is free because the mass is provably zero |

Note: the decode tok/s delta is an ENGINE-level measurement — the riir-ai KV-path wiring is a consumer-side follow-up (the retention predicate + window calc are the katgpt-rs primitives; scope note in the module docs).

## P3 addendum — Lemma-1 incremental decode entmax (2026-09-11)

**T3.1** `IncrementalEntmax1p5` in new `dash_attn/entmax_incremental.rs`, gated `asentmax_schedule`. Below-τ pushes: O(1) (one compare + one zero write — Lemma 1: existing probabilities bit-unchanged). Support-entry pushes: O(candidates) rescan + O(len) probs rebuild. **Drop-below-τ is exact, not approximate:** the Peters-scan threshold is monotone non-decreasing under additions (weighted-average argument; cross-checked 20k spiky streams, 0 violations), so a dropped score can never re-enter the support — and the rescan over surviving candidates reproduces the full re-sort bit-for-bit (same sorted prefix ⇒ same `t_k` arithmetic).

**T3.2 G1 PASS** (`tests/asentmax_p3_incremental_g1.rs`, 5/5): after EVERY push, `(probs, τ, |support|)` == fresh `entmax_1p5(&history)` bit-for-bit, on: random mixed streams (2k pushes, parity every 128), **threshold-brushing** (exactly-at-τ, 1-ULP-below = no-event; 1-ULP-above = event), **spike-after-plateau** (the support-SHRINK case — rising staircase evicts plateau members to exactly 0.0), all-equal (worst case: every push events), descending/ascending monotone streams (descending stabilizes: 0 events after step ~64).

**G2 PASS**: realistic decode stream @ n→512k (5 spikes then N(0,1) bulk): **4 events over 524,288 pushes**, mean **0.046 µs/step**, tail (10k @ n≈512k) **0.015 µs/step** vs full-resort **68,448 µs/step** — the per-step cost is compare+write, not sort. (Full-resort checkpoints at 4k/16k include first-touch effects on the 1M-sized scratch; the 512k checkpoint gates.)

**G4 PASS** (`tests/asentmax_p3_alloc_check.rs`): below-τ steady state within the pre-sized reservation = 0 allocations (CountingAllocator, 1000 measured pushes).

**Verdict P2+P3: 🟢 PASS, stays opt-in** (same family flag `asentmax_schedule`; the window calc and incremental entmax are not yet wired into a production KV/decode path — the riir-ai consumer follow-up would be their hot-path gate).

## P2/P3 substrate check (substrate-first skill)

- Searched vocabulary variants: `eviction`/`window`/`kv_retention`/`prune` on attention paths (katgpt-kv `cache_prune::SummedAreaTable` is a different concern — saliency-based, not distance-windowed; `WallPrefixState::min_retention_at_block` is decay-based, not theorem-backed); `incremental`/`streaming` entmax (nothing — `entmax_1p5_into` is the full-resort baseline P3 replaces on the decode path).
- `AlibiAction` (katgpt-core `position_group_action`) pins the sign convention `b = −β·(t−j)` the window formula assumes — consumed as documentation parity, no code dep (the window is pure arithmetic).
