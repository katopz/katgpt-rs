# Bench 690 — `anti_common_mode` gate + `anchored_reach` operator (Issue 696 T1+T2)

**Issue:** `696` ·
**Research:** [433](../../riir-train/.research/433_RVM_Anchored_Velocity_Regression.md) (riir-train)
**Paper:** [arXiv:2608.23664](https://arxiv.org/abs/2608.23664) (RVM — velocity matching, Choi et al. 2026; §C.3 DT/DT2 reward + Eq. 7 anchored regression)
**Date:** 2026-08-28 · **Features:** `anti_common_mode` + `anchored_reach` (BOTH opt-in, independent)
**Modules:** `crates/katgpt-core/src/anti_common_mode.rs` (462 lines) + `crates/katgpt-core/src/anchored_reach.rs` (447 lines)
**Scope:** T1+T2 ONLY (primitive extraction + unit/perf gates). T3/T4/T5 stay open — promotion is T3's to grant.

## Verdict table

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| **G1 correctness** | T1: median cancels a population-dominant mode exactly; both band tails score 0; peak-quantile ≠ mean on a minority-active distribution. T2: the 5 regimes produce predicted outputs; bit-identical at A=1/A=0 with plain adoption/clamp; schedules in documented ranges | 9 + 8 unit gates green (see §G1); constant population cancels to `m' = +0.0` bit-exact for every c ∈ {0, 10, 1e6, −3.5}; `+1000` Sterbenz shift keeps `m' = 4.0` bit-exact; the A=1 pole pinned over a 16×16 value sweep **including subnormals/extremes** vs the demonstrated naive-form counterexample (`1e-30 + (1e-40 − 1e-30) = 0.0` — candidate LOST) | **PASS** |
| **G2 perf** | sub-µs at N=1000 (issue estimate); zero-alloc | `anchored_reach::blend` **234 ns/blend** @ N=1000 (0.23 ns/element, vectorized FMA loop) → **PASS**, 4× under; `anti_common_mode::score` **6023 ns/call** @ N=1000 → **FAIL at the literal ask** (see §G2 — two exact `select_nth_unstable` passes ≈ 6 ns/element; sub-µs holds at N ≲ 160) | **PASS + honest FAIL** (one axis each) |
| **G3 no-regression** | default-feature lib suite green; feature-on = default + module tests only | default debug **1978 passed / 0 failed / 7 ignored**; `anti_common_mode` debug **1987/0/8** (+10 = 9 unit + 1 G2), `anchored_reach` debug **1985/0/8** (+8 = 7 unit + 1 G2); release: anti **1980/0/6**, anchored **1978/0/6**; `cargo check -p katgpt-core` (default) clean; clippy **0 warnings** in all three states (default / each feature) | **PASS** |
| **G4 alloc-free** | zero allocs in scored/blended hot paths | debug TrackingAllocator: `score` + `peak_quantile` + `median` + `band_window` over N=1000 → **0 allocs**; `blend_scalar_into` + `blend_into` + `reach_scalar` + schedules → **0 allocs** (partition/scan/FMA only; the `&mut [f32]` reorder contract means no scratch exists to allocate) | **PASS** |
| **Promotion** | only via T3's consumer PoC | not requested — both features stay **opt-in**; the value claim (CLR crowd-panic re-enable: border-band at baseline AND Bench-010 detection retained) is T3's to measure | **DEFERRED by design** |

## G1 — unit gates

### anti_common_mode (9 tests)

- `g1_peak_quantile_is_top_five_percent_mean` — `[10×95, 14×5]` → peak **exactly 14.0** (k = ceil(5%·100) = 5, pivot included); multiset preserved through the permutation; N=1000 mixed fixture → top-50 mean ≈ 124.5 ± 1e-3.
- `g1_peak_quantile_differs_from_mean_on_minority_active` — `[0×95, 100×5]`: peak = 100.0 vs mean = 5.0 (the estimator contrast the issue requires; a mean-based statistic collapses into the quiet majority).
- `g1_median_cancels_population_dominant_mode_exactly` — constant populations cancel to `m' = +0.0` **bit-exact** for every common-mode value c (the anti-common-mode property); the +1000 Sterbenz shift keeps `m' = 4.0` bit-exact; even-N median = exact two-middle average (2.5 on [1,2,3,4]).
- `g1_band_tails_score_zero` — both extremes of the band → 0.0 exactly (raw `band_window` + end-to-end `score` with tail-shaped populations); a mid-band population earns 1.0 at `m' = τmid`.
- `g1_band_window_shape_rises_then_falls` — 0 → 0.5 → 1.0 → 0.5 → 0 across the band; degenerate bands (incl. the zero-τ band) and NaN m admit nothing.
- `g1_context_threshold_paper_parameterization` — `context_threshold(640/256)` = 15.0 reproduces the paper's `τ = 6·min(H,W)/256` for a 640×H frame; negative/zero/NaN context → degenerate band → score 0.
- `g1_empty_and_degenerate_refusals` — empty/single-element populations → 0; NaN-poisoned populations partition deterministically under `total_cmp` and refuse to a finite 0.0.
- `g1_statistic_is_permutation_invariant_and_multiset_preserving` — re-scoring a permuted buffer is identical; sorting the post-call buffer recovers the original multiset exactly (the `&mut` contract held).
- `g2_score_under_budget_at_n1000` / `g4_alloc_free_scored_path` — see G2/G4.

### anchored_reach (8 tests)

- `g1_five_regimes_produce_predicted_outputs` — clamp (exact anchor bits) / interior blend (4.0 at A=0.5 of [2,6]) / adopt (exact candidate bits) / overshoot (8.0 at A=1.5) / repel (0.0 at A=−0.5), scalar AND slice paths, axis-wise.
- `g1_bit_identity_at_poles` — 16×16 sweep × {A=1, A=0} × scalar + both slice paths, all `to_bits()` equal; **plus the pinned counterexample** proving the fast path load-bearing: `anchor=1e-30, cand=1e-40` → naive `anchor + 1.0·(cand−anchor)` = **0.0** (the subnormal is absorbed by the subtract and lost), fast path returns the candidate verbatim. A brute-force sweep found 36 such naive-form non-identical pairs in the subnormal/cancellation class — "plain adoption" is NOT floating-point-safe without the fast path.
- `g1_per_axis_matches_scalar_elementwise` — heterogeneous per-axis A (one of each regime class) equals elementwise `reach_scalar` bitwise; uniform per-axis A equals the scalar-A slice path bitwise.
- `g1_schedule_constructors_documented_ranges` — linear identity; `2σ(kr)−1` closed in [−1,1], monotone, exact 0.0 at r=0 (σ(0)=0.5 exactly), saturating ±1.0 at |kr|>40; sign-flip at β̄=1: r=0 → −1 (repel), r=1 → +1 (adopt), r=2 → +3 (overshoot ×3 — the paper's `2r−1 > β̄` prediction); dead schedules (k=0, β̄≤0, non-finite inputs) → 0.0; `clip_reward` pins the paper's ±5.
- `g1_schedule_feeds_operator_integration` — one row per schedule composed through `reach_scalar` (σ-blend / overshoot ×3 / clipped-linear repel).
- `g1_length_mismatch_panics` — `assert_eq!` on lengths (house `_into` convention).
- `g2_blend_under_budget_at_n1000` / `g4_alloc_free_blended_path` — see G2/G4.

## G2 — the honest breakdown

| Operator | Measured @ N=1000 (release, M3 Max) | Issue ask | Verdict |
|---|---|---|---|
| `anchored_reach::blend_scalar_into` + `blend_into` (one each per run) | **234 ns/blend** (0.23 ns/element) | sub-µs | **PASS** — the pole check hoists out of the loop (scalar-A form); the interior path is a branch-free FMA loop that auto-vectorizes |
| `anti_common_mode::score` | **6023 ns/call** (6.0 ns/element) | sub-µs | **FAIL at the literal ask** — structural, not accidental: exact top-5% + exact median need TWO `select_nth_unstable_by` passes (the issue's estimate assumed one O(N) pass); sub-µs holds at **N ≲ 160** |

The G2 tests pin regression floors at ~2× measured (12 µs / 5 µs), NOT the issue's sub-µs ask — the ask is recorded here as missed on one axis, with the per-element number as the honest scaling law. A caller needing sub-µs at N=1000 has two documented options: (a) sample the population (the statistic is permutation-invariant; the paper's own per-frame populations are ≫1000 with stable peak/median ratios), or (b) accept the ~6 µs — at the T3 consumer's cadence (CLR emotion tick, 20 Hz, 1000 NPCs) this is 0.012% of the tick budget, which is the comparison that actually matters.

Sigmoid note: `schedule_sigmoid` delegates to the crate's `fast_sigmoid` (Cephes, ~1 ULP, sign-branch stable) — the house form, sigmoid-not-softmax by construction (pointwise; no cross-group normalization, matching RVM which never normalizes across the group).

## G3 — validation matrix

| Suite | Debug | Release |
|---|---|---|
| default features | **1978 / 0 / 7i** | (1970 / 0 / 6i derived) |
| `--features anti_common_mode` | **1987 / 0 / 8i** (+10) | **1980 / 0 / 6i** (+10) |
| `--features anchored_reach` | **1985 / 0 / 8i** (+8) | **1978 / 0 / 6i** (+8) |

`cargo check -p katgpt-core` (default) clean · clippy **0 warnings** × 3 states (default / anti / anchored) · rustfmt clean on both files (edition 2024).

**Pre-existing flake encountered during validation (not this change — documented per the re-run rule):** `subspace_phase_gate::tests::jacobian_svd_r8x8_latency_gate` (debug) and `::thin_svd_rank_deficient_not_slower_than_full_rank` (release) each failed intermittently under ambient load and passed on re-run — `jacobian_svd` reproduced at pristine develop HEAD `850d3e45` (debug, 200 µs vs its budget) and `thin_svd` passed 3/3 in an interleaved branch-vs-base A/B after failing 2/3 in the anchored release run. Ambient-sensitive latency comparisons in an unrelated module; flagged for their owner, untouched here.

## G4 — alloc note

Debug-mode TrackingAllocator over the measured region (construction outside): `score` path (partition + scan + band) and `blend` path (copy/FMA) both **0 allocations**. Structural corroboration: every function is `&mut [f32]`-in-place or caller-owned-`out`; there is no `Vec`/`Box`/`String` in either module, so no allocation site exists (the G4 gate is belt-and-suspenders for the by-construction claim).

## Honest scope notes

- **Opt-in POC until T3.** The primitive exists to kill a measured failure (CLR crowd-panic: one monster panics the entire swarm to the borders — riir-mmorpg-examples Plan 019 demotion). Its value claim IS the T3 PoC (border-band occupancy at baseline AND Bench-010's 200-NPC distributed-threat detection retained); until that lands this is an unproven extraction, not a GOAT. `anchored_reach` additionally awaits its T4 consumer A/Bs (lead prediction / contrastive aversion / planner-as-anchor), each filed separately if adopted.
- **Band edges are configuration, not paper-derived.** τmid = τ (the paper's DT saturation point); τlo = τ/2, τhi = 2τ are ours ([`BAND_LO_FRAC`]/[`BAND_HI_FRAC`]) — the paper gives the band form without its edges. T3's consumer may re-pin them; the constants are the tuning surface.
- **No β(r) scale arm.** Eq. 7's scale c(r) is a training-side loss weight — routed to riir-train Plan 360 by Research 433, deliberately not shipped here.
- **Peak-quantile estimator confirmed** against Research 433: "mean magnitude of the fastest 5% of pixels" → mean of the top `k = max(1, ceil(0.05·N))` (P95 boundary element included). DT's saturation form `min(m/τ, 1)` is the band's τmid; DT2's median-subtraction-then-band-window is the composite `score`.
- **NaN contract is refuse-to-zero**, matching the crate's modelless gate style (`finish_finite`); `total_cmp` keeps a poisoned population deterministic (no panic, no iteration-order dependence).

## API surface shipped

```rust
// anti_common_mode (feature `anti_common_mode`)
pub const PEAK_FRAC: f32 /* 0.05 */; TAU_GAIN: f32 /* 6.0 */;
pub const BAND_LO_FRAC: f32 /* 0.5 */; BAND_MID_FRAC: f32 /* 1.0 */; BAND_HI_FRAC: f32 /* 2.0 */;
pub fn peak_quantile(values: &mut [f32]) -> f32;            // top-5% mean, in-place partition
pub fn median(values: &mut [f32]) -> f32;                   // exact both parities
pub fn context_threshold(context_scale: f32) -> f32;        // τ = 6·context
pub struct BandThresholds { tau_lo, tau_mid, tau_hi }
  BandThresholds::from_context(context_scale: f32) -> Self; BandThresholds::from_tau(tau: f32) -> Self;
pub fn band_window(m: f32, tau_lo: f32, tau_mid: f32, tau_hi: f32) -> f32;
pub fn score(values: &mut [f32], context_scale: f32) -> f32; // the composite DT2 gate

// anchored_reach (feature `anchored_reach`)
pub const RVM_CLIP: f32 /* 5.0 */;
pub fn clip_reward(r: f32) -> f32;
pub fn reach_scalar(anchor: f32, candidate: f32, a: f32) -> f32;   // bit-exact poles at A∈{0,1}
pub fn blend_scalar_into(anchor: &[f32], candidate: &[f32], a: f32, out: &mut [f32]);
pub fn blend_into(anchor: &[f32], candidate: &[f32], a: &[f32], out: &mut [f32]); // per-axis A
pub fn schedule_linear(r: f32) -> f32;
pub fn schedule_sigmoid(r: f32, k: f32) -> f32;             // 2σ(kr)−1 via fast_sigmoid
pub fn schedule_sign_flip(r: f32, beta_bar: f32) -> f32;    // (2r−1)/β̄
```

## Next

- T3 (open) — the CLR crowd-panic consumer PoC in the mmorpg harness; the only promotion path.
- T4 (open) — the three anchored-reach consumer A/Bs.
- T5 (open) — group-definition A/B sub-note, folds into whichever consumer lands first.

## Resolution — Issue 696 CLOSED (2026-08-29, all tasks done; file removed per the noise-reduction rule)

- **T3 PASS** — riir-ai `50591686c`, [Bench 794](../../riir-ai/.benchmarks/794_clr_anti_common_mode_poc.md): panic+gate border occupancy 27.1% → 0.0% (baseline level, 8/8 seeds) with Bench-010 detection retained bit-identically. **CLR re-promotion DECIDED: stays opt-in** — the G2-with-CLR perf axis is still unmeasured in the composed default path (~5 ms/tick collective scan at production scale, Issue 054 T9) and the Plan-019 demotion was a gameplay-feel call; the gate evidence is in hand for the owner to flip when they next play.
- **T4(a) MEASURED** — riir-ai `4915a0425`, [Bench 796](../../riir-ai/.benchmarks/796_belief_lead_contrastive_aversion_ab.md): the A=1 dead-reckon read beats frozen belief 2.5–5×; **overshoot (λ>0) REFUTED on turns**; adoption filed as riir-ai `Issue 777`.
- **T4(b) PASS** — same bench: A=−1 collapses acceptance ≤0.005, A=0 bit-unchanged; adoption filed as riir-ai `Issue 778`.
- **T4(c) ALREADY COVERED** — riir-ai `b05d3bd57`, [Bench 797](../../riir-ai/.benchmarks/797_guidance_anchor_follower_ab.md): the shipped leader-plans-once pattern owns the ≥4× economics (measured 768× fewer planner calls / 420× less wall); the (0,1) reach band adds nothing measured — no wiring.
- **T5 CONFIRMED** — Bench 796: Zone-vs-World standardization flips the emergent structure (incl. an aversion-direction inversion); folded into riir-ai Issue 778 T3/T4.
