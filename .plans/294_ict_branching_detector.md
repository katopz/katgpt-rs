# Plan 294: ICT Distributional Branching-Point Detector — Open Primitive

**Date:** 2026-06-19
**Research:** [katgpt-rs/.research/270_Beyond_Entropy_ICT_Distributional_Branching_Detector.md](../.research/270_Beyond_Entropy_ICT_Distributional_Branching_Detector.md)
**Private guide:** [riir-ai/.research/142_Distributional_Branching_Point_NPC_Guide.md](../../riir-ai/.research/142_Distributional_Branching_Point_NPC_Guide.md)
**Source paper:** [arxiv 2606.19771](https://arxiv.org/pdf/2606.19771) — Beyond Entropy / ICT framework (Feng et al., 18 Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/ict/` (new module) + Cargo feature `ict_branching`
**Status:** ✅ COMPLETE (Plan 294 implementation, 2026-06-19). All 8 phases shipped; G1/G3/G4/G5/G6/G10 PASS, G2 BORDERLINE-FAIL documented in Issue 033. Stays opt-in pending G8 (riir-ai Plan 324).

---

## Goal

Ship the public, generic, MIT-licensed modelless primitives that implement ICT's distributional branching-point detector: `collision_purity`, `renyi_h2`, `js_divergence`, `is_critical_branching`, `branching_point_mask`, and the `BranchingDetector` struct. Run the GOAT gates G1–G6 and G10 (defined in `riir-ai/.research/142`).

**Critical:** G3 (orthogonality to H1, ρ < 0.5) is the make-or-break. If it fails, **stop** — downgrade the Super-GOAT verdict to Gain, demote `ict_branching` to opt-in, and keep only the H1→H2 upgrade for Bebop (G10) and Curiosity Pulse (riir-ai Plan 274). Run G3 before G7–G9 (riir-ai Plan 324).

No game IP. No chain IP. Pure information-theoretic math + selector + EMA tracker. The private selling-point guide (R142) and runtime fusion (riir-ai Plan 324) consume this primitive.

---

## Constraints (per AGENTS.md + research skill)

| Constraint | How this plan satisfies it |
|---|---|
| Modelless / inference-time | All primitives are pure functions on distribution arrays; zero backprop, zero weight mutation. |
| Latent-to-latent preferred | Operates on probability simplices; decodes nothing — outputs are derived scalars + a 1-bit mask. |
| Sigmoid, not softmax | Top-k% selector is a hard sigmoid on uniqueness scores. The probability simplices themselves can come from softmax-of-logits — that's *representation*, the rule against softmax is for projection gates onto direction vectors (which we don't do here). |
| Zero-alloc hot path | All `*_into` variants take pre-allocated scratch buffers; `_view` variants borrow slices. |
| CPU/SIMD/GPU auto-route | Detection fits in L1 cache → CPU/SIMD only. No GPU needed for K ≤ 32 trajectories. |
| Feature flag isolation | `ict_branching` is opt-in (NOT default-on) until GOAT gate G3 passes and downstream riir-ai Plan 324 promotes the fusion. |

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/ict/mod.rs` with module declarations and feature-gated exports
- [x] **T1.2** Add Cargo feature `ict_branching = []` to `katgpt-rs/Cargo.toml` (opt-in, NOT in `default = [...]`)
- [x] **T1.3** Wire `pub mod ict;` in `katgpt-rs/crates/katgpt-core/src/lib.rs` behind `#[cfg(feature = "ict_branching")]`
- [x] **T1.4** Create `katgpt-rs/crates/katgpt-core/src/ict/math.rs`:
  - `collision_purity(probs: &[f32]) -> f32` — `Σ p²` with manual SIMD-friendly accumulation
  - `collision_purity_into(probs: &[f32], out: &mut f32)` — zero-alloc variant
  - `renyi_h2(probs: &[f32]) -> f32` — `−log Σ p²`
  - `shannon_h1(probs: &[f32]) -> f32` — for G3 baseline comparison
  - `js_divergence(p: &[f32], q: &[f32], scratch_m: &mut [f32]) -> f32` — symmetric JS using `m = (p+q)/2` scratch
  - `js_divergence_batch<'a>(dists: &[&[f32]], scratch_m: &mut [f32]) -> Vec<f32>` — pairwise to mean
- [x] **T1.5** Create `katgpt-rs/crates/katgpt-core/src/ict/branching.rs`:
  - `is_critical_branching(prob_of_action: f32, beta: f32, eta: f32) -> bool` — `|π(a*) − β| < η`
  - `branching_point_mask(uniqueness_scores: &[f32], k_percent: f32, mask: &mut [bool])` — top-k% selector writing into pre-allocated mask
  - `branching_point_mask_into(uniqueness_scores: &[f32], threshold: f32, mask: &mut [bool])` — threshold-based variant
- [x] **T1.6** Create `katgpt-rs/crates/katgpt-core/src/ict/detector.rs`:
  - `pub struct BranchingDetector { k_trajectories, action_dim, k_percent, eta, scratch_p_avg, scratch_m, scratch_u, scratch_mask, ema_beta, ema_alpha }`
  - `fn new(k_trajectories, action_dim, k_percent, eta) -> Self` — pre-allocates all scratch
  - `fn observe_and_detect(&mut self, trajectories: &[&[f32]]) -> BranchingReport` — main entry; returns per-step mask + per-step β
  - `fn reset(&mut self)` — clear EMA
- [x] **T1.7** Create `katgpt-rs/crates/katgpt-core/src/ict/types.rs`:
  - `pub struct BranchingReport { mask: Vec<bool>, beta_per_step: Vec<f32>, uniqueness_scores: Vec<f32> }`
  - Doc-comment referencing R270 §2.3 and the proof of correctness
- [x] **T1.8** Unit tests in `katgpt-rs/crates/katgpt-core/src/ict/math.rs` (inline `#[cfg(test)]`):
  - `collision_purity_uniform_distribution` — `[1/n; n]` → `1/n`
  - `collision_purity_degenerate` — `[1, 0, 0, ...]` → `1.0`
  - `collision_purity_known_value` — `[0.5, 0.5]` → `0.5`
  - `renyi_h2_uniform` — matches `log(n)`
  - `js_divergence_identical` → `0.0`
  - `js_divergence_disjoint` → `log(2)` (bounded)
  - `js_divergence_symmetric` — `js(p, q) == js(q, p)`
  - `shannon_h1_uniform` — matches `log(n)` (for G3 baseline)

---

## Phase 2 — GOAT Gate G1 (paper-proof — must always pass)

### Tasks

- [x] **T2.1** Create `katgpt-rs/tests/bench_294_ict_g1.rs` with `g1_distributional_discrimination` test:
  - Construct two distributions with **identical H1 but different β** (paper Figure 1a construction):
    - `p_A = [0.5, 0.5, 0, 0, 0, 0]` — H1 = ln 2, β = 0.5
    - `p_B` constructed to have H1 = ln 2 but β ≠ 0.5 (e.g., via Lagrange multiplier)
  - Assert `collision_purity(p_A) != collision_purity(p_B)` within tolerance
  - Assert `shannon_h1(p_A) ≈ shannon_h1(p_B)` within tolerance
  - **Pass criterion:** β distinguishes where H1 cannot. Always passes (paper proof of capability).
- [x] **T2.2** Add 3 additional paper-proof test cases covering the bifurcation regimes:
  - `regime_h_collapse`: `π(a*) = 0.9, β = 0.82` (π > β) → `is_critical_branching(0.9, 0.82, 0.05)` returns false (collapse regime, not a branching point)
  - `regime_l_explosion`: `π(a*) = 0.05, β = 0.5` (π < β) → returns false (explosion regime)
  - `critical_branching`: `π(a*) = 0.5, β = 0.5, η = 0.05` → returns true
- [x] **T2.3** Document G1 results in `katgpt-rs/.benchmarks/294_ict_g1.md` — short, paper-proof-only

---

## Phase 3 — GOAT Gate G2 (inflection at ~10%)

### Tasks

- [x] **T3.1** Create `katgpt-rs/tests/bench_294_ict_g2.rs`:
  - Synthetic NPC-decision suite: 1000 decision points, each with K=8 candidate trajectories sampled from a controlled-mixture policy (one of three regimes: committed / undecided / noise)
  - For each decision point: compute `u_{k,s}` for each `k`, sort, find the inflection point (second-derivative peak)
  - Plot histogram of inflection locations across 1000 points
- [x] **T3.2** Assert that the median inflection location is in `[5%, 20%]` (paper §A.4.1 reports ~10%)
- [x] **T3.3** Document G2 results in `katgpt-rs/.benchmarks/294_ict_g2.md`:
  - Inflection histogram (ASCII art acceptable)
  - Median + IQR
  - If median > 30% or absent: **STOP, file issue**, re-evaluate (paper claim may not transfer from LLM to NPC domain)

---

## Phase 4 — GOAT Gate G3 (orthogonality to H1 — MAKE-OR-BREAK)

### Tasks

- [x] **T4.1** Create `katgpt-rs/tests/bench_294_ict_g3.rs`:
  - Same synthetic suite as G2
  - For each decision point: compute `h1_s = shannon_h1(p_avg)` and `u_max_s = max_k u_{k,s}`
  - Collect paired samples `(h1_s, u_max_s)` across all 1000 × step points
  - Compute Spearman rank correlation ρ
- [x] **T4.2** Assert **ρ < 0.5** (Super-GOAT proceeds). If ρ ≥ 0.9 → fail, downgrade verdict.
- [x] **T4.3** Document G3 results in `katgpt-rs/.benchmarks/294_ict_g3.md`:
  - Scatter plot (ASCII art acceptable)
  - Spearman ρ + 95% CI (bootstrap)
  - **Verdict block:** "Super-GOAT PROCEEDS — JS captures structurally-different information from H1" OR "DOWNGRADE to Gain — H1 already captures the signal"

**Downgrade path if G3 fails:**
1. Update `katgpt-rs/.research/270_*.md` §3 verdict from "Super-GOAT" to "Gain"
2. Update `riir-ai/.research/142_*.md` §TL;DR with downgrade note
3. Cancel Plan 324 (riir-ai runtime fusion) — not worth the engineering cost
4. Keep T1.4-T1.7 (the primitives are still useful for the H1→H2 upgrade in G10)
5. File issue in `katgpt-rs/.issues/` documenting the downgrade + reason

---

## Phase 5 — GOAT Gates G4-G6 (perf, alloc, isolation)

### Tasks

- [x] **T5.1** Create `katgpt-rs/benches/bench_294_ict_perf.rs` (G4 — hot-path cost):
  - K=8 trajectories, action_dim=32
  - `BranchingDetector::observe_and_detect` 10K iterations
  - `std::time::Instant` (not criterion — see Bench 284 deviation note for rationale)
  - Report mean, p50, p99 µs
- [x] **T5.2** Create `katgpt-rs/tests/bench_294_ict_g5.rs` (G5 — zero heap alloc):
  - Use `katgpt_rs::alloc::TrackingAllocator` (debug-only, existing)
  - Warmup: 100 ticks (allocates scratch)
  - Measure: 1000 ticks — assert per-call allocs = 0
- [x] **T5.3** Create `katgpt-rs/tests/bench_294_ict_g6.rs` (G6 — feature isolation):
  - `cargo build --no-default-features --features ict_branching` succeeds
  - `cargo build --no-default-features` succeeds
  - `nm target/release/libkatgpt_rs.dylib | grep -i ict_branching` → zero matches in disabled build
- [x] **T5.4** Document G4-G6 results in `katgpt-rs/.benchmarks/294_ict_goat_gates.md`

**Pass criteria:**
- G4: ≤ 50µs per `observe_and_detect` call at K=8, action_dim=32 (plasma budget)
- G5: 0 allocs per call after warmup
- G6: feature isolation confirmed via `nm`

---

## Phase 6 — GOAT Gate G10 (Bebop H1→H2 upgrade — Drop-in)

### Tasks

- [x] **T6.1** Create `katgpt-rs/src/ict/bebop_upgrade.rs` (behind `ict_branching`):
  - `AcceptanceForecastH2 { a: f32, b: f32, ema_beta: f32, ema_alpha: f32 }`
  - `fn observe_and_forecast(&mut self, next_token_logits: &[f32]) -> f32` — uses `β = collision_purity(softmax(logits))` instead of Bebop's H1
  - `fn adaptive_gamma(&self, target_accept_length, gamma_min, gamma_max) -> usize` — same as Bebop R243 §4 sketch
- [x] **T6.2** Create `katgpt-rs/tests/bench_294_ict_g10.rs`:
  - Calibrate `AcceptanceForecastH2` on a workload with 50% `max π > 0.37` and 50% `max π < 0.37`
  - Compare mean forecast error: H1 forecast (Bebop baseline) vs H2 forecast (this primitive)
  - Assert H2 has lower mean error overall, with improvement concentrated in `< 0.37` regime
- [x] **T6.3** Document G10 results in `katgpt-rs/.benchmarks/294_ict_g10.md`
- [x] **T6.4** If G10 passes: amend `katgpt-rs/.research/243_Bebop_*.md` Issue 023 with H1→H2 upgrade recommendation

---

## Phase 7 — Curiosity Pulse H1→β Upgrade Spec (Reference Only)

> **Note:** The actual implementation lives in riir-ai (R041 / Plan 274 / Plan 187). This phase documents the public spec only.

- [x] **T7.1** Add doc-comment block in `katgpt-rs/crates/katgpt-core/src/ict/math.rs` referencing R041 (Curiosity Pulse) and showing the drop-in:
  ```rust
  // Curiosity Pulse (R041) currently uses:
  //   u_i(t) = shannon_h1(relevance_scores)
  // Drop-in upgrade per ICT §1.5 + A.3.3:
  //   u_i(t) = collision_purity(relevance_scores)  // = β
  // H1 is "blind exploration" (ICT §1); β captures concentration — the right
  // curiosity trigger. ∂H_2/∂π(a) < 0 unconditionally; H1 only valid for π > e⁻¹.
  ```
- [x] **T7.2** Add example `katgpt-rs/examples/ict_curiosity_pulse_upgrade.rs` showing the drop-in call site

---

## Phase 8 — Documentation + Promotion Decision

### Tasks

- [x] **T8.1** Update `katgpt-rs/README.md` Feature Showcase with Plan 294 section (gated behind `ict_branching`, opt-in)
- [x] **T8.2** Update `katgpt-rs/.docs/01_overview.md` feature table with `ict_branching` row
- [x] **T8.3** Add 3 example files:
  - `examples/ict_minimal.rs` — basic `collision_purity` + `js_divergence` walkthrough
  - `examples/ict_branching_detector.rs` — `BranchingDetector::observe_and_detect` on synthetic trajectories
  - `examples/ict_paper_figure_1a.rs` — reproduces paper Fig 1a (two distributions, same H1, different β)
- [x] **T8.4** **Promotion decision:**
  - If G3 passes AND G8 (riir-ai Plan 324) passes: promote `ict_branching` to default-on
  - If G3 fails: leave opt-in; only the H1→H2 upgrade (T6.x) is broadly useful
  - If G3 passes but G8 fails: leave opt-in; document why in `katgpt-rs/.benchmarks/294_ict_promotion.md`
- [x] **T8.5** Cross-link `katgpt-rs/.research/270_*.md` and `riir-ai/.research/142_*.md` with the GOAT gate results once complete

---

## Files (new)

| File | LOC | Purpose |
|------|-----|---------|
| `katgpt-rs/crates/katgpt-core/src/ict/mod.rs` | ~25 | Module declarations + feature-gated re-exports |
| `katgpt-rs/crates/katgpt-core/src/ict/math.rs` | ~180 | `collision_purity`, `renyi_h2`, `shannon_h1`, `js_divergence` + 8 unit tests |
| `katgpt-rs/crates/katgpt-core/src/ict/branching.rs` | ~120 | `is_critical_branching`, `branching_point_mask` + 6 unit tests |
| `katgpt-rs/crates/katgpt-core/src/ict/detector.rs` | ~250 | `BranchingDetector` struct + `observe_and_detect` + 10 unit tests |
| `katgpt-rs/crates/katgpt-core/src/ict/types.rs` | ~60 | `BranchingReport` + doc-comments |
| `katgpt-rs/src/ict/bebop_upgrade.rs` | ~120 | `AcceptanceForecastH2` (Bebop H1→H2 drop-in) |
| `katgpt-rs/tests/bench_294_ict_g1.rs` | ~120 | G1 paper-proof test |
| `katgpt-rs/tests/bench_294_ict_g2.rs` | ~180 | G2 inflection test |
| `katgpt-rs/tests/bench_294_ict_g3.rs` | ~180 | G3 orthogonality test (MAKE-OR-BREAK) |
| `katgpt-rs/tests/bench_294_ict_g5.rs` | ~80 | G5 zero-alloc test |
| `katgpt-rs/tests/bench_294_ict_g6.rs` | ~60 | G6 feature isolation test |
| `katgpt-rs/tests/bench_294_ict_g10.rs` | ~150 | G10 Bebop H1→H2 upgrade test |
| `katgpt-rs/benches/bench_294_ict_perf.rs` | ~100 | G4 hot-path perf benchmark |
| `katgpt-rs/examples/ict_minimal.rs` | ~80 | Example: basic primitives |
| `katgpt-rs/examples/ict_branching_detector.rs` | ~120 | Example: BranchingDetector on synthetic |
| `katgpt-rs/examples/ict_paper_figure_1a.rs` | ~80 | Example: reproduce paper Fig 1a |
| `katgpt-rs/examples/ict_curiosity_pulse_upgrade.rs` | ~80 | Example: R041 drop-in reference |

**Estimated total:** ~2,000 LOC including tests + examples

---

## Cargo.toml changes

- Add `ict_branching = []` to `[features]` (NOT in `default = [...]`)
- Add `[[test]]` entries for `bench_294_ict_g1` through `bench_294_ict_g10`
- Add `[[bench]]` entry for `bench_294_ict_perf`
- Add 4 `[[example]]` entries

---

## Dependencies

- None new. Uses only `std`. SIMD via manual unroll (4 or 8 chunks for f32 dot-products).

---

## Validation Protocol (recap from R142 §5)

| Gate | Test | Pass criterion | Phase |
|------|------|----------------|-------|
| G1 | `bench_294_ict_g1.rs` | β distinguishes where H1 cannot (paper proof) | 2 |
| G2 | `bench_294_ict_g2.rs` | Median inflection in [5%, 20%] | 3 |
| G3 | `bench_294_ict_g3.rs` | Spearman ρ(H1, JS-uniqueness) < 0.5 (**MAKE-OR-BREAK**) | 4 |
| G4 | `bench_294_ict_perf.rs` | ≤ 50µs per call at K=8, action_dim=32 | 5 |
| G5 | `bench_294_ict_g5.rs` | 0 allocs/call after warmup | 5 |
| G6 | `bench_294_ict_g6.rs` | Feature isolation via `nm` | 5 |
| G10 | `bench_294_ict_g10.rs` | H2 forecast beats H1 on `< 0.37` regime | 6 |

G7 (latent/raw boundary), G8 (ICT × CLR fusion), G9 (ICT × HLA fusion) → **riir-ai Plan 324**.

---

## Risks

| Risk | Mitigation |
|------|-----------|
| G3 fails (ρ ≥ 0.9) | Downgrade R270 to Gain. Keep T1-T6 (primitives + H1→H2 upgrade). Cancel Plan 324. File issue. |
| G2 fails (no 10% inflection) | The 10% is LLM-token-specific. Sweep k% to find our inflection. May be 20-30% for NPCs. Document in T3.3. |
| `js_divergence` SIMD autovectorization fails | Manual 4-way chunk unroll on the inner `Σ p·log(p/m)` loop. If still slow, GPU route via existing `katgpt-rs/src/device_selector.rs`. |
| Bebop H1→H2 upgrade (G10) shows no improvement | H2 unconditionally valid (proven), but practical magnitude may be small if LLM top-tokens are mostly > 0.37. Document either way in T6.3. |
| Scope creep into riir-ai runtime | HARD LINE: nothing in this plan touches riir-ai. Plan 324 owns the runtime fusion. |

---

## Implementation Order

T1 (skeleton + primitives) → T2 (G1 paper-proof) → T3 (G2 inflection) → T4 (G3 orthogonality) → **decision point** → T5 (G4-G6 perf) → T6 (G10 Bebop upgrade) → T7 (Curiosity Pulse spec) → T8 (docs + promotion)

**If T4 fails: skip T5 perf and T8 promotion. Keep T6 (H1→H2 upgrade is independently valuable).**

---

## TL;DR

Ship the public, generic ICT primitives (`collision_purity`, `renyi_h2`, `js_divergence`, `is_critical_branching`, `branching_point_mask`, `BranchingDetector`) behind feature flag `ict_branching`. Run GOAT gates G1-G6 and G10 (G7-G9 live in riir-ai Plan 324). **G3 (Spearman ρ(H1, JS-uniqueness) < 0.5) is the make-or-break** — if it fails, downgrade R270 from Super-GOAT to Gain, cancel Plan 324, and keep only the H1→H2 upgrade for Bebop (Plan 243) and Curiosity Pulse (riir-ai Plan 274). No game IP, no chain IP. Pure math + selector + EMA tracker.
