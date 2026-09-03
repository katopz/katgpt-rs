//! katgpt-core: Shared types and SIMD kernels for katgpt-rs and riir-engine.
//!
//! This crate contains the common core shared between the two projects:
//! - **types**: Config, Rng, math utilities, LoRA, DomainLatent
//! - **simd**: NEON/AVX2 accelerated linear algebra kernels
//! - **hla**: Higher-order Linear Attention substrate (cache types + kernels)
//! - **mcts**: Generic Monte Carlo Tree Search over any `GameState`
//! - **delta_mem**: δ-mem associative memory substrate (state, hasher, multi-domain)
//! - **traits**: Shared traits for game AI and speculative decoding
//! - **speculative**: Speculative-decoding substrate types + sampling primitives
//!   (TreeNode, DraftResult, configs, LDT conflict detector, TES credit
//!   assignment, CDF/residual samplers)
//!
//! No feature flags on types/simd/hla/mcts/delta_mem/speculative — both projects
//! get the full substrate. Composition layers (root-only types like
//! `BanditRolloutPolicy`, `MemorySteeredPruner<P>`) stay in the consuming crate.

/// Standard logistic sigmoid: `σ(x) = 1 / (1 + e^{-x})`.
///
/// Numerically stable sigmoid σ(x) = 1/(1+e^{-x}), output in (0, 1).
///
/// Delegates to [`simd::fast_sigmoid`] (Cephes 6th-order polynomial exp,
/// ~1 ULP accurate, ~1.7× faster than libm `exp` on aarch64). The two-branch
/// form that previously lived here was a workaround for libm `exp`'s overflow
/// behavior; `fast_sigmoid` handles this internally via ±40 early-exits.
///
/// Always available — no feature gate — because it's a pure math utility
/// consumed across many domains (band conditioning, CGSP, faithfulness gates,
/// personality composition, etc.). Hoisted here from `band_conditioner::sigmoid`
/// (Proposal 003 Phase 0.1) so the upcoming `katgpt-band` extraction doesn't
/// drag a math utility into the band crate. Per the project rule: sigmoid,
/// never softmax.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    simd::fast_sigmoid(x)
}

/// Noisy-OR span aggregation `1 − Π(1−kᵢ)`, direct product form.
///
/// Generalized from the riir-games-civ salience gate's literal
/// `1 − (1−c)(1−boost)` (Research 491 distilled item 5 / Issue 672 T4 — the
/// civ site delegates here). Inputs are clamped to `[0, 1]` (each k is a
/// probability-style weight). Boundary identities: all k = 0 ⇒ exactly 0;
/// any k = 1 ⇒ exactly 1; monotone non-decreasing in every k.
///
/// **Bit-compatibility note:** for the two-term case the accumulation is
/// exactly `(1.0−k₀)·(1.0−k₁)` in source order — bit-identical to the civ
/// inline formula it replaces (pinned by the sterling module's test).
///
/// Always available — no feature gate — same rationale as [`sigmoid`]: a
/// pure math utility consumed across domains, with a DEFAULT-ON consumer
/// (civ salience gate) that must not gain a feature dependency for a
/// behavior-preserving refactor.
#[inline]
pub fn noisy_or(ks: &[f32]) -> f32 {
    let mut acc = 1.0f32;
    for &k in ks {
        let one_minus = 1.0 - k.clamp(0.0, 1.0);
        acc *= one_minus;
    }
    1.0 - acc
}

/// Log1p-stable noisy-OR for spans of MANY small weights:
/// `1 − exp(Σ ln(1−kᵢ)) = −expm1(Σ log1p(−kᵢ))`.
///
/// The direct product form underflows to `1 − 0 = 1` prematurely when
/// enough small factors accumulate (catastrophic for long spans of tiny
/// probabilities); the log1p/expm1 form keeps resolution. Inputs clamped
/// to `[0, 1]`: k = 1 yields `log1p(−1) = −∞` ⇒ the stable form returns
/// exactly 1.0 — the correct saturated limit.
#[inline]
pub fn noisy_or_stable(ks: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &k in ks {
        let kc = k.clamp(0.0, 1.0);
        sum += (-kc).ln_1p(); // ln(1 − k), stable for small k
    }
    -sum.exp_m1()
}

#[cfg(feature = "tiled_attention")]
pub mod attention;

// Newton-Schulz orthogonalization + Muon momentum (Plan 152, Research 114,
// GOAT 25/25 Bench 050). Pure substrate — self-contained f32 linear algebra,
// zero crate:: deps, zero external deps. Extracted from katgpt-rs/src/ per
// Issue 355 Phase 1a. Re-exported by katgpt-rs root so historical
// `katgpt_rs::newton_schulz::*` paths continue to resolve.
#[cfg(feature = "newton_schulz")]
pub mod newton_schulz;

// float_order — NaN-safe total-order f32/f64 comparators (generalizes riir-rag
// score_cmp_desc; the workspace partial_cmp-unwrap_or(Equal) panic class):
// NaN sinks under both directions, -0.0 ties +0.0, NaN-free ordering is
// identical to the replaced idiom. Always compiled (ungated) — sorts run in
// every build, including --no-default-features consumers (riir-ai Issue 832
// T1: the cfg had landed on this module instead of `rating`, inverting both
// modules' documented gating).
pub mod float_order;

// rating — Elo + Plackett-Luce rating primitives (Issue 686, promoted from
// four in-stack copies: katgpt-pruners arena EloCalculator + proof
// lambda_to_elo, riir-ai riir-games ruliology ParadigmRanking, riir-clippy
// src/elo.rs). Pure modelless arithmetic; bit-identical expression trees;
// f64 canonical + f32 twins (riir-clippy's persisted ratings keep their
// exact numerics). Zero-cost-unless-invoked.
#[cfg(feature = "rating")]
pub mod rating;

// laprop — LaProp normalize-before-accumulate momentum (Issue 689, from
// riir-train Research 428 C2/C3): EMA over RMS-normalized intake with the
// closed-form accumulator bound |m| ≤ 1/√(1−ν) — downstream accumulators
// delete their clamps and get a theorem instead. Vector + scalar twins;
// opt-in until a consumer adopts.
#[cfg(feature = "laprop")]
pub mod laprop;

// prover_selection — prover-selection statistics kernels (Issue 692,
// from Research 509 — arXiv:2410.08146): T1 ships the D/Al complementarity
// selector (distinguishability + alignment + Theorem 3.1's γ·(D+Al)
// predicted-gain bound + its sigmoid-gated exposure), T2 the first_pit
// changepoint kernel (first index where Q̂ < ε) consumed by riir-clippy's
// PAV data curation (riir-train Plan 356 A1) via a same-signature twin.
// T3 ships the K* interior-optimum law — k_star (the closed-form rollout
// count maximizing the gap) + bok_advantage (the BoK gap
// A(K) = (1−V)^K − (1−Q)^K), gate-pinned against the empirical argmax on
// an exhaustive (Q,V) grid. Pure modelless arithmetic, no deps, no
// allocs, zero-cost-unless-invoked.
//
// DEFAULT-ON since 2026-08-27 (Bench 684 GOAT G1–G5 PASS — the T5
// head-to-head: D+Al-ranked prover selection beats strength-ranked on a
// controlled PAV harness, 16 seeds; the `rating` precedent — pure math,
// no dep surface, zero-cost-unless-invoked). The Cargo feature remains as
// an inert alias.
pub mod prover_selection;

// linking_fold — Linking-Number Detector + Fold Correction (Plan 410,
// Research 391, arXiv:2606.31856 Ren & Lim ICML 2026). SPLIT (Plan 410 T4.4
// Option C, 2026-07-07) into two independently-gated sub-features:
//   - linking_fold_fold     (hot-path |x−c| fold correction) — DEFAULT-ON
//   - linking_fold_detector (cold-path Algorithm-1 linking detector) — opt-in
//   - linking_fold          (umbrella = fold + detector) — opt-in
// The fold passes every GOAT gate modellessly and ships default-on; the
// detector's G2 budget is the audit-cadence-appropriate 500 ms @ n=2×200
// (Issue 050 Option A, resolved 2026-07-07) and it stays opt-in. The
// module root exists when EITHER sub-feature is on; submodules gate their own
// parts.
#[cfg(any(feature = "linking_fold_fold", feature = "linking_fold_detector"))]
pub mod linking_fold;
// best_belief — ε-quantile Beta lower bound for conservative selection
// (Plan 336, Research 320, RQGM arXiv:2606.26294 Prop. 4). Complements
// `sample_beta` (Thompson sampling for EXPLORATION) with a conservative
// EXPLOITATION / SELECTION counterpart. DEFAULT-ON (Phase 2 G2-unblock,
// 2026-06-28): LUT hot path 3.38ns, G1 3.099e-5<1e-4, G4 0 allocs.
#[cfg(feature = "best_belief")]
pub mod best_belief;
#[cfg(feature = "best_belief")]
pub use best_belief::{best_belief_score, best_belief_scores, select_best_belief};
// hint_regret — paired-rollout value-of-information estimation (Plan 576,
// Research 496 SPADE arXiv:2608.19197 / Research 500 EnvHarness, game-side
// guide riir-ai .research/340). The frontier-curriculum discriminator: how
// much would ONE hint (demo / revealed arm) improve the return on this
// content? Paired CRN estimator (Welford, zero-alloc) + Hoeffding /
// empirical-Bernstein machinery + sigmoid band gate + three-regime triage
// + Beta-LCB frontier ordering (DRY over best_belief) + regret-scored
// memory with absorbing-intractable eviction. Opt-in pending the GOAT
// verdict; consumers: riir-ai Guide 340 P0 (supersedes the mmorpg inline
// collapse), riir-train Plan 346 arena opponent selection.
#[cfg(feature = "hint_regret")]
pub mod hint_regret;
// risk_control_exit — modelless dual-threshold compute-exit (Plan 575,
// Research 494, "Conformal Thinking" arXiv:2602.03814): stop-when-confident
// upper threshold + parametric stop-when-not-progressing lower schedule,
// offline UCB/Hoeffding calibration with guaranteed realized exit-risk,
// efficiency-loss selection among feasible pairs, App. C disarm tripwire.
// Opt-in pending the Bench 681 GOAT.
#[cfg(feature = "risk_control_exit")]
pub mod risk_control_exit;
// distributional_steering — population steering toward a measure-defined
// target (Plan 577, Research 505, arXiv:2608.08770): MeasureReward
// first-variation table + Feynman-Kac log-weights with the mean-field Ψ̇
// Picard correction + weighted empirical measure exposure. Opt-in pending
// the Bench 682 targeting gate.
#[cfg(feature = "distributional_steering")]
pub mod distributional_steering;
// twist_cache — Plan 581 opaque-reward twisted-SMC steering + modelless
// twist amortization (Research 517, arXiv:2605.23346 CDM): x̂₀ posterior-mean
// reward proxy (1 query/particle-step vs M rollouts), BLAKE3 state-keyed
// value memo, one-shot ridge readout table, β/KL-budget selection
// (entropic_tilt hoist rule). Consistency for any positive ψ — amortization
// is variance reduction, never correctness. Opt-in; the trained-head
// counterpart is riir-train Plan 361 (same GOAT gate, two arms — Bench 692).
#[cfg(feature = "twist_smc")]
pub mod twist_cache;
// entropic_tilt — KL-budgeted max-seeking advantage tilt (TTT-Discover
// arXiv:2601.16175; prior art RS-GRPO / RSPO). The max-seeking counterpart to
// `best_belief` above: that scores a candidate from its OWN history counts,
// this scores it from the SHAPE of the current group. Shared math with exactly
// two consumers — riir-clippy `selection_entropic` (Issue 026, ranking) and
// riir-train `loss_grpo` (Plan 341, gradient scaling) — hoisted here rather
// than forked. Opt-in pending the Plan 341 Phase 2/3 GOAT.
#[cfg(feature = "entropic_tilt")]
pub mod entropic_tilt;
#[cfg(feature = "entropic_tilt")]
pub use entropic_tilt::{
    KL_BUDGET_LN2, solve_beta, tilt_advantages_into, tilt_advantages_loo_into, tilted_weights,
};
// tether — closed-form outcome-fit estimator blend (Issue 675, Research 426
// via arXiv:2608.16739 "Le Critique" TETHER baseline): ρ* per window by OLS
// against realized outcomes with the exact in-sample never-worse guarantee,
// the lag law encoded as API shape (same-window application unrepresentable),
// EMA smoothing, an explained-variance accumulator + control-variate gate,
// and the horizon-decay LUT λ = c^(1/L). Two documented hazards in-source:
// Report-the-Floor (blending does not discharge the promotion gate) and
// prediction-vs-ranking (measured NEGATIVE on a ranking consumer, Bench 042
// — fit ρ against the consumer's own metric). Consumers: riir-train
// loss_grpo TETHER baseline (Plan 345). Opt-in; G1–G4 PASS (Bench 670).
#[cfg(feature = "tether")]
pub mod tether;
#[cfg(feature = "tether")]
pub use tether::{
    control_variate_improves, fit_rho, horizon_decay, sse, EvAccumulator, TetherBlend, TetherStats,
    DEFAULT_EMA_DECAY, DEFAULT_RHO, DEFAULT_WINDOW, DEGENERATE_EPS,
};
// ignition — closed-form logistic ignition (Issue 459 T5, Research 422 §3.5
// via arXiv:2608.13335): z(t) = K·σ(ζt − ln((K−z₀)/z₀)) per singular mode,
// the patience law t* = ln(1/ε)/ζ, and a ζ-descending ordering helper
// (modes ignite in ζ order). Sigmoid-in-time is the adoption shape GD itself
// produces — the second grounding for sigmoid-not-softmax (after R315).
// Opt-in; GOAT G1–G4 PASS (Bench 666). Promotion needs the consumer pilot
// win (selection patience ∝ 1/ζ vs fixed patience).
#[cfg(feature = "ignition_schedule")]
pub mod ignition;
#[cfg(feature = "ignition_schedule")]
pub use ignition::{IgnitionSchedule, ignition_time, order_by_ignition_into};
// Conformal Predictive Intervals — modelless UQ overlay (Plan 340, Research
// 322, arXiv:2605.03789 CSP + arXiv:2606.09473 "Report the Floor"). Wraps any
// PointForecaster with a per-channel × per-horizon-bucket exp-recency-
// weighted residual ring buffer, reads empirical quantiles to produce
// coverage-guaranteed predictive intervals. The
// ConformalIntervalCalibrator<SeasonalNaiveForecaster> with m=1 is the
// canonical conformal-naive floor per the "Report the Floor" rule (Issue 010,
// AGENTS.md Feature Flag Discipline). DEFAULT-ON (Plan 468 promotion,
// 2026-07-20): primitive-level G1–G4 GOAT PASSed (Bench 340, 2026-06-30);
// runtime-consumer promotion gate satisfied by Bench 564 (MCTS collapse) +
// Bench 565 (Salience Tri-Gate ΔF1=+0.3145 at 6.3× margin). Plan 513 width-
// definition fix vindicated Bench 565 bit-identically. Consumer-level gates
// (riir-engine karc_conformal_width + salience_conformal_width + 4 probes)
// STAY opt-in — this promotion removes the katgpt-core re-forward friction
// only; consumers still choose.
#[cfg(feature = "conformal_predictive_intervals")]
// Issue 580: the LIMIT adversarial retrieval fixture (arXiv:2508.21038 §5.2).
// Opt-in — a cold-path eval fixture, never linked into production builds.
#[cfg(feature = "limit_fixture")]
pub mod limit_fixture;
pub mod conformal;
// Issue 837 / riir-ai Research 359 — consumption-weight evidence tripwire
// (D-SCAN transliteration): σ-gate metrics + Kendall τ + split-conformal
// benign-quantile threshold. Measured verdict (riir-ai Bench 832): the rank-
// inversion channel discriminates, entropy is composition-coupled telemetry.
#[cfg(feature = "evidence_tripwire")]
pub mod evidence_tripwire;
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::metrics::{
    crps, crps_interval, empirical_coverage, mean_crps_interval, mean_winkler, winkler_score,
};
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::{
    ConformalIntervalCalibrator, DecayUnit, PointForecaster, PredictiveInterval, ResidualMode,
    ResidualRingBuffer, RingBuffer, SeasonalNaiveForecaster, SeasonalPoolForecaster,
    seasonal_naive_floor,
};
// Plan 340 Phase 2 (T2.1) — KARC adapter for the conformal overlay.
// Gated on BOTH features: needs the conformal substrate AND the KARC forecaster.
#[cfg(all(
    feature = "conformal_predictive_intervals",
    feature = "karc_forecaster"
))]
pub use conformal::KarcChannelForecaster;
// Issue 010 T2 — "Report the Floor" comparison harness. Re-exported for
// T3–T7 (BoMSampler, Sleep-Time, Best-Belief, Alien Sampler adapters).
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::{
    FloorAdapter, FloorComparisonReport, OverallVerdict, PredictiveOutput, TrajectoryCorpus,
    UqMetrics, UqPrimitiveUnderTest, empirical_quantile_interval, run_floor_comparison,
};
#[cfg(feature = "coda_fusion")]
pub mod coda;
#[cfg(feature = "dec_operators")]
pub use katgpt_dec as dec;
#[cfg(feature = "dec_operators")]
pub mod dec_freeze;
#[cfg(feature = "dec_operators")]
pub use dec_freeze::CochainFreezeEnvelope;
pub mod delta_mem;
// Higher-order Linear Attention (HLA) substrate — cache types + streaming
// kernels. Spun out to the `katgpt-hla` crate (Issue 007 Phase E Tier 2 #4)
// and re-exported here as `katgpt_core::hla` for backwards compatibility.
// All `crate::hla::*` and `katgpt_core::hla::*` paths resolve unchanged. The
// composition layer (`forward_hla` / `forward_ahla`, depends on ForwardContext)
// stays in katgpt-core; the cognitive role-aware variants stay in riir-engine.
pub use katgpt_hla as hla;
// Shared leaky-integrator primitive. Spun out to the `katgpt-types` leaf
// (Issue 007 Phase E Tier 1 #3) so both katgpt-core (`sense::reconstruction`)
// and `katgpt-micro-belief` (`LeakyIntegrator::step`) can consume it without
// a cycle. Re-exported here as `katgpt_core::leaky_core` for backwards
// compatibility.
pub use katgpt_types::leaky_core;
/// Generic Monte Carlo Tree Search over any [`crate::traits::GameState`].
///
/// Always-on substrate. Composition that needs root-only types
/// (`BanditRolloutPolicy` depends on `crate::pruners::bandit::BanditStats`)
/// stays in the consuming crate.
pub mod mcts;
/// State-Action Pair Cache for MCTS over Deterministic Inference Actions
/// (Plan 390, Research 386, arXiv:2602.04344 UnMaskFork).
///
/// Opt-in extension to [`mcts`]: a standalone search over an opaque
/// `InferenceActionSpace` (no `GameState` / game-IP coupling), backed by a
/// lock-free `StateActionCache` keyed on `(blake3::Hash, InferenceAction)`.
/// Gated behind `mcts_state_action_cache` so the always-on `mcts` substrate
/// stays dep-free.
#[cfg(feature = "mcts_state_action_cache")]
pub mod mcts_state_action_cache;
// Shared freeze/thaw disk I/O for `repr(C)` knowledge structs.
// Extracted from `katgpt-pruners::freeze` (Plan 388 Phase 1) to break the
// katgpt-pruners ↔ katgpt-speculative cycle. Pure stdlib (Path + fs + mem).
// Re-exported by katgpt-pruners::freeze for backwards compatibility.
pub mod freeze;
// Proof goal deduplication cache core types (GoalHash, GoalResult,
// GoalVerifier, ProofGoalCache). Extracted from `katgpt-pruners::proof::goal_cache`
// (Plan 388 Phase 2) to break the katgpt-pruners ↔ katgpt-speculative cycle.
// Pure substrate (blake3 + HashMap + AtomicU64). Re-exported by
// katgpt-pruners::proof::goal_cache for backwards compatibility.
pub mod proof_cache;
// Per-query thinking mode tag. Extracted from `katgpt-pruners` (Plan 388
// Phase 3) to break the katgpt-pruners ↔ katgpt-speculative cycle. Pure
// 4-variant `#[repr(u8)]` enum, no pruners-specific knowledge. Re-exported
// by katgpt-pruners and katgpt_rs::speculative for backwards compatibility.
#[cfg(feature = "parallax_attn")]
pub mod parallax_attn;
pub mod thinking_mode;
// Algebraic-structure primitives. Currently home to the tropical (max, +)
// semiring (Plan 337, Research 321). Opt-in via `tropical_algebra`.
#[cfg(feature = "tropical_algebra")]
pub mod algebra;
pub mod shard_embedding;
// SSMax — length-aware log-N attention temperature (Plan 411, Research 392,
// arxiv 2607.01538 Gollapudi et al. *Drowning in Documents at Million Token
// Scale*). Multiplicative pre-attention logit rescaling that cancels the
// attention dilution at large N. Default `s_L = 1.0` is truly modelless.
// Composes with parallax_attn (sigmoid) and attention.rs (SDPA); does NOT
// apply to funcattn (Research 261 closed negative: basis-mode has no (n,n)
// attention matrix → no dilution). DEFAULT-ON (Plan 411 Phase 5, 2026-07-07):
// G1+G2+G3+G4+G5 ALL PASS.
#[cfg(feature = "ssmax_temperature")]
pub mod ssmax;
// SIMD LUT Dequantization — software analog of StreamDQ near-memory DQ
// (Plan 431, Research 418, arXiv:2607.08993 Jeong et al. SK Hynix 2026). Generic
// format-parameterized dequantize primitive that replaces the per-element
// integer-arithmetic INT→FP cast with a pre-computed f32 LUT lookup. INT4 LUT
// is [f32;16] = one cache line; INT8 is [f32;256] = 1 KB. Phase 1 ships the
// scalar reference; NEON/AVX2 inner loops land in Phase 2; fused dequant+dot
// in Phase 3. DEFAULT-ON (2026-07-18): G1+G2 PASS — see .benchmarks/432_simd_lut_dequant_goat.md.
// Realistic target 1.0-1.5x (the paper's
// 7x is hardware-only). Pure modelless (LUT build + indexed reads).
#[cfg(feature = "simd_lut_dequant")]
pub mod simd_lut_dequant;
// Smooth-min soft pattern matching — modelless latent-space utility for
// fuzzy multi-token retrieval (Research 385, SoftMatcha 2, Issue 041; Issue 041 removed, see git history).
// GOAT PoC PASS: +12pp recall@5 over plain cosine, ~0ns overhead.
// Consumer GOAT PASS (T6 SmoothMinAligned): recall@5 = 1.000 vs Cosine 0.495
// (+50.5pp) on position-aligned multi-token retrieval.
// DEFAULT-ON (2026-07-12): first consumer GOAT gate passes with modelless gain.
#[cfg(feature = "smooth_min_similarity")]
pub mod similarity;
#[cfg(feature = "smooth_min_similarity")]
pub use similarity::{edit_penalty, smooth_min_similarity};
// recos — Rearrangement-Inequality Cosine Similarity (Plan 437, Research 421,
// arXiv:2602.05266). Saturates under ordinal concordance — wider capture range
// than cosine. Sits inside the `similarity` module (which is gated on
// smooth_min_similarity; `recos` implies smooth_min_similarity so the module
// compiles under --no-default-features). Opt-in until the Phase 2 GOAT gate.
#[cfg(feature = "recos")]
pub use similarity::{recos_sim, recos_sim_ranking, recos_sim_slice, recos_sim_slice_into};
// Elasticity-Gated Update — DSOM error-scaled neighborhood update primitive
// (Plan 429, Research 415, Rougier & Boniface 2010 ⟨inria-00495827⟩).
// Time-invariant, error-scaled latent update: step scales with error,
// neighborhood weights are error-gated Gaussian. Pure modelless (exp +
// weighted average). Zero-alloc (stack [f32;32] weights, &mut [f32] output).
// STAYS OPT-IN in katgpt-core (consumer enables transitively) — Consumer
// GOAT PASS (riir-neuron-db, 2026-07-12, Bench 429): G1–G6 ALL PASS; consumer
// feature `elasticity_gated_heal` PROMOTED to default-on in riir-neuron-db
// (behavior opt-in via `.with_neighbor_eta(1.0)`). Primary consumer: neighbor_heal.
#[cfg(feature = "elasticity_gated_update")]
pub mod elasticity_gated_update;
#[cfg(feature = "elasticity_gated_update")]
pub use elasticity_gated_update::{
    ElasticityConfig, compute_error, effective_neighborhood_size, elasticity_gated_update_into,
    neighborhood_weight,
};
// Plan 429 Phase 5 T5.1: Error-weighted graph Laplacian — DSOM neighborhood
// weights composed with the DEC graph Laplacian. Requires both
// `dec_operators` and `elasticity_gated_update` features.
#[cfg(all(feature = "dec_operators", feature = "elasticity_gated_update"))]
pub use elasticity_gated_update::{
    error_weighted_graph_laplacian, error_weighted_graph_laplacian_into,
};
// Position-Offset Reveal-Time Schedule for Set Diffusion (Research 376).
// Canonical source for `PositionOffsetSchedule` — pure math (CDF/inverse-CDF/
// ordering), RNG-agnostic via closure-based sampling. No feature gate because
// it's a zero-dep math substrate consumed by both katgpt-rs (runtime) and
// riir-train (training). Eliminates the 3-way DRY violation that previously
// had copies in katgpt-rs/src/dllm.rs, riir-train/.../set_diffusion_schedule.rs,
// and riir-ai/crates/riir-poc/.
pub mod set_diffusion_schedule;
pub use set_diffusion_schedule::{
    PositionOffsetSchedule, ar_order, block_causal_gen_steps, mdlm_gen_steps, order_to_gen_steps,
    uniform_order, uniform_order_with,
};
// UGC — Unmasking Growth Complexity certified schedules for masked diffusion
// (arXiv:2608.13520, Research 485 / Issue 664). Always-on: pure math + a
// denoiser trait, zero deps beyond crate::types::Rng — the same no-gate
// rationale as set_diffusion_schedule. Feature-flag plan (`ugc_schedule`)
// opens ONLY if the Issue 664 G1b promotion gate passes.
pub mod ugc_schedule;
pub use ugc_schedule::{
    UgcBlockPlan, UgcDenoiser, UgcIntervalEstimate, UgcProfile, UgcScratch, UGC_MASK,
    bernoulli_unmask_with_grid, certified_block_plan, certified_iteration_count,
    dp_partition, equal_sqrt_mass_grid, estimate_interval, estimate_profile,
    inv_log_reveal_odds, log_reveal_odds, reveal_grid_from_plan, reveal_odds,
};
// SwitchCostTable — directed pairwise switch-difficulty table (skill-entropy
// distillation, Research 484 / arXiv:2608.05139, Issue 663). Opt-in per the
// issue's GOAT-gate discipline: promotion to default requires a riir-ai
// consumer A/B (F1: SkE-gated preemptive re-estimation vs the coherence-only
// arm on the Issue-054 stuck-rate scenario).
#[cfg(feature = "switch_cost")]
pub mod switch_cost;
#[cfg(feature = "switch_cost")]
pub use switch_cost::{
    FactorizedSwitchCost, SwitchCostSnapshot, SwitchCostTable, DEFAULT_ALPHA, NEUTRAL_ACC,
    cdf_rank,
};
// Extension-count (freedom-of-function) selection criterion — closed-form
// near-best selection over a declared finite output partition (Research 486,
// arXiv:2608.05423, Issue 665). Opt-in PoC-gated: promotion to default
// requires the Issue 665 PoC gate (freedom-guided near-best beats min-loss
// AND random-near-best under a declared distribution shift) plus a
// production consumer.
#[cfg(feature = "freedom_selection")]
pub mod extension_count;
#[cfg(feature = "freedom_selection")]
pub use extension_count::{
    ExtensionOccupancy, LossGate, FIRST_ACTIVATION_GAIN, freedom_gain, log_freedom,
};
// Effective Degree — function-space simplicity via polynomial representations
// along data-anchored interpolation paths (Research 488 / arXiv:2605.29823,
// Issue 668). Modelless measurement only; the paper's differentiable
// regularizer is training-side (riir-train). Reuses `karc::ChebyshevBasis` +
// `linalg`'s damped Cholesky, so the feature implies `karc_forecaster`.
// Opt-in: promotion to default is blocked on a consumer verdict
// (riir-neuron-db Issue 602 freeze-gate PoC — does ED beat `output_flatness`?).
#[cfg(feature = "effective_degree")]
pub mod effective_degree;
#[cfg(feature = "effective_degree")]
pub use effective_degree::{
    EdConfig, EdError, EdResult, EdScratch, MAX_ED_DEGREE, MAX_ED_TERMS, ed_from_coeff_norms,
    ed_over_pairs, effective_degree_along_path, effective_degree_along_path_multi,
    randomized_cosine_nodes,
};
// SIMD-accelerated linear algebra kernels (NEON / AVX2 / WASM-SIMD128 /
// scalar fallback). Spun out to the `katgpt-types` crate (Issue 007 Phase E
// Tier 1 #2) and re-exported here as `katgpt_core::simd` for backwards
// compatibility. All `crate::simd::*` paths resolve unchanged.
pub use katgpt_types::simd;
pub mod speculative;
pub mod traits;

// Prompt-backend trait — generic prompt→string inference contract (Issue 580).
// Hoisted from riir-game-sdk::gm::prompt so multiple consumers (riir-agents
// Phase 2, the SDK's gm::prompt module, future callers) share one trait.
// Always-on: pure trait + zero-dep mock, no feature gate needed.
pub mod prompt_backend;
pub use prompt_backend::InferenceBackend;
// Shared configuration, RNG, math utilities, LoRA, domain embeddings, and
// inference types. Spun out to the `katgpt-types` crate (Issue 007 Phase E
// Tier 1 #2) and re-exported here as `katgpt_core::types` for backwards
// compatibility. All `crate::types::*` paths resolve unchanged.
pub use katgpt_types as types;

// CGSP — Curiosity-Guided Self-Play modelless triad (Plan 274, Research 240).
// Self-contained: Direction/Target/Candidate, CgspLoop, PoolConjecturer,
// BeliefGridProjectionGuide, BreakevenDifficultyFilter, ColinearityBatchGate,
// EntropyCollapse, CuriosityPrioritySnapshot (BLAKE3-committed).
// Consumed by riir-engine Plan 299 (NPC curiosity runtime).
#[cfg(feature = "cgsp")]
pub mod cgsp;
#[cfg(feature = "cgsp")]
pub use cgsp::{
    BatchQualityGate,
    BeliefGridProjectionGuide,
    BreakevenDifficultyFilter,
    Candidate,
    CgspConfig,
    CgspLoop,
    ColinearityBatchGate,
    CollapseSignal,
    ComplexityWeights,
    CuriosityConjecturer,
    CuriosityPrioritySnapshot,
    CycleResult,
    CycleStats,
    DEFAULT_BELIEF_DIRECTION_DIM,
    DEFAULT_K,
    DEFAULT_POOL_SIZE,
    DifficultyFilter,
    Direction,
    EntropyCollapse,
    HintDeltaBandit,
    NoOpBatchGate,
    NoOpDifficultyFilter,
    PoolConjecturer,
    Priority,
    QualityGuide,
    ScratchBuffers,
    SolveRate,
    Solver,
    Target,
    entropy_nats,
    structural_complexity,
    // Note: `sigmoid` is no longer re-exported here — it's now an always-on
    // top-level `katgpt_core::sigmoid` (Proposal 003 Phase 0.1). The module-local
    // `katgpt_core::cgsp::sigmoid` (in cgsp/types.rs) remains for `cgsp::*` paths.
};

// CGSP dual-pool extension — DecentMem distillation (Plan 282, Research 249).
#[cfg(feature = "cgsp_dual_pool")]
pub use cgsp::{DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter};

// Issue 364 T4 — modelless k_npc selector (wraps GainCostLoopHalter, Plan 304).
// Needs both cgsp (the host module) and gain_cost_halt (the halter kernel).
// Consumed by riir-ai's per-NPC CLR cadence wiring (Phase 30 of tick_map).
#[cfg(all(feature = "cgsp", feature = "gain_cost_halt"))]
pub use cgsp::{KnpcDecision, KnpcSelector};

// ActionBridge — generic latent→raw action bridge (Plan 262).
#[cfg(feature = "action_bridge")]
pub mod bridge;
#[cfg(feature = "action_bridge")]
pub use bridge::ActionBridge;

// Re-export consolidated traits (Plan 107 Phase 0)
pub use traits::{
    ActionSpaceLog, BestBuddyAligner, BinaryScreeningPruner, ConstraintPruner, DominoPruner,
    FeatureClass, GameState, NoPruner, NoScreeningPruner, RandomRolloutPolicy, RolloutPolicy,
    ScreeningPruner, StateHeuristic, best_buddies, pearson_correlation,
};
pub use traits::{GenerativeConstraintPruner, SpeculativeGenerator};

// RecursionLogits — opt-in trait for generators that expose pre/post recursion
// logits so AdvantageMarginGate can wrap them (Plan 283 T2.3, arxiv:2511.16886).
// Opt-in: not in default feature list. Non-recursing generators do not implement it.
#[cfg(feature = "recursion_logits")]
pub use traits::RecursionLogits;

// Q-Guided Flow (Plan 268) — test-time Q-gradient guidance primitive.
#[cfg(feature = "qgf_oracle")]
pub use traits::{NoGuidanceOracle, QGradientOracle};
#[cfg(feature = "qgf")]
pub mod qgf;

// MicroRecurrentBeliefState — per-entity recurrent state kernel (Plan 276, Research 242).
// Trait + Family A (attractor) + Family C (leaky) + BLAKE3 snapshot + sigmoid bridge.
// Spun out to the `katgpt-micro-belief` crate (Issue 007 Phase E Tier 1 #3) and
// re-exported here as `katgpt_core::micro_belief` for backwards compatibility.
#[cfg(feature = "micro_belief")]
pub use katgpt_micro_belief as micro_belief;
#[cfg(feature = "micro_belief")]
pub use micro_belief::{
    AttractorKernel, KernelConfig, LeakyIntegrator, MicroRecurrentBeliefState,
    MicroRecurrentKernelSnapshot, RecurrenceFamily, SNAPSHOT_VERSION, project_to_scalars,
};

// BoMSampler — K-hypothesis single-pass belief sampling (Plan 281, Research 248).
// Opt-in extension of MicroRecurrentBeliefState; gated on bom_sampling which implies micro_belief.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{BoMSampler, NoiseQueryConfig, QmcMethod, SeedStrategy, dot_product_scorer};

// Plan 370 — QMC noise-fill convenience entry point (constructs the right
// QmcSource from a QmcMethod tag + seed, zero-alloc). Used by
// MultiHypothesisBoMMinimaxPlanner when NoiseQueryConfig::qmc_method is Some.
#[cfg(all(feature = "qmc_sampling", feature = "bom_sampling"))]
pub use speculative::fill_noise_queries_gaussian_qmc_by_method;

// BoM G2 arena harness — Plan 281 T2.3.
// Engine-side traits + synthetic reference env. riir-ai implements the traits
// over a real bomber/go sim to produce the empirical G2 gate.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{
    ArenaAction, ArenaEnvironment, BeliefPlanner, BoMMeanPlanner, BoMMinimaxPlanner,
    ComparisonResult, DeterministicPlanner, EnvHint, PlannerOutcome, SyntheticThreatArena,
    bom_mean_attractor, bom_minimax_attractor, bom_minimax_leaky, run_arena_comparison,
};

// FaithfulnessProbe — causal intervention diagnostic for injected memory (Plan 278, Research 244).
// Moved from katgpt root to katgpt-core so riir-engine (Plan 308) can consume via katgpt-core.
// Two features:
// - `triggered_injection` (default-ON after GOAT G3): sigmoid-thresholded inject/skip hot-path gate.
// - `faithfulness_probe` (opt-in, audit cadence): full intervention suite + perturbation + attribution.
// The module is compiled when EITHER feature is on; submodules are individually gated in `mod.rs`.
#[cfg(any(feature = "faithfulness_probe", feature = "triggered_injection"))]
pub mod faithfulness;

// Pruners module (Plan 054 review_metrics + Plan 320 indicator_probe_bank, etc.).
// Parent module is always compiled; individual submodules gate their own features.
// (Previously the whole `pruners` module was gated behind `review_metrics`; that
// coupling was broken in Plan 320 so indicator_probe_bank can gate independently.)
pub mod pruners;

// Temporal Derivative Kernel — dual fast/slow EMA surprise signal (Plan 277, Research 243).
// Turns any streaming latent vector into a signed "surprise" signal — the implicit
// prediction-error channel for credit assignment, computed locally with no backprop.
// DEFAULT-ON (Plan 277, 2026-06-16): 4/4 fusion gates PASS — see .benchmarks/277_temporal_deriv_goat.md.
#[cfg(feature = "temporal_deriv")]
pub mod temporal_deriv;
#[cfg(feature = "temporal_deriv")]
pub use temporal_deriv::{TemporalDerivativeKernel, sigmoid_surprise_gate};

// HOLA Hippocampal Exact KV Cache — surprise-evicted (β·‖e‖) bounded KV cache with
// decoupled RMSNorm-γ read (Plan 395, Research 378, arxiv 2607.02303). Complements
// the GDN2 fixed-size recurrent state with a top-w exact KV set for long-range
// retrieval. STAYS OPT-IN — G1–G4 GOAT PASS + consumer wiring PASS (modelless
// gain, Issue 038 production wiring in forward_gdn2 via HippocampalCacheDyn);
// promotion deferred to G5 riir-train gate (perplexity on real text). Pure
// stdlib + katgpt-types.
#[cfg(feature = "hippocampal_cache")]
pub mod hippocampal_cache;
#[cfg(feature = "hippocampal_cache")]
pub use hippocampal_cache::{HippocampalCache, SortedSlotCache};

// HOLA dynamic (runtime D/W) variant — the production consumer for forward_gdn2
// which uses runtime config.head_dim. Same algorithm as HippocampalCache<D,W>
// but Vec-based for runtime dimensions. Alloc-free read path (pre-allocated
// scratch). Plan 395 Phase 5 (Issue 038 production wiring).
#[cfg(feature = "hippocampal_cache")]
pub mod hippocampal_cache_dyn;
#[cfg(feature = "hippocampal_cache")]
pub use hippocampal_cache_dyn::HippocampalCacheDyn;

// Tiered Hot/Warm/Cold K/V Store — the route-and-fetch substrate for sparse
// long-context attention (Plan 397, Research 379, arxiv 2606.30709). Generic
// trait + in-memory reference impl. Always-on (no feature gate) because it's
// a generic primitive with no attention-layer deps; the HGA-specific consumer
// is gated by `hga`.
pub mod tiered_kv;

// Hierarchical Global Attention (HGA) — chunk→group→token routing with
// RoPE-aware mixed-frequency summaries (Plan 397, Research 379, arxiv 2606.30709).
// Three refinements of the sparse-attention routing slot: group middle tier,
// mixed-RoPE summarizer, tiered route-and-fetch consumer. Opt-in until the
// Phase 2 GOAT gate (G2 head-to-head vs DashAttention) passes.
//
// NOTE: the HGA forward path (which needs dash_attn::entmax_1p5) lives in
// katgpt-attn/src/hga_forward.rs, not here — katgpt-core cannot import
// katgpt-attn without a circular dependency.
#[cfg(feature = "hga")]
pub mod hga;
#[cfg(feature = "hga")]
pub use hga::{GroupSummaryCache, MixedRopeSummarizer};

// Renoise-CE Self-Verifier — perturb a completed state, re-resolve through the
// same operator, measure drift as a verifier-free correctness score (Plan 406,
// Research 369, arxiv 2606.29150). Third orthogonal self-eval signal alongside
// CLR (claim-vote) and CoE (trajectory-shape). Operator-agnostic trait over any
// state->state map. DEFAULT-ON (Phase 10, 2026-07-04). NOT a UQ primitive
// (raw ranking signal; conformal wrapping required for any UQ claim).
#[cfg(feature = "renoise_ce")]
pub mod renoise_ce;
#[cfg(feature = "renoise_ce")]
pub use renoise_ce::{
    Proposer, RenoiseCeConfig, RenoiseCeProbe, RenoiseCeScore, best_of_n_stability,
    renoise_ce_score, verify_and_restart,
};
// Freedom-guided sibling of best_of_n_stability (Issue 665 / Research 486):
// near-best drift gate + Δ-log-extension-count selection over a caller-owned
// occupancy table. Gated on freedom_selection (implies renoise_ce).
#[cfg(feature = "freedom_selection")]
pub use renoise_ce::best_of_n_freedom;

#[cfg(feature = "dual_leo")]
pub use traits::{
    ActingMode, AlphaSchedule, AutocurriculumSampler, BcConfig, BcTarget, DualLeoMixer,
};
#[cfg(feature = "leo_all_goals")]
pub use traits::{AllGoalsUpdate, LeoHead, sigmoid_bounded_q};

// Re-export key types at crate root for convenience
#[allow(deprecated)]
pub use shard_embedding::{EMBED_DIM, JlProjectionMatrix, STYLE_DIM as JL_STYLE_DIM};
#[cfg(feature = "loop_stability_fix")]
pub use types::LoopStabilityMode;
#[allow(deprecated)]
pub use types::ShardEmbedding;
#[allow(deprecated)]
pub use types::sample_token;
pub use types::{
    AttentionMode, AttentionProjection, CacheLayout, CalibrationMode, Config, ConvergenceSelector,
    CopyLateShape, DashAttnConfig, DilationConfig, HlaMode, HybridPattern, InferenceOverrides,
    InferenceResult, LoopMode, LoraAdapter, LoraPair, ModelArchitecture, ResidualGate,
    RetrievalHeadRole, Rng, RtTurboConfig, SdpaOutputGate, WeightDtype, kv_dim, lora_apply,
    matmul, matmul_f16, matmul_f16_parallel, matmul_parallel, matmul_relu, rmsnorm,
    sample_token_into, softmax, softmax_scaled,
};

#[cfg(feature = "domain_latent")]
pub use types::DomainLatent;

#[cfg(feature = "sr2am_configurator")]
pub use types::{ConfiguratorContext, PlanningDecision};

#[cfg(feature = "data_gate")]
pub use types::{DataGate, GateDecision, ProposerTask, TaskType};

#[cfg(feature = "sparse_mlp")]
pub use types::sparse_matmul;

#[cfg(feature = "coda_fusion")]
pub use coda::{
    GateActivation, MoaConfig, compute_rstd, simd_matmul_residual,
    simd_matmul_residual_partial_rms, simd_matmul_rmsnorm_activation, simd_matmul_rmsnorm_rope,
    simd_matmul_rmsnorm_swiglu, simd_matmul_rmsnorm_swiglu_split,
};

#[cfg(all(feature = "coda_fusion", feature = "moa_inference"))]
pub use coda::{MoaActivation, moa_swiglu, simd_matmul_rmsnorm_moa_swiglu};

#[cfg(feature = "tiled_attention")]
pub use attention::{
    tiled_attention_batched, tiled_attention_forward, tiled_attention_forward_with_scores,
};

#[cfg(feature = "parallax_attn")]
pub use parallax_attn::{
    ParallaxActivation, ParallaxConfig, ParallaxScratch, compute_rho, parallax_correction,
    tiled_attention_parallax_forward, tiled_attention_parallax_forward_retaining,
};

// Sink-aware composition (Plan 289). Requires both parallax_attn (for the
// forward) and sink_aware_attn (for the classifier + flat gate). The
// `tiled_attention_parallax_forward_sink_aware` entry point short-circuits to
// vanilla parallax when policy = Uniform, so this is a zero-cost abstraction
// for callers who construct the scratch but never enable DualPolicy.
#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
pub use parallax_attn::{SinkAwareParallaxScratch, tiled_attention_parallax_forward_sink_aware};

pub use simd::SimdLevel;

#[cfg(feature = "hydra_budget")]
pub use types::{HydraBudgetConfig, HydraLayerProfile};

#[cfg(feature = "collapse_aware_thinking")]
pub use types::ThinkingBudget;

#[cfg(feature = "questbench")]
pub mod questbench;
#[cfg(feature = "questbench")]
pub use questbench::{
    CspDomain, MemoryTier, QuestBenchDecision, SyntheticCsp, UnderspecConfig, find_sufficient_set,
    generate_synthetic_csps, tier_from_score, underspecification_score,
};

#[cfg(feature = "tf_loop")]
pub use types::{CacheStrategy, IterationMode, SubStepStrategy, TrainingFreeLoopConfig};

#[cfg(feature = "plasma_path")]
pub use simd::{simd_ternary_matmul_batch, simd_ternary_matvec, ternary_matvec_scalar};
#[cfg(feature = "plasma_path")]
pub use types::TernaryWeights;

#[cfg(feature = "binary_plasma")]
pub use simd::{binary_matvec_scalar, simd_binary_matmul_batch, simd_binary_matvec};
#[cfg(feature = "binary_plasma")]
pub use types::{BinaryWeights, GROUP_SIZE as BINARY_GROUP_SIZE};

// Issue 578: the Q2_0_g128 container — ternary {-1,0,+1} bit-planes with the
// per-128 f16 group scale. Neither shipped tier could hold it (plasma_path has
// the zero state but per-row scale; binary_plasma has the group scale but no
// zero state).
#[cfg(feature = "ternary_group_scale")]
pub use simd::{
    simd_ternary_group_matmul_batch, simd_ternary_group_matvec, simd_ternary_group_matvec_parallel,
    ternary_group_matvec_scalar,
};
#[cfg(feature = "ternary_group_scale")]
pub use types::{
    TernaryFfnHook, TernaryGroupWeights, TernaryInputProjHook, TernaryMatvecHook,
};

// Issue 582: the base-3 footprint tier — same alphabet and group scale as the
// Q2_0_g128 container, 5 trits per byte instead of two bit-planes (1.75 vs
// 2.125 bits/weight, -17.6%).
#[cfg(feature = "ternary_trit_pack")]
pub use simd::{
    simd_ternary_trit_matvec, simd_ternary_trit_matvec_parallel, ternary_trit_matvec_scalar,
};
#[cfg(feature = "ternary_trit_pack")]
pub use types::{TRITS_PER_BYTE, TernaryTritWeights};

#[cfg(feature = "peira_distill")]
pub mod peira;
#[cfg(feature = "peira_distill")]
pub use peira::{PeiraConfig, PeiraCovariance, peira_aux_loss};

#[cfg(feature = "dirichlet_energy")]
pub mod dirichlet;
#[cfg(feature = "dirichlet_energy")]
pub use dirichlet::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};

#[cfg(feature = "spectral_hierarchy")]
pub mod spectral_hierarchy;
#[cfg(feature = "spectral_hierarchy")]
pub use spectral_hierarchy::{cauchy_interlacing_check, eigenspace_alignment, haar_wavelet_basis};

#[cfg(feature = "sigmoid_margin")]
pub use simd::{
    ArgmaxAudit, argmaxable_witness, audit_argmaxable, compute_retrieval_margin,
    dim_capacity_ceiling, dim_capacity_floor, dim_capacity_required, dim_sufficiency_bound,
    ln_binomial, matrix_rank, sigmoid_margin_loss,
};

#[cfg(feature = "dual_gram_pca")]
pub use simd::simd_gram_f32;

#[cfg(feature = "roofline_cost")]
pub mod roofline;
#[cfg(feature = "roofline_cost")]
pub use roofline::{
    ComputeBound, Dtype, HardwarePeaks, OpType, RooflineCost, gemm_cost, gemv_cost, gram_cost,
    roofline_estimate,
};

#[cfg(feature = "ane_roofline")]
pub mod ane_roofline;
#[cfg(feature = "ane_roofline")]
pub use ane_roofline::{
    AneBound, AneCost, AneFamily, AneOpShape, AnePeaks, Device, ane_conv3x3_cost, ane_estimate,
    ane_gemm_cost, ane_gemv_cost,
};

#[cfg(feature = "and_or_dtree")]
pub mod and_or;
#[cfg(feature = "and_or_dtree")]
pub use and_or::AndOrNode;

#[cfg(feature = "partial_scoring")]
pub use traits::{GameTrace, PartialScorer};

#[cfg(feature = "problem_mutator")]
pub use traits::{GameConfig, MutantConfig, MutationKind, ProblemMutator};

#[cfg(feature = "modal_spec")]
pub mod linoss;
#[cfg(feature = "mux_pruner")]
pub mod mux;

// Sense substrate was spun out to the `katgpt-sense` crate (Issue 007 Phase E
// Tier 2 #7, Plan 338). `spectral_threat` stayed local (depends on `linoss`);
// it lives at `crate::sense_threat` and is re-exported through the
// `sense::spectral_threat` shim path below to preserve external consumers'
// `katgpt_core::sense::spectral_threat::*` paths bit-for-bit.
#[cfg(feature = "sense_composition")]
pub mod sense {
    pub use katgpt_sense::*;
    #[cfg(feature = "spectral_threat")]
    pub mod spectral_threat {
        pub use crate::sense_threat::*;
    }
}
#[cfg(feature = "spectral_threat")]
pub mod sense_threat;

#[cfg(feature = "slod")]
pub mod slod;
#[cfg(feature = "slod")]
pub use slod::{
    ScaleBoundary, SlodConfig, SlodOperator, SlodPruner, exp_map, frechet_mean,
    heat_kernel_weights, log_map, poincare_distance,
};

// Spectral Irrep Pruner - spectral flatness-based speculative decoding pruning (Plan 246).
// Prunes tokens when logit spectrum shows competing modes (high spectral flatness).
// GOAT PASS: +3.6% overhead, default-ON.
#[cfg(feature = "spectral_pruner")]
pub mod irrep_pruner;

// Subspace phase-gate primitives — participation ratio, numerical rank, N≥d
// phase-transition gate (Wang et al. Thm 4, arXiv:2409.02426), and runtime
// Jacobian SVD via forward differences (Plan 301, Research 279). Pure numeric,
// no game/shard/chain semantics. Consumers (riir-neuron-db Plan 002, future
// riir-ai HLA self-discovery plan) apply these to their own maps.
// DEFAULT-ON (Plan 301 Phase 5 T5.1, 2026-07-02): G1 PASS + G3-precursor PASS
// + T3.4 latency PASS + G4 PASS. Transitively enabled via viable_manifold_graph
// + tucker_factorization (both default-on).
#[cfg(feature = "subspace_phase_gate")]
pub mod subspace_phase_gate;

// Group Invariance Probe — modelless symmetry discovery on a hypothesis Lie
// group (Plan 356, Research 355 — distilled from LieFlow, arXiv:2512.20043).
// Generalizes `subspace_phase_gate` from "subspace of R^d" to "subgroup of G":
// score each sampled g ∈ G by direct invariance testing σ(β·(1−d(q, g·q))),
// then classify the discovered H as Discrete / Continuous / Partial / None via
// a participation-ratio-style concentration measure on the score histogram.
// Pure numeric, no game/shard/chain semantics, zero deps. Sibling of
// `subspace_phase_gate`. STAYS OPT-IN — 8/8 GOAT gates PASS (Bench 356,
// 2026-07-01); not promoted to default because no shipped consumer exists yet
// (Issue 011 Q2+Q3 verdict + riir-ai fusion plan still pending).
#[cfg(feature = "group_invariance_probe")]
pub mod group_invariance_probe;

// Latent Trajectory Geometry — probe-free geometric diagnostic (length +
// mean turning-angle curvature + min adjacent cosine + bifurcation ratio).
// Distilled from Pandey et al., arXiv:2606.09287 (Plan 342, Research 324).
// Pure numeric over `&[&[f32]]`, no extra deps. Opt-in until the Phase 3
// game-related gate (curvature catches the oscillation failure mode entropy
// misses) passes; promotion to a routing role is a separate follow-up plan.
#[cfg(feature = "latent_trajectory_geometry")]
pub mod latent_trajectory_geometry;

// SWE Trajectory Freezer — modelless committed freeze of an inference
// attempt's trajectory geometry (Proposal 011 Phase 5, Task T5.5). Composes
// latent_trajectory_geometry + committed_field_blend + a local BLAKE3
// envelope. Opt-in — research-validation primitive; promotion requires the
// T5.6 G5 gate (cross-snapshot discrimination) to pass on real-model
// trajectories (currently PARTIAL — T5.4 G3 FAIL at 29% on Kimi-K3 depth
// trajectories; see .benchmarks/012_kimi_k3_trajectory_geometry.md).
#[cfg(feature = "swe_trajectory_freeze")]
pub mod swe_trajectory_freeze;

// Latent Confounder Audit — three modelless forward-pass diagnostics
// (R₀ zero-transition response + R_shift shift-invariance response + L
// shortcut leakage) auditing a conditioning latent for action-irrelevant
// confounders. Distilled from CD-LAM §III-B + Appendix A (Wei et al.,
// arXiv:2607.09185; Research 460, Issue 194). The diagnostic half of CD-LAM
// (the L_emb/L_ctr/L_cal training recipe → riir-train). Pure numeric over a
// caller-supplied encoder closure; zero deps. Opt-in — diagnostic primitive,
// not a runtime capability. Stays opt-in until a consumer (MAG/TILR/Steering)
// benchmarks a quality gain from running the audit before deployment.
#[cfg(feature = "latent_confounder_audit")]
pub mod latent_confounder_audit;

// Interpolation Geometry — iMAUVE + 5-way intervention probe for committed
// latent substrates (Issue 158, Research 445 — Prabhudesai & Geng, *Latent
// Thought Flows with Text Compression*, Jun 2026). Generic `LatentSpace`
// trait abstracting over HLA [f32;8] / style_weights[64] / archetype-blend
// π / KarcShard / ZoneGeometryPod / MerkleFrozenEnvelope — the six substrates
// cataloged in Research 445 §2.6. Two protocols: `imauve_score` (nearest-
// neighbor midpoint coherence — the paper's headline metric, Pearson r=0.99
// with downstream quality) + `intervention_battery` (matched/shuffled/zero/
// mean/noise 5-way probe extending Plan 278's FaithfulnessProbe to per-entity
// committed state). Pure modelless evaluation methodology — NOT a training
// primitive. Opt-in until the PoC reports interpolation geometry across the
// substrates (Phase 4 decision branch in `.issues/158_*`).
#[cfg(feature = "interpolation_geometry")]
pub mod interpolation_geometry;

// Viable Manifold Graph — discrete safe-manifold navigation primitive.
// Distillation of arXiv:2206.00106 (González-Duque et al., *Mario Plays on a
// Manifold*, 2022). Generic over any smooth map `f: R^n → R^m` (closure) and
// a viability predicate `V(z)`. Computes the pullback volume field
// `log det(J_f^T J_f)` (via Plan 301's `jacobian_svd_at`), filters a latent
// sample to a discrete safe-manifold subgraph, and runs A* / random-walk
// navigation that stays inside the viable set by construction. Game / shard /
// chain wiring lives in riir-ai (R154). DEFAULT-ON (Plan 312 Phase 5,
// 2026-06-24): G1-G7 PASS + perf bench PASS (CSR adjacency, manifold_random_walk
// 7.10 ns/step).
#[cfg(feature = "viable_manifold_graph")]
pub mod viable_manifold_graph;

// Certified Frontier — modelless safe-set expansion (Plan 580, Research 510,
// arXiv:2606.08802 De Santi et al.; SAFEOPT lineage). The acquisition half of
// the grow-then-navigate stack whose navigation half is `viable_manifold_graph`
// above: grow a monotone provably-valid cell set from binary verifier outcomes,
// then answer where to look next and when to stop. Phase 0 PASS (Bench 687:
// 0 violations, monotone, 51.4x frontier-vs-passive). Opt-in.
#[cfg(feature = "certified_frontier")]
pub mod certified_frontier;

// Usage-Rate (Mass/Age) KV Eviction Scoring + Generation-Runaway Canary
// (Plan 585, Research 523, arXiv:2608.19920 "Learning how to Forget" Seeger
// et al., AWS 2026). The paper's normalized H2O score `cum_mass / max(1,
// age)` — O(1)/row/step over caller-supplied attention-mass increments (the
// `suspect_indices` house pattern; mass producers are consumer-side, riir-ai
// Issue 836 pull-gated on this plan's GOAT). Plus `RunawayStats` /
// `runaway_gate`: the R/p128 generation-runaway canary, the promotion gate
// the Issue 750 lossy-surface rule lacks on the generation axis. Opt-in
// (Plan 585; GOAT .benchmarks/697).
#[cfg(feature = "usage_rate_eviction")]
pub mod kv_eviction;


// Canvas Schema Compiler — declared causal topology for attention masks
// (Plan 419, Research 398, Valdez *Canvas Engineering* July 2026). The
// modelless half: a typed CanvasSchema compiler that lowers a declared region
// layout + directed topology into an AttentionMaskSpec (consumable by
// AC-Prefix / VortexFlow), a LossWeightMask, and a can_reach / reachability
// primitive proving exact marginal independence for binary masks (absent
// edge ⟹ no influence, by construction). Plus a transfer_distance semantic-
// type compatibility scalar. Pure structure compilation, zero gradient
// descent. STAYS OPT-IN — G1–G6 GOAT gates ALL PASS (Bench 419, 2026-07-09);
// not promoted to default because the fusion PoC resolved inconclusively
// (Issue 043, removed) and the primitive's constituents are already default-on
// with runtime consumers.
#[cfg(feature = "canvas_schema")]
pub mod canvas;

// Zone Affective Manifold — crowd-scale PCA via power iteration + deflation
// on the (N, D) crowd-activation covariance (Issue 001). Top-k principal
// directions ("zone mood axes") + per-NPC projections. Rayon-parallel for
// N > parallel_threshold, cold-start identity fallback for small crowds,
// sign-fixed for temporal continuity. Pure modelless. Opt-in until G1-G6 pass.
#[cfg(feature = "zone_affective_manifold")]
pub mod zone_manifold;

// Zone Density Routing — modelless per-zone physical compute scheduler (Plan
// 351, Research 350 — Treuille Continuum Crowds + Fokker-Planck-on-cochains).
// Three primitives: zone_density_classify (mobility = fast_sigmoid(-β·(ρ−ρ₀))
// → tier + composite cache_key), schedule_outer_first (stable ascending-density
// sort — outer/sparse zones compute first), ZoneDensityCache<V> (papaya-backed
// LRU with tier-transition / density-drift / TTL invalidation rules). Sibling to
// Plan 305 cognitive gating (Plan 305 gates learning compute; this gates
// movement compute) — they compose orthogonally, NOT overlap. Population is
// raw/synced; mobility/tier/cache_key are latent/local. DEFAULT-ON (Plan 351 Phase 3, 2026-06-29): GOAT PASS —
// G5a (Shannon entropy ≥+15% vs mean-agg) + G5b (≥50% compute saved on dense-dominated)
// + G5c (zero stale reads during stampede) all pass. No UQ claim — mobility is
// a deterministic [0,1] weight, not a probability/interval/coverage.
#[cfg(feature = "zone_density_routing")]
pub mod zone_density;

// AC-GPT Arbitrary-Conditional Prefix — modelless mask builder + sequence
// augmenter that turns any causal Transformer forward into a single-pass
// arbitrary-conditional forward p(xe | xc) via position-aware copies of xc at
// the front and a [xc-bidirectional | causal-everywhere-else] attention mask
// (Lu et al., Mila, arXiv:2606.14943, Plan 313, Research 295). Phase 1 ships
// types + bit math only — no attention kernel dep, no SVD. DEFAULT-ON (Plan 313, 2026-06-24): G1–G4 PASS via modelless Path 2 (`attends_dedup`) — see .benchmarks/313_ac_prefix_modelless.md. Multi-layer equivalence remains a non-blocking riir-train follow-up.
#[cfg(feature = "ac_prefix")]
pub mod ac_prefix;

// Causal Head-Importance Calibration & Scale-Normalized Heterogeneous Fusion
// (Plan 358, Research 362, arXiv:2606.20097 HydraHead). Modelless
// causal-intervention head scorer: activation patching (Eq 10) + path patching
// (Eq 11) + span-level logit-diff readout (Eq 9) + cross-capability fusion
// (Eq 12) + head partition mirroring RTPurbo's HeadCalibration. Plus
// scale-normalized heterogeneous-branch fusion (Eq 13–14, currently unused).
// Pure numeric over `&[f32]` + a caller-supplied patched-forward-pass closure;
// the patched forward pass itself lives in riir-engine. Sibling of
// `faithfulness_probe` (causal-intervention diagnostic pattern). Opt-in until
// G1–G4 GOAT gate passes; competes for the RTPurbo calibration slot.
#[cfg(feature = "causal_head_importance")]
pub mod causal_head_importance;

// Causal-ID — Algorithmic Syntactic Causal Identification (Plan 457,
// Research 450, arXiv:2403.09580 Cakiqi & Little 2024). Pure modelless
// graph rewriting on Acyclic Directed Mixed Graphs (ADMGs) with bidirected
// confounders: `identify(Y, do(A))` returns the interventional signature
// backbone `Y⋆ = An(Y in G[V\A])` via the recursive Shpitser-Pearl ID
// algorithm (Cakiqi-Little Theorem 1 distillation). The Issue 545
// defend-wrong PoC proved S2 strictly dominates Canvas FlowGraph
// reachability on a 13-node game KG with a `NPC1 ↔ NPC2` confounder: S2
// produces a 5-node interventional signature that correctly excludes NPC1
// (the confounder neighbor); S1 yields only a boolean reaches=true and
// cannot see the confounder. Sibling of `causal_head_importance` (activation
// patching) + `canvas` (declared directed-only topology) — causal_id adds
// the bidirected-confounder dimension they cannot see. Pure modelless
// (graph rewriting on BLAKE3-committed NodeId set). Offline-only (24µs on
// 13 nodes is well outside the 20Hz tick; subgraph extraction mandatory to
// stay ≤32 nodes per query). DEFAULT-ON since 2026-07-18 (Plan 457 Phase 5
// promotion): Plan 457 Phase 2 GOAT G1+G2+G3+G4 ALL PASS; §T4.7 promotion
// gate PASS on Consumer A (synthetic 100-node game-world KG, OR criterion —
// Consumer B T4.6 sleep-cycle remains BLOCKED on real-trace capture but does
// not block promotion).
#[cfg(feature = "causal_identification")]
pub mod causal_id;
#[cfg(feature = "spectral_pruner")]
pub use irrep_pruner::{
    IrrepPruner, IrrepPrunerConfig, irrep_pruner_from_config, spectral_flatness,
};
#[cfg(feature = "subspace_phase_gate")]
pub use subspace_phase_gate::{
    IntrinsicDimMethod, JacobianSvdScratch, SvdResult, SvdResultScratch, SvdScratch,
    estimate_intrinsic_dim, jacobian_svd_at, jacobian_svd_at_into, numerical_rank,
    participation_ratio, phase_transition_gate, thin_svd, thin_svd_into,
};

#[cfg(feature = "group_invariance_probe")]
pub use group_invariance_probe::{
    GroupAction, Matrix, SubgroupClass, SubgroupReport, classify_subgroup, classify_subgroup_with,
    commutant_basis, commutant_binary_association, commutant_of_matrices, commutant_shift,
    discover_subgroup, discover_subgroup_into, invariance_score, score_concentration,
    score_variance,
};

#[cfg(feature = "causal_head_importance")]
pub use causal_head_importance::{
    ScaleNormalizedFusion, SpanLogitDiffReadout, direct_effect_importance,
    fuse_across_capabilities, indirect_effect_importance, partition_by_causal_score,
    per_capability_score,
};
// Adaptive Causal Calibration (Proposal 004) — cheap-proxy escalate. Opt-in.
// Re-exported alongside the causal head-importance primitives it builds on.
#[cfg(feature = "adaptive_causal_calibration")]
pub use causal_head_importance::{adaptive_partition, suspect_indices};

// Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic
// (Plan 431, Research 417, arXiv:2607.08393 — Knowing-Using Gap). Two
// modelless primitives: (1) `permeation_scan_into` — a 2D
// `(src_stage, dst_stage)` intervention heatmap reusing Plan 358's
// `direct_effect_importance` as the cell score, plus two-cluster
// classification; (2) `RelocateOp` — an applied operator that snapshots an
// anchor's state at one stage and overwrites at another, with the paper's
// `(0.82L→0.45L) + (0.10L→0.45L)` fixed default. Both behind
// `cross_stage_relocation` feature flag, opt-in. Implies `causal_head_importance`
// (the cell-score function). Phase 3 defend-wrong PoC in `riir-poc/` is
// MANDATORY before any promotion — the 58–75% recovery is a quality claim on
// the paper's LLM substrate, not ours.
#[cfg(feature = "cross_stage_relocation")]
pub mod cross_stage_relocation;
#[cfg(feature = "cross_stage_relocation")]
pub use cross_stage_relocation::{
    ClusterClass, PermeationMap, RelocateOp, RelocatePair, RelocatingForward, permeation_scan_into,
    permeation_scan_square_into,
};

#[cfg(feature = "latent_trajectory_geometry")]
pub use latent_trajectory_geometry::{
    BifurcationResult, LatentTrajectoryGeometry, bifurcation_ratio, fast_acos, from_states,
    from_states_into,
};

#[cfg(feature = "swe_trajectory_freeze")]
pub use swe_trajectory_freeze::{
    FrozenAttempt, FrozenValueAttempt, GeometrySummaryEncoder, StateMagnitudeEncoder,
    SweTrajectoryFreezer, SWTF_MAGIC, SWTF_VERSION, TrajectoryFreezeEnvelope, derive_directions,
    derive_directions_and_centroid,
};

#[cfg(feature = "latent_confounder_audit")]
pub use latent_confounder_audit::{AuditScratch, LatentConfounderAudit, audit_confounders};

#[cfg(feature = "interpolation_geometry")]
pub use interpolation_geometry::{
    EuclideanLatentSpace, FixtureRng, GaussianMixtureSpace, ImauveScore, InterventionReport,
    LatentSpace, imauve_score, intervention_battery,
};

#[cfg(feature = "viable_manifold_graph")]
pub use viable_manifold_graph::{
    ClosurePredicate, GraphBuildConfig, SafeManifoldGraph, ViabilityPredicate, VolumeFieldConfig,
    build_safe_manifold_graph, manifold_curiosity_walk, manifold_geodesic, manifold_random_walk,
    pullback_volume,
};

#[cfg(feature = "certified_frontier")]
pub use certified_frontier::{
    CertifiedFrontier, DilationFeasibility, DualPosteriorBuffer, FrontierCell, FrontierConfig,
    LinearPosterior, PosteriorBuffer,
    SIGMOID_LIPSCHITZ, SPHERE_EXCLUSION_MAX_CENTERS, SphereExclusion, advance_horizon,
    beta_mean_variance, beta_union_bound, confidence_schedule, laurent_massart_radius, linear_information_gain, prefer_dual, should_advance,
    sphere_exclusion_coverage, spherical_cap_bound, vendi_diversity,
};

// Plan 580 T4.1 — grow-then-navigate: certified cells become the node source
// the Viable Manifold Graph was missing. Needs both halves.
#[cfg(all(feature = "certified_frontier", feature = "viable_manifold_graph"))]
pub use certified_frontier::{CertifiedManifoldGraph, certified_manifold_graph};

#[cfg(feature = "ac_prefix")]
pub use ac_prefix::{AcPrefix, AcPrefixMask};

#[cfg(feature = "flow_field_nav")]
pub mod flow;
#[cfg(feature = "flow_field_nav")]
pub use flow::{
    FlowField, FlowFieldCache, FlowFieldConfig, LeoPotentialGrid, blend_steering, fft_smooth,
    fft_smooth_into, flow_steering, inflate_obstacles, should_use_flow_field,
};

// Spectral primitives — Fourier-basis algebra on discrete samples.
// Distilled from the FNO practical-perspective survey (Research 307).
// Each operator ships behind its own feature flag and is independently GOAT-gated.
// - `continuation` (feature `fourier_continuation`, Plan 323): Fourier
//   continuation for non-periodic latent fields — closed-form polynomial
//   periodic extension so the FFT does not produce Gibbs ringing at the
//   boundaries. The one modelless FNO primitive the codebase genuinely
//   lacked (Research 307 §3 candidate plan #1). DEFAULT-ON (Plan 323, 2026-06-25): G1–G4 ALL PASS.
// - `differentiation` (feature `spectral_differentiation`, Plan 325):
//   standalone FFT-based spectral differentiation on periodic uniform 1D
//   grids — multiply FFT coefficients by `(iω)^m`, IFFT back. The
//   specialized 1D-periodic case where DEC `exterior_derivative` is
//   overkill. DEFAULT-ON (Plan 323, 2026-06-25): G1–G4 ALL PASS.
#[cfg(any(feature = "fourier_continuation", feature = "spectral_differentiation"))]
pub mod spectral;
#[cfg(feature = "fourier_continuation")]
pub use spectral::continuation::{
    FcConfig, FcScratch, FourierContinuationError, MAX_POLY_ORDER, fourier_continue,
    fourier_continue_into,
};
#[cfg(feature = "spectral_differentiation")]
pub use spectral::differentiation::{
    MAX_ORDER, SpecDiffConfig, SpecDiffError, SpecDiffScratch, spectral_differentiate,
    spectral_differentiate_into,
};

// Merkle octree — hierarchical BLAKE3 commitment for KG latent octree nodes (Plan 221-M).
#[cfg(feature = "merkle_octree")]
pub mod merkle;
#[cfg(feature = "merkle_octree")]
pub use merkle::{
    HASH_SIZE, MERKLE_OCTREE_DEPTH, MERKLE_OCTREE_INTERNAL, MERKLE_OCTREE_LEAVES,
    MERKLE_OCTREE_NODES, MerkleOctree, MerkleProof,
};

// Curator verification layer for Merkle octree (Plan 253).
#[cfg(feature = "merkle_octree")]
pub mod curator;
#[cfg(feature = "merkle_octree")]
pub use curator::{
    CuratorArm, CuratorBandit, CuratorVerdict, FrozenTarget, MerkleEnvelope, MerkleFrozenStore,
    verification_weight,
};

// RTDC — Resolution-Tiered Deterministic Commitment (Plan 302, Research 280).
// Wraps `MerkleOctree` with 3 per-depth roots aligned to SLoD σ-boundaries,
// enabling trust-minimized semantic zoom: a light client verifies its
// fog-of-war view is a faithful sub-summation of the chain-committed full KG,
// with O(log n) proof at the abstraction level it operates at.
//
// Phase 1 ships the open primitive (types + trait + depth-2 sound proofs).
// Cross-depth soundness (`subtree_inclusion`) is Phase 3: Candidate C
// (probabilistic sampling) shipped behind `rtdc_subtree_inclusion`.
// Candidate A (Pedersen deterministic) research closed dormant — see
// `riir-chain/.research/006_RTDC_Candidate_A_Pedersen_Resolution.md`.
// LatCal-backed `DeterministicLeafEncode` impl lives in riir-chain (Plan 003).
#[cfg(feature = "rtdc")]
pub mod rtdc;
#[cfg(feature = "rtdc")]
pub use rtdc::{
    DepthSelector, DepthTieredMerkleOctree, DepthTieredRoots, DeterministicLeafEncode, RtdcError,
    RtdcProof,
};
#[cfg(feature = "rtdc_subtree_inclusion")]
pub use rtdc::{RTDC_SUBTREE_DEFAULT_K, SubtreeProof, min_k_for_95pct_confidence};

// GPart isometric partition adapter — replaces LoRA's bilinear BA with single isometric Pθ_d (Plan 257).
#[cfg(feature = "gpart_adapter")]
pub use types::{GPART_MAGIC, GPART_VERSION, GpartAdapter, GpartPair, GpartPrepared};

#[cfg(feature = "dendritic_gate")]
pub mod dendritic_gate;
#[cfg(feature = "dendritic_gate")]
pub use dendritic_gate::{DendriticGate, dendritic_sigmoid};
#[cfg(feature = "dendritic_gate")]
pub use simd::{coincidence_score, entropy_f32};

// CompressionDrafter — Hot-tier modelless LZ4 corpus-as-model drafter (Plan 285,
// Research 256, nathan.rs/gzip-lm). The compressor IS the model: score candidate
// continuations by compressed length against a frozen corpus. Corpus is appendable
// for online learning and is itself the wired format (bytes + BLAKE3).
// Opt-in until G1–G3 GOAT gate passes.
#[cfg(feature = "compression_drafter")]
pub mod compression_drafter;
#[cfg(feature = "compression_drafter")]
pub use compression_drafter::{CompressionDrafter, Lz4FlexDrafter};

// BabelCodec — Readability-relaxed semantic codec (Plan 331, Research 312,
// arXiv:2606.19857 BabelTele). Successor text codec to CompressionDrafter:
// where CompressionDrafter failed G2 twice on the Seal corpus (byte-level LZ4
// matching on short quest-grammar strings), BabelCodec operates on semantic
// STRUCTURE (BT-P8 fixed symbolic mapping rules) — purpose-built for KG-triple
// / entity-attribute / config / quest-grammar surfaces. Ships three pieces:
// (1) generic `BabelCodec` trait, (2) `FixedRuleTextCodec` (deterministic BT-P8
// text codec, the modelless subset of BabelTele), (3) `SigmoidLatentCodec<D>`
// (generic-trait facade over existing DensityBudget infrastructure, latent-level
// analog — value is API uniformity, NOT new capability), plus BLAKE3 commitment
// for the future LatCal chain bridge (Issue 002, resolved + removed). Sigmoid, not softmax.
// Opt-in until the G1–G5 GOAT gate passes — the same G2 (≥ 2× on real corpus)
// gate that killed CompressionDrafter twice.
#[cfg(feature = "babel_codec")]
pub mod babel_codec;
#[cfg(feature = "babel_codec")]
pub use babel_codec::{
    BabelCodec, BabelCommitment, BabelPair, CompressedLatent, FixedRuleTextCodec,
    SigmoidLatentCodec,
};

// Analytic Lattice — k×k transport operator chain composer + ASOC trait shapes
// + direction-vector SIMD decoder + spectral audit (Plan 330, Research 311).
// katgpt-core half: pure math primitives + generic trait shapes (NO GpuFuture
// import — leaf-clean). The ComposerTick: GpuFuture impl + Join3 combinator
// ship in riir-engine under the `analytic_lattice_runtime` feature (Phase 1b).
// Opt-in until G1–G6 GOAT gate passes.
#[cfg(feature = "analytic_lattice")]
pub mod analytic_lattice;
#[cfg(feature = "analytic_lattice")]
pub use analytic_lattice::{
    ChainError, ComposerCtx, LatticeVector, PlasmaDraft, RederiveOp, TransportOperator,
    apply_operator_into, audit::AuditReport, audit::spectral_audit, batch_compose_chain,
    batch_compose_chain_into, compose_chain, compose_chain_into, decoder::direction_vector_decode,
    decoder::direction_vector_decode_into, decoder::direction_vector_decode_slice,
};

// Functional Attention — closed-form Tikhonov spectral transport operator
// (Plan 286, Research 257, arxiv 2605.31559, Xiao et al. ICML 2026). DUAL FORM
// matching the reference implementation (`.raw/FUNCATTN/PDE-StandardBenchmark/model/
// Functional_attention.py`): convex-combo regularization `(1-α)·K̃ᵀK̃ + α·I_d`,
// column-normalized slice tokens, per-slice-token to_q/to_k/to_v linear
// projections. Sigmoid-basis default per AGENTS.md (partition-of-unity holds
// for any row-normalized non-negative kernel). Gain-tier open primitive:
// paper itself defers NLP validation (§6); promote only after G1–G5 GOAT
// gate passes.
#[cfg(feature = "funcattn")]
pub mod funcattn;
#[cfg(feature = "funcattn")]
pub use funcattn::{
    FuncAttnBasis, FuncAttnConfig, FuncAttnError, FuncAttnScratch, compute_basis_into,
    funcattn_forward, pre_rotate_basis_weights_into, solve_convex_combo_dual,
};
// Plan 332 — principled multi-scale basis constructors (DCT-log, Haar-packet).
// gated by the dedicated `funcattn_structured_basis` feature (implies funcattn).
#[cfg(feature = "funcattn_structured_basis")]
pub use funcattn::{make_dct_log_basis, make_haar_packet_basis};

// Plan 353 — Head Substitution Gate (Gain-tier, opt-in). Small decision
// struct that decides when a FuncAttn-style surrogate should substitute for
// a real attention head, using the paper's IoU cheap-proxy (§3 Fig 5b r>0.9)
// + cached FaithfulnessProfile veto (Plan 287 SinkAware cadence). NOT a new
// primitive — the original draft proposed a redundant ProgramSynthesizedHead
// primitive that was dropped after re-review identified FuncAttn (above) as
// the existing primitive surface. Stays opt-in: Gain-tier, and the plan's
// own Risk note flags it as borderline-thin for a feature flag.
#[cfg(feature = "functional_substitution_gate")]
pub mod functional_substitution;
#[cfg(feature = "functional_substitution_gate")]
pub use functional_substitution::{HeadSubstitutionGate, iou, worst_case_behavior_delta};

// Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN (Plan 310,
// Research 291, arxiv 2605.31559). Generalizes FUNCATTN to d_src ≠ d_dst,
// enabling train-on-small-deploy-on-large latent transfer without retraining.
// Open primitive: frozen BLAKE3-committed bases + zero-alloc transport.
// DEFAULT-ON (Plan 310 Phase 4, 2026-06-23): G1 mean cos 0.8944>=0.85, G2-A rank
// preservation, G3 elbow k=8, G4 0 allocs.
#[cfg(feature = "cross_resolution_transport")]
pub mod cross_resolution;
#[cfg(feature = "cross_resolution_transport")]
pub use cross_resolution::{
    CrossResScratch, CrossResolutionBases, CrossResolutionError, project_to_spectral_into,
    reconstruct_from_spectral_into, transport_cross_domain_cross_resolution_into,
    transport_cross_resolution, transport_cross_resolution_into,
};

// Latent Field Steering — top-down direction-vector injection into mutable
// latent state (Plan 309, Research 290, CAA + functional emotions). The missing
// fourth quadrant: CNA mutates neurons, EmotionDirections is read-only, FPCG
// refuses mutation — this injects directly into the latent state on the hot
// path. Zero-alloc SIMD SAXPY + sigmoid-falloff localized support.
// DEFAULT-ON (Plan 309 Phase 4, 2026-06-23): G1-G5 ALL PASS (G2 mean cos 0.9958,
// G4 19.2us<1ms, G5 0 allocs).
#[cfg(feature = "latent_field_steering")]
pub mod latent_steering;
#[cfg(feature = "latent_field_steering")]
pub use latent_steering::{
    BELIEF_AROUSAL, BELIEF_CALM, BELIEF_DESPERATION, BELIEF_DIM, BELIEF_FEAR, BELIEF_VALENCE,
    FieldSupport, LatentField, LatentSteeringError, LatentSteeringVector, apply_field_to_crowd,
    apply_latent_steering, apply_latent_steering_weighted, kernel_weight,
};

// Subspace Steering Field — k-dim manifold steering (Plan 412, Research 393,
// arxiv 2606.25234 Goodfire BSF). The k-dim generalization of Plan 309: an
// orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`, math
// `s' = s + Σ_j α_j · u_j`. At K=1 bit-identical to Plan 309; at K≥2 enables
// manifold walking (sweep alphas over a grid → concept variations). Pure
// modelless consumer of pre-discovered blocks (Plan 301 Jacobian SVD,
// SpectralQuant eigenbasis, or hand-constructed sets). Phase 3 finding:
// Newton-Schulz DIVERGES on non-square K<D matrices → Gram-Schmidt is the
// orthonormalize constructor. DEFAULT-ON (Plan 412 Phase 5, 2026-07-08):
// G1+G3+G4+G5 ALL PASS — G1 K=1 parity 0 mismatches/800 comparisons, G3 0
// allocs/1000 calls × K={1,2,4}, G4 sizes 68/104/176 bytes exact, G5
// commitment + walk_manifold deterministic.
#[cfg(feature = "subspace_steering")]
pub mod subspace_steering;
#[cfg(feature = "subspace_steering")]
pub use subspace_steering::{
    SubspaceSteeringError, SubspaceSteeringField, apply_subspace_steering, block_energy,
    compute_block_commitment, walk_manifold,
};

// Region-Conditioned Subspace Field — MFA local-geometry steering (Plan 416,
// Research 396, arxiv 2602.02464 Shafran et al. "From Directions to Regions").
// The region-conditioned generalization of Plan 412: K regions, each with a
// centroid μ_k and a local R-dim factor-analyzer subspace W_k. Two-mode
// steering: centroid interpolation (move toward a region) + local subspace
// offset (walk within a region). Per-region sigmoid membership gates (reformulated
// from the paper's softmax responsibilities per the AGENTS.md sigmoid mandate —
// more expressive: multi-region membership). Pure modelless consumer of a frozen
// MFA-like artifact {μ_k, W_k, Ψ, π} (trained offline via riir-train GD, or
// deterministically constructed via K-means + per-region PCA). At the degenerate
// limit (K=1, μ=0, W=I) steer_local is bit-identical to Plan 412. DEFAULT-ON (Plan 416 Phase 4, 2026-07-09):
// G1–G5 ALL PASS (G1 K=1 parity is the load-bearing gate).
#[cfg(feature = "region_subspace_steering")]
pub mod region_subspace;
#[cfg(feature = "region_subspace_steering")]
pub use region_subspace::{
    RegionDecomposition, RegionSubspaceError, RegionSubspaceField, compute_field_commitment,
    reconstruct,
};

// Phase-Modulated Subspace Rotation Gate — norm-preserving latent coupling
// `cos α ⊙ a + sin α ⊙ b` with phase from a sigmoid projection onto a frozen
// direction vector (Plan 322, Research 305, arxiv 2605.12700 UFO). The
// genuinely-new operation class: every other latent op in the crate is
// additive / convex-combo / dot-projection / wedge-detection / linear-transport
// / spatial-sum — none has the `sin²α+cos²α=1` Pythagorean norm-preservation
// invariant. §3.5 modelless Path 2 unblock: the trained `γ_θ` is replaced with
// `α = sigmoid(⟨state, direction⟩ · λ) · π/2` (closed-form). DEFAULT-ON (Plan 322 Phase 2, 2026-06-25):
// G1–G4 ALL PASS (G1 norm-preservation <1e-4 is the kill switch).
#[cfg(feature = "phase_rotation_coupling")]
pub mod phase_rotation;
#[cfg(feature = "phase_rotation_coupling")]
pub use phase_rotation::{
    PhaseRotationError, PhaseRotationGate, PhaseRotationScratch, compute_phase_from_projection,
    compute_phase_per_channel_into, phase_rotation_gate_into,
};

// GRAPE-M — Rank-2 Rodrigues Exponential for arbitrary plane (a, b).
// Distilled from Zhang et al. *GRAPE* (arXiv:2512.07805, ICLR 2026 §2.3).
// O(d) closed-form application of `exp(n·ω·L)·x` where `L = abᵀ − baᵀ` (rank-2
// skew) — two dot products + one FMA triad, never materialises the d×d matrix
// (beats LieRE's O(d³) torch.matrix_exp). Subsumes phase_rotation's scalar
// 2D rotation as the canonical-basis special case `a = e_i, b = e_{i+D/2}`;
// the new capability is rotation in a *learned* plane (per-NPC HLA personality
// rotation, per-shard rotation in MerkleFrozenEnvelope — see Issue 159).
// Pure modelless float arithmetic; learning the plane is → riir-train.
// STAYS OPT-IN — G1–G4 GOAT gate ALL PASS (Bench 457, 2026-07-17); promotion
// deferred per Issue 159 T6: gain is modelless but perf-only on a NEW capability
// (arbitrary-plane rotation), not a faster way to do something the crate already
// does. Re-evaluate when a concrete consumer lands (riir-ai HLA personality
// rotation, riir-neuron-db per-shard rotation).
#[cfg(feature = "grapem_rodrigues")]
pub mod grapem;
#[cfg(feature = "grapem_rodrigues")]
pub use grapem::{GrapemError, Rank2Plane, grapem_apply_into};

// PositionGroupAction — unified trait (RoPE / ALiBi / FoX / Wall / NoPE /
// GRAPE-M) per GRAPE §2.2 + §4.1. Vocabulary bridge for position-encoding-
// agnostic tooling (KV compaction, attention matching). Every positional
// encoding is an instance of G(n) = exp(n·ω·L); the trait abstracts over
// where the generator L lives (SO(d) multiplicative vs GL(d+2) additive
// homogeneous lift). Hot-path code keeps using PositionFreeCompactor /
// WallDiagonalGate directly; the trait is for cold-path interop.
// Implies grapem_rodrigues (GrapeMAction wraps Rank2Plane).
// STAYS OPT-IN — G1–G4 GOAT gate ALL PASS (Bench 458, 2026-07-17); promotion
// deferred per Issue 160 T4: vocabulary bridge — no hot-path consumer today.
// Re-evaluate when a position-encoding-agnostic tool (KV compactor, attention
// matcher) lands. Should be promoted together with grapem_rodrigues.
#[cfg(feature = "position_group_action")]
pub mod position_group_action;
#[cfg(feature = "position_group_action")]
pub use position_group_action::{
    AlibiAction, FoxAction, GrapeMAction, NopeAction, PositionGroupAction, RopeAction, WallAction,
};

// RoVE — Rotary Value Embeddings Attention (Plan 557, Research 452,
// arXiv:2606.11275). Extends RoPE from Q/K to the V projection + inverse-
// rotates the aggregated output, yielding an attentive convolution with
// offset-indexed block-Toeplitz kernel ψ_δ = R_δ·W_V. Parameter-free,
// FlashAttention-compatible. The first hot-path consumer of GRAPE's
// PositionGroupAction trait — turns the "vocabulary bridge" into a real
// attention variant. Implies position_group_action (consumes RopeAction).
// STAYS OPT-IN until Phase 5 retrofit PoC settles the open question of
// whether inference-time RoVE onto RoPE-trained checkpoints helps or hurts.
#[cfg(feature = "rotary_value_embedding")]
pub mod rotary_value_embedding;

// GRAPE-AP — Vector-Similarity Path-Integral Decay Gates (GRAPE §5).
// Content-aware extension of Wall Attention: ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)
// with g = log_sigmoid. Tokens whose positional embedding matches the query's
// decay slower; mismatching tokens decay faster. The headline gain in the
// GRAPE paper (+1.15 avg on 770M FineWeb-Edu). Wall Attention is the scalar
// special case (endpoint-independent embeddings). The positional-embedding
// projection is user-supplied (modelless); learning it is → riir-train.
// STAYS OPT-IN — G1–G5 GOAT gate ALL PASS (Bench 459, 2026-07-17); promotion
// deferred: positional-embedding projection is user-supplied; no hot-path
// consumer yet. The G5 magnitude gate was revised from "divergence > 2× noise
// floor" (infeasible on unit-norm synthetic) to a direction check; the
// magnitude gate is deferred to riir-train integration.
#[cfg(feature = "grape_ap_vector")]
pub mod grape_ap;
#[cfg(feature = "grape_ap_vector")]
pub use grape_ap::{GrapeApError, GrapeApGate, RotationSchedule, log_sigmoid};

// GRAPE Joint Lift — GL(d+2) block-diagonal composition of rotary + additive
// (GRAPE Appendix E, arXiv:2512.07805). Single-pass fused score that composes
// GRAPE-M (Issue 159's Rank2Plane, the rotary SO(d) part) with GRAPE-A
// (paper §4.1, the additive logit bias via softplus gates) into one group
// action. Closes the composition story: today Wall *replaces* RoPE; this
// primitive proves they *compose* into a single one-parameter subgroup of
// GL(d+2) while preserving the exact relative law. The plane (a, b) and gate
// vectors (u, v) are user-supplied (modelless); learning is → riir-train.
// Implies grapem_rodrigues (wraps Rank2Plane). Decoupled omega_rot/omega_add
// is a strict generalization of the paper's shared ω.
// STAYS OPT-IN — G1–G4 GOAT gate ALL PASS (Bench 460, 2026-07-17); promotion
// deferred per Issue 163 T6: thin composition layer — value is unified API +
// correctness guarantee (Appendix E's block-diagonal GL(d+2) proof), not a
// perf gain over calling the parts separately. No hot-path consumer today.
#[cfg(feature = "grape_joint_lift")]
pub mod grape_joint_lift;
#[cfg(feature = "grape_joint_lift")]
pub use grape_joint_lift::{GrapeJointLift, JointLiftError, softplus};

// Spherical Steering — single-target geodesic Slerp rotation
// `sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T` toward a unit-norm target
// direction on S^{d-1}, with sigmoid-translated vMF confidence gate (Plan 405,
// Research 382, arxiv 2602.08169 You/Deng/Chen ICML 2026). Sibling to Plan 322's
// 2-subspace phase rotation — same norm-preservation thesis, different
// parameterization: Plan 322 rotates *within* the (a,b) plane; Plan 405 rotates
// *toward* a target outside the input's direction (Slerp identity holds for all
// θ ∈ (0,π)). vMF gate reduces to sigmoid via Eq 17:
// `δ = -tanh(κ·s_T) = 1 − 2·sigmoid(2κ·s_T)`. §3.5 modelless Path 3 (closed-form
// trig + sigmoid; no training). DEFAULT-ON (Plan 405 Phase 2, 2026-07-06):
// G1-G5 ALL PASS.
#[cfg(feature = "spherical_steering")]
pub mod spherical_steering;
#[cfg(feature = "spherical_steering")]
pub use spherical_steering::{
    SlerpError, SlerpScratch, slerp_steering_into, spherical_steering_into, vmf_confidence_gate,
};

// Sphere Sampling — modelless primitives for sampling from unnormalized
// densities on the unit hypersphere S^{d-1}. Distilled from Flow Sampling
// (arxiv 2605.03984 Havens/Karrer/Shaul FAIR+Weizmann May 2026; Issue 544). The
// paper trains a drift u_θ via backprop; we ship only the modelless core:
// for vMF-family targets r(x) = κ·μ^T x the score ∇_M r is closed-form, so the
// entire conditional drift on the sphere integrates via Euler–Maruyama with no
// learned component. Three primitives: parallel_transport_householder_into
// (Eq 42 Householder reflection about X_1+X_t midpoint hyperplane),
// jacobian_logdet_cot_correction (Eq 44 curvature `(d−1)·(t·cot(t·ω_1) − cot(ω_1))·Ẋ_1/ω_1`),
// sphere_exp_map_into (Riemannian exp, the Euler–Maruyama step from Eq 29).
// Sibling to Plan 405 above: Plan 405 deterministically pulls a drifted vector
// toward μ_T via Slerp + Eq-17 gate; this module samples a distribution over
// directions via Euler–Maruyama on the manifold. The deterministic gate
// produces one direction; sampling produces a distribution — a capability
// class the gate cannot serve. Opt-in pending Issue 544 defend-wrong PoC
// verdict: the most likely failure mode (per Research 049 PTRM cautionary
// flag) is that the Riemannian sampler produces the same KL as Wood (1994)'s
// exact vMF sampler at the same N — in which case the complexity is
// unjustified for the vMF-only case. Promotion requires a non-vMF consumer.
#[cfg(feature = "sphere_sampling")]
pub mod sphere_sampling;
#[cfg(feature = "sphere_sampling")]
pub use sphere_sampling::{
    COT_FLOOR, EXP_MAP_FLOOR, SphereError, TRANSPORT_FLOOR, jacobian_logdet_cot_correction,
    parallel_transport_householder_into, sphere_exp_map_into,
};

// MAG — Mining via Activation Geometry (Plan 418, Research 397, arXiv:2607.04222
// LeVi/David/Fomin ICML 2026 FAGEN). Unsupervised direction mining + modelless
// transfer prediction. The missing acquisition step for the direction-vector
// ecosystem: today every direction is designer-authored (Plan 309) or
// supervised-extracted (Plan 162). MAG mines them unsupervised from the host's
// own verdict y_M. mine_direction / mine_contrast_direction extract unit-norm
// feature directions; reconstruction_error gives the ϵ_Q linearity diagnostic;
// calibrate_alpha normalizes injection strength; apply_operator computes the 8
// readout summaries; transfer_score / rank_candidates predict dataset transfer
// (the §4 94.7% Top-1 result). Mined directions are BLAKE3-committed (same
// envelope as LatentSteeringVector / MerkleFrozenEnvelope). Pure modelless
// (mean-difference + cosine geometry). DEFAULT-ON since 2026-07-09 (Phase 2
// GOAT G1-G6 ALL PASS): G2 (the headline kill-it gate) verified contrast
// directions from self-labeled classes ARE linearly separable (LOO acc 0.925
// at σ=1.5, 0.810 at σ=3.0). G4: MAG class-conditional transfer Top-1 0.720
// vs raw cosine 0.220 (3.3×). Phase 2 added mine_direction_into +
// transfer_score_into zero-alloc hot-path variants.
#[cfg(feature = "mag_mining")]
pub mod mag;
#[cfg(feature = "mag_mining")]
pub use mag::{
    DataSet, MagDirection, MagError, MagOperator, RankEntry, TransferMetric, apply_operator,
    calibrate_alpha, mine_contrast_direction, mine_direction, rank_candidates,
    reconstruction_error, transfer_score,
};
// NOTE: `apply_operator_into` is NOT re-exported at crate root — it collides with
// `analytic_lattice::apply_operator_into` when both features are on. Access it
// via `katgpt_core::mag::apply_operator_into`.

// ChunkedContentStore — Lore-distilled chunked content-addressed Merkle store (Plan 448, Research 262).
// Open primitive: chunks → BLAKE3 → dedup via papaya → binary Merkle root. No game/chain IP.
// Consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow).
//
// NOTE: the binary-Merkle `MerkleProof` here is renamed on re-export to
// `BinaryMerkleProof` to avoid colliding with `merkle_octree::MerkleProof`
// when both features are active simultaneously (caught by `cargo check
// --all-features`). Internal callers still reach the type via
// `crate::content_store::MerkleProof`.
#[cfg(feature = "chunked_content_store")]
pub mod content_store;
#[cfg(feature = "chunked_content_store")]
pub use content_store::{
    BlobId, ChunkFetcher, ChunkRange, ChunkedContentStore, ChunkerConfig, ChunkingStrategy,
    FastCdcChunker, FixedSizeChunker, InMemoryChunkedStore, MerkleProof as BinaryMerkleProof,
    StoreStats, build_binary_merkle_proof, build_binary_merkle_root, verify_binary_merkle_proof,
};

// Closure-Expansion Instrument (CEI) — PTG recorder + motif miner + PRI/CDG/TaR metrics
// (Plan 290, Research 264, arxiv 2606.15386, Momennejad & Raileanu). Open measurement
// layer: turns open-ended inference into observable metrics. DEFAULT-ON
// (Plan 290 T4.7 + G4 fix, 2026-06-26): G1 67us<100us, G2 638us<5ms, G3
// synthetic-proxy monotone, G4 0.296MB<1MB. 55/55 tests green.
#[cfg(feature = "closure_instrument")]
pub mod closure;
#[cfg(feature = "closure_instrument")]
pub use closure::{
    OperatorKind, PrimitiveKind, PrimitiveTransitionGraph, PtgEdge, PtgNode,
    admit::{GateResult, MotifAdmitter, RejectionReason},
    bridge::{
        DEFAULT_MOTIF_DIRS, MotifDirections, motif_embedding_to_tar_score, ptg_to_motif_embedding,
    },
    commitment, deserialize_postcard,
    metrics::{CdgScore, PriScores, compute_cdg, compute_pri, compute_tar_score, motif_multiset},
    mining::{SleepCycleClosureReport, fold_cdg_at_sleep_cycle, mine_motifs_at_sleep_cycle},
    motif::{
        FixedU32Set, MAX_MOTIF_EDGES, MAX_MOTIF_NODES, Motif, MotifMiner, RING_BUFFER_K,
        enumerate_subgraph_hashes,
    },
    serialize_postcard,
    trace::{DEFAULT_TRACE_CAPACITY, NodeId, PtgRecorder},
};

// Issue 040 — PTG × latent_functor edge composition. Ships `FunctorPtg`
// (composite wrapper over an unchanged `PrimitiveTransitionGraph`),
// `FunctorEdgeParams` (per-edge continuous-functor params), and
// `apply_functor_edge_into` (zero-alloc sigmoid-gated apply path). Gated by
// `ptg_functor_edges` (implies `closure_instrument`). Wire-format safe: the
// inner PTG is byte-identical to a bare PTG.
#[cfg(feature = "ptg_functor_edges")]
pub use closure::{FunctorEdgeParams, FunctorPtg, apply_functor_edge_into, functor_edge_gate};

// Sink-Aware Attention — NOP/Broadcast classifier + dual-policy sigmoid gate
// (Plan 287, Research 258, arxiv 2606.08105, Fesser et al.). Per-head
// classifier (value-norm-ratio + stable-rank-of-update) decides whether a
// sink is Adaptive NOP (gate it via sigmoid) or Broadcast (preserve it).
// Staged integration: the policy enum + standalone apply_dual_policy_gate
// ship here; direct wiring into parallax_attn / funcattn forward paths is
// deferred until synthetic G2 + latency G3 gates pass on a real model
// (validation fallback per Plan 287 §Validation).
//
// Plan 404 (2026-07-06): the parent module is now always-on. The pure
// information-theoretic substrate (markov/nll/typical_set/dirichlet_energy/
// claim) moved here from root `src/data_probe/`. The sink-aware classifier
// (`sink_classify`) + `geometry` remain gated `sink_aware_attn` inside the
// module. The gated re-exports below preserve `crate::data_probe::SinkKind`
// etc. for internal consumers (notably `parallax_attn.rs`). The always-on
// re-exports (markov/nll/typical_set/dirichlet_energy/claim) live in
// `data_probe/mod.rs`.
pub mod data_probe;
#[cfg(feature = "sink_aware_attn")]
pub use data_probe::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

// mi_est — Modelless Mutual-Information Estimator over fixed critics
// (Plan 583, Research 521 — MINE arXiv:1801.04062 modelless extraction).
// DV/NWJ/InfoNCE/JS bound VALUES in nats + LOO bias control + K-ladder
// tightness + permutation calibration (uniform/circular/block/stratified,
// dCor non-vacuity control) + a Gaussian closed-form arm gated by the shipped
// `sketched_gaussianity` + the frozen-representation IB ratio. Opt-in
// feature `mi_est` — diagnostic surface with no default consumer yet
// (no-default-consumer rule); consumers: riir-train dist-guard third audit
// axis (T3.4), quant-fidelity probes (T3.5), riir-train plan 365 DV core.
#[cfg(feature = "mi_est")]
pub mod mi;

// ICT Distributional Branching-Point Detector — open generic math (Plan 294,
// Research 270, arxiv 2606.19771). Collision purity β(π) = Σ π² (proven
// unconditionally monotone, ICT §A.2.5 — H₁ is wrong below π > e⁻¹ ≈ 0.37),
// Rényi H₂, Jensen-Shannon divergence to group mean, BranchingDetector
// (top-k% selector over K candidate trajectories + per-step β EMA), and the
// Bebop H₁→H₂ acceptance-forecast upgrade. No game semantics, no chain;
// runtime fusion (CLR gating, HLA updates, KG emission) is riir-ai Plan 324.
// STAYS OPT-IN — G3 (Spearman ρ < 0.5) PASS (Bench 294/3) AND G10
// (Bebop H₁→H₂ upgrade) PASS (Bench 294/10). Default-on promotion requires
// G8 (riir-ai Plan 324 runtime validation) per Plan 294 §Phase 8 T8.4 — G3
// alone is necessary but not sufficient. See .benchmarks/294_ict_promotion.md.
#[cfg(feature = "ict_branching")]
pub mod ict;
#[cfg(feature = "ict_branching")]
pub use ict::{
    AcceptanceForecastH2, BranchingDetector, BranchingReport, branching_point_mask,
    branching_point_mask_into, collision_purity, collision_purity_into, is_critical_branching,
    js_divergence, js_divergence_batch, renyi_h2, shannon_h1,
};

// ── Induced Code World Model (Plan 296, Research 275, arxiv 2510.04542) ───────
//
// Open half of the CWM Super-GOAT: a marker trait over `GameState` for forward
// models that are verifiable, BLAKE3-committable, and hot-swappable. The
// LLM-induction pipeline that *produces* an `InducedCwmKernel` impl is private
// (riir-ai Plan 326). The runtime never sees the LLM — only the frozen kernel.
#[cfg(feature = "induced_cwm")]
pub mod induced_cwm;
#[cfg(feature = "induced_cwm")]
pub use induced_cwm::{
    BeliefInferenceFn, CwmCommitment, InducedCwmKernel, TransitionTestFailure, TransitionUnitTest,
    make_transition_tests_from_trajectory, verify_transition,
};

// Phase 2 (Plan 296 T2.1–T2.5): Information-Set MCTS over an induced CWM +
// belief fn. Self-contained search tree (does NOT reuse root-crate
// `mcts_search` — that lives in katgpt-rs/src, katgpt-core cannot depend on the
// root). Gated by `induced_cwm_ismcts` (which auto-enables
// `induced_cwm`).
#[cfg(feature = "induced_cwm_ismcts")]
pub use induced_cwm::{InformationSet, NodeStats, ismcts_search_with_inference};

// ── Bisimulation Operator Inference (Plan 324, Research 308, arxiv 2602.19260) ─
//
// Open primitive: quotient an observed transition graph into bisimulation-
// equivalent state classes (signature-based partition refinement, O((S+E)
// log² S log d)) and infer an abstract PDDL-like operator schema. The
// lighter-weight PDDL-side counterpart to Induced CWM (Plan 296): where CWM
// induces executable *code* via an LLM, this induces an *operator schema* via
// a deterministic graph algorithm. Closes Research 264 §2.2 gaps #1 (PTG) +
// #2 (motif mining). Opt-in by design — downstream pipelines (riir-ai NPC
// runtime, riir-chain LatCal consumer) opt in by enabling the feature.
#[cfg(feature = "bisimulation_operator_inference")]
pub mod bisimulation;
#[cfg(feature = "bisimulation_operator_inference")]
pub use bisimulation::{
    BisimulationQuotient, OperatorDef, OperatorLabel, OperatorSchema, Plan, QuotientEdge,
    StateClassId, StateId, Transition, TransitionGraph, TransitionGraphBuilder, partition_refine,
    plan as bisimulation_plan,
};
// Issue 586 — rule-application consistency metric (BDH-CQ §6.4 analog):
// 3-bin strict/partial/none task histogram + sigmoid-guarded gap +
// structure-preservation breakdown + complexity-cluster regime detection.
// Gates `infer_operators` output via `promotion_verdict`; the
// ComplexityClustered regime is the exemplar-seeking trigger consumed by
// riir-ai Issue 672.
#[cfg(feature = "operator_consistency")]
pub use bisimulation::consistency::{
    ApplicationOutcome, ConsistencyGateConfig, ConsistencyRegime, ConsistencyReport,
    PromotionVerdict, promotion_verdict, rule_consistency,
};

// ── FORE — Fitted Occupancy-Ratio Estimator (Plan 438, Research 423, arxiv 2607.05375) ─
//
// Open primitive: generic modelless fitted-iteration estimator for the
// discounted occupancy ratio ω_π,γ = d^π,γ / d_ν in offline policy evaluation.
// The substrate-independent contribution is the adjoint Bellman KL contraction
// (paper Lemma 3.1): each fitted KL projection contracts relative entropy by
// factor γ, so convergence requires only realizability of the target ratio —
// no Bellman completeness of a value/critic class. Three downstream fusion
// targets (CLR re-estimation stabilization, freeze/thaw convergence guarantee,
// FORE-ratio state equivalence) are tracked in Research 423 as out-of-scope
// follow-ups requiring PoC validation. Phase 1 ships the type/trait surface
// only; Phase 2 adds the fitted-iteration loop once Algorithm 1 is verified.
// Opt-in — promotion to default-on requires a downstream consumer to
// demonstrate the gain empirically (riir-poc Fusion A).
#[cfg(feature = "occupancy_ratio")]
pub mod occupancy;

// ── Personality-Weighted Layer Composition (Plan 297, Research 276) ──────
//
// Open MIT-licensed primitive for the Entity Cognition Stack Super-GOAT.
// A `PersonalityWeightedComposition<N, D>` kernel composes `N` latent
// direction vectors via per-layer sigmoid-gated weights, then drifts those
// weights via a reward-surprise Hebbian update. Zero-allocation, sigmoid-gated
// (NOT softmax — per AGENTS.md), belief-gated, BLAKE3-snapshot-integrated.
// Entity-agnostic (NPC, player, predator, prey, robot, recommender user).
//
// Consumed by riir-ai Plan 327 (runtime wiring) — the game-specific 7-layer
// mapping, archetype table, taming transition stay private in riir-ai.
// DEFAULT-ON (Plan 297 Phase 4): G4 (79.585ns < 1µs target) + G5 (zero alloc) PASS.
//
// Substrate lives in the katgpt-personality crate (Issue 007 Phase E Tier 2
// #5, 2026-06-28). Re-exported here as `katgpt_core::personality_composition`
// for backwards compatibility — all `crate::personality_composition::*` paths
// resolve unchanged. The `personality_composition` Cargo feature turns on the
// `dep:katgpt-personality` dependency; the substrate compiles unconditionally
// inside the crate itself.
#[cfg(feature = "personality_composition")]
pub use katgpt_personality as personality_composition;
#[cfg(feature = "personality_composition")]
pub use personality_composition::{
    ArchetypeLabel, EntityCognitionComposition, LayerDirectionSource, PersonalityConfig,
    PersonalitySnapshot, PersonalityWeightedComposition, sigmoid as personality_sigmoid,
    sigmoid_into as personality_sigmoid_into,
};

// ── Committed Field Blend (Plan 321, Research 302) ───────────────────────
//
// Open MIT-licensed primitive: the sampling-invariant half of the FAME
// Super-GOAT. A `CommittedFieldBlend<N, D>` computes blend weights pi ONCE
// from a trajectory summary via sigmoid projection, then FREEZES them for
// the entity's lifetime. The blended field f_pi(z) = Σ_k sigmoid(pi_k/tau) ·
// f_k(z) governs dynamics; because both pi and the fields are frozen, the
// trajectory is sampling-invariant (FAME Proposition 3 / Young-integral).
// Zero-alloc apply + BLAKE3-committed. Reuses personality_composition's
// sigmoid + simd::simd_fused_scale_acc (DRY).
// DEFAULT-ON since 2026-06-28 (Issue 005 executed): Plan 321 G1–G5 + riir-ai
// Plan 336 G6a–G6e + G7a ALL PASS. G2 (sampling invariance, the make-or-break
// gate) worst-case Δpi=1.19e-6/100 entities. Private selling-point guide at
// riir-ai/.research/158.
#[cfg(feature = "committed_field_blend")]
pub mod committed_field_blend;
#[cfg(feature = "committed_field_blend")]
pub use committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend, TriArchetypeBlend};

// ── Variable-Rank Domain Expert Clusters (Plan 558, Research 453) ─────────
//
// Open MIT-licensed composition layer: applies LatentMoE's transferable
// principle (arXiv:2601.18089 — the paper itself is PASS; this distills the
// principle) to per-NPC cognition. Different NPC tasks have different
// intrinsic feature ranks (movement ~8 dims, combat ~16, quest/social ~32).
// Compressing each domain to its rank ℓ_d and scaling expert count by
// α = D_full/ℓ_d preserves total K×D compute while boosting archetype
// diversity (1.63× entropy gain validated in Research 453 PoC).
//
// Three small primitives over the existing `CommittedFieldBlend<N, D>`
// (Plan 321, DEFAULT-ON): pick_domain (argmax routing), project_guided
// (zero-cost dim gather — NOT blind JL/PCA, mitigates Plan 230 cautionary
// flag), VariableRankRouter<DOMAINS> (heterogeneous-rank dispatch).
//
// Opt-in — Plan 558 GOAT gate pending; promotion to default requires
// release-mode latency ≤1.0× baseline at 10K NPCs.
#[cfg(feature = "variable_rank_domain_expert")]
pub mod variable_rank_domain_expert;

// ── Engram — Hash-Addressed Pattern Memory (Plan 299, Research 278) ───────
//
// Open MIT-licensed primitive: the first conditional-MEMORY axis in the
// katgpt stack (complementary to Raven's conditional-COMPUTATION axis).
// N-gram-suffix → multi-head hash → O(1) slot lookup → sigmoid gate (RMSNorm
// dot σ) → residual-fuse into hidden state. Frozen table, atomic swaps for
// updates, BLAKE3 commitment as sync-boundary audit artifact.
//
// CRITICAL: sigmoid, not softmax — per AGENTS.md. No `softmax` symbol here.
//
// Open half of the Engram Super-GOAT: private selling-point guide lives in
// riir-ai Guide 147; chain commitment bridge is riir-chain R001 (TODO).
// Compiled in default transitively via cognitive_architecture_root → engram chain (Issue 039). G1+G2+G4 PASS; G6 (effective depth) deferred to riir-ai integration (requires live inference pipeline).
#[cfg(feature = "engram")]
pub mod engram;
#[cfg(feature = "engram")]
pub use engram::{
    CacheResult, CacheTier, ColdFetcher, EngramConfig, EngramHash, EngramHotSwap, EngramTable,
    EngramTableBuilder, EngramTableId, HashHead, IDENTITY_KERNEL, InMemoryEngramTable, K_MAX,
    SigmoidFusionConfig, StagingEngramTable, StagingError, SurjectiveMap, SurjectiveMapLoadError,
    TokenId, TokenizerSpec, ZipfianCacheHierarchy, ZipfianStats, ZipfianStatsSnapshot,
    build_merkle_root, build_surjective_map, compress_token, conv_causal_dyn_into,
    conv_causal_into, fuse_into_hidden_state, multi_head_hash, rmsnorm_into, sigmoid_fuse_into,
    sigmoid_fuse_multi_branch_into, try_compress_token,
};

// Issue 656 — counterfactual privilege gating for engram fusion (modelless δ
// from LOPD, riir-train Research 419 §5.2). Adds the missing *utility* axis to
// the similarity-only gate: `out = (base_gate · σ((Δ_slot − m)/s)) · v`, where
// Δ is an outcome-weighted EMA of the counterfactual advantage
// `score(state + fuse) − score(state)`. Modelless — two evaluations and a
// comparison, no gradients. Opt-in.
#[cfg(feature = "engram_privilege")]
pub use engram::{
    CreditAssignment, PrivilegeConfig, PrivilegeLedger, PrivilegeTrace,
    fuse_into_hidden_state_privileged, sigmoid_fuse_scaled_into,
};

// ── Product Key Memory — O(√N) Factored Retrieval (Plan 408, Research 387) ─
//
// Open MIT-licensed primitive: the fourth complexity class in the retrieval
// stack (Raven O(1) / Engram O(1)-hash / δ-Mem O(r) / PKM O(√N)). Splits a
// d_k-dim query into two halves, scores two √N codebooks, takes top-k of the
// k² Cartesian product — yielding `2√N + k²` cost instead of `N`. Scales to
// ~10⁶ slots at sub-linear retrieval cost.
//
// Modelless (constraint #1): the FwPKM paper's gradient-descent half (L_mem
// GD on V, L_addr GD on K, n-iter TTT) is forbidden. Replaced by shipped
// δ-rule analog (Plan 053). This primitive ships ONLY the inference-time
// factored retrieval; the optional δ-rule write path lands in Phase 5
// (product_key_memory_episodic).
//
// Phase 1 (this commit): types only — const-generic
// `ProductKeyMemory<SQRT_N, D_K, D_V>`, `ScoreFn` (Dot/Idw), fixed-size
// `PkQuery<K>`. Leaf-clean (zero deps). Phase 2 ships the kernel + GOAT gate.
// DEFAULT-ON since 2026-07-07 (Plan 408 Phase 3 GOAT): G1 latency 1670×
// speedup, G2 top-k Jaccard 1.0000 vs brute-force, G3 IDW centroid-ness PASS,
// G4 0 allocs/1000 steady-state query_into calls. See `.benchmarks/408_pkm_goat.md`.
#[cfg(feature = "product_key_memory")]
pub mod product_key_memory;
#[cfg(feature = "product_key_memory")]
pub use product_key_memory::{
    D_K_FLOOR, PkEntry, PkQuery, PkmScratch, ProductKeyMemory, SQRT_N_FLOOR, ScoreFn, score_dot,
    score_idw,
};

// MOP — Maximum Occupancy Principle value-iteration primitive (Plan 573 /
// Research 478, arXiv:2205.10316). The paper's Eq. 7 fixed-point map in
// log-space LSE form over a frozen tabular kernel — reward-free optimal
// policy with emergent survival (absorbing states V=0 bit-exact) and a β
// risk knob. Opt-in `mop_path_entropy`. Consumers: riir-ai Plan 538.
#[cfg(feature = "mop_path_entropy")]
pub mod mop;
#[cfg(feature = "mop_path_entropy")]
pub use mop::{MopConfig, MopConfigError, MopScratch, MopSolver, MopSolution};
// Phase 4 (F4 fusion) — freeze/thaw wrapper around ProductKeyMemory. Gated
// separately so the leaf-clean retrieval primitive (above) stays usable
// without the Arc<RwLock<Arc<...>>> + BLAKE3 commitment machinery. See
// `product_key_memory/freeze.rs` for the pattern rationale (mirrors
// `induced_cwm/hot_swap.rs`).
#[cfg(feature = "product_key_memory_freeze")]
pub use product_key_memory::FrozenProductKeyMemory;

// Plan 408 Phase 5 — δ-rule write gate over PKM (F1 fusion: PKM × δ-Mem).
// PkmEpisodicStore wraps FrozenProductKeyMemory + a mutable working copy.
// Gated on `product_key_memory_episodic` (implies `product_key_memory_freeze`).
#[cfg(feature = "product_key_memory_episodic")]
pub use product_key_memory::PkmEpisodicStore;

// Gain/Cost Loop Halting Primitive — open substrate-agnostic kernel for per-loop
// halting decisions (Plan 304, Research 282, arXiv:2606.18023, LoopCoder-v2).
//
// halt when marginal refinement gain < marginal drift cost × τ; oscillation
// early-halt via cos θ < 0; L_min floor protects representational capacity.
// Composes with the shipped elastic-loop override (Issue 035) — Phase 2 will wire
// this into `forward_looped()` (separate scope). Phase 1 ships the kernel only.
//
// Latent vs Raw: gain/cost signals are local latent (per-loop hidden-state
// deltas); the halt count L is a deterministic raw scalar safe to sync/replay.
//
// STAYS OPT-IN — G2/G3/G4 GOAT gate ALL PASS (Bench 304, 2026-06-23); synthetic
// kernel-only bench confirms the contract on three reference regimes (crowd-NPC
// savings, important-NPC no-regression, oscillation detection). Real-world
// validation requires actual game loops → riir-ai Plan 330 is the gating
// dependency. The cost_floor is the load-bearing knob for G2.
#[cfg(feature = "gain_cost_halt")]
pub mod gain_cost_halt;
#[cfg(feature = "gain_cost_halt")]
pub use gain_cost_halt::{
    GainCostLoopHalter, HaltDecision, HaltReason, angular_change, hidden_erank, step_size,
};
// Issue 699 T1-T3 — structural CoT halting (TRACE, arXiv:2510.07880):
// answer-space cycle detection on reasoning traces — the black-box halt
// family (no logits/hidden states/LLM rater). A third independent halt-vote
// family beside the numeric arbiter above; the two compose via
// structural_cot_halt::vote_from_numeric when BOTH features are on.
// Opt-in (not default-on) — T4 defend-wrong PoC (riir-poc) + T5 GOAT gate
// (≥30% token savings at ≤1% accuracy delta) are pending.
#[cfg(feature = "structural_cot_halt")]
pub mod structural_cot_halt;
#[cfg(feature = "structural_cot_halt")]
pub use structural_cot_halt::{
    BacktrackRevisitHalt, ClassifiedPattern, HaltPolicy, HaltVote, Pattern, SelfLoopHalt,
    StructuralHaltDecision, StructuralHaltReason, StructuralTransition, StructuralTraceMonitor,
    compose_votes, normalized_answer_hash,
};

// Issue 720 T1 — ConvergenceCadence (Research 529, HRM mechanistic dissection,
// Finding 4): windowed update-magnitude OUTCOME classifier — solved runs'
// ‖Δz‖ decays (0.30 by step 7-8), failed runs plateau HIGH (1.46, ~4.9×).
// The outcome read the halt-only families above lack: halt ≠ classify, and
// plateau-high churn warrants ESCALATION (damp per Issue 717 — tangential-
// first, cos_updates ≈ 0 — deliberate per NPC think loops, restart per CGSP).
// Three laws pinned in-module: absolute Δ (never relative — R35/717-T6
// trap), windowed shape (never single-step), tangential-first. Zero-alloc
// fixed ring, caller-fed norms. Opt-in — T2 A/B (riir-poc `b18b7b2bb`) +
// T3 consumer (mmorpg Issue 054 L2, riir-ai `681786288`) LANDED 2026-09-04.
#[cfg(feature = "cadence_gate")]
pub mod convergence_cadence;
#[cfg(feature = "cadence_gate")]
pub use convergence_cadence::{CadenceConfig, CadenceVerdict, ConvergenceCadence};

// Cross-Datapoint Set Attention — sigmoid-gated, permutation-equivariant
// cross-entity refinement kernel (Plan 354, Research 354, arXiv:2106.02584
// Kossen et al. NeurIPS 2021, Non-Parametric Transformers). The inference-time
// operator only — training of Q/K/V via BERT-style masking stays in riir-train.
// Substrate-agnostic: `&[f32]` → `&mut [f32]`, no opinion on what the vectors
// mean. The riir-ai runtime (Plan 355) wires it onto HLA belief states for
// crowd-scale NPC joint inference; the open primitive is just the math.
//
// Sigmoid gates (NEVER softmax per AGENTS.md §2) — each pair α_ij ∈ (0,1)
// independently, so an entity may attend to 0 peers (lonely), 1 peer (paired),
// or many peers (formation). Softmax would force artificial competition.
//
// Permutation-equivariant by construction (NPT Lemma 4, Appendix A) —
// shuffling input rows shuffles output rows identically. The G1 test
// verifies this bit-exactly.
//
// Latent vs Raw: the primitive is substrate-agnostic. The sync boundary is
// the caller's responsibility (see the riir-ai runtime plan 355 for the
// HLA-specific wiring + the unchanged 5-scalar bridge).
//
// DEFAULT-ON since 2026-07-01 (Plan 354 Phase 2 + Plan 355 G6/G7/G9):
// G1 permutation equivariance bit-exact, G2 identity-floor meaningfulness,
// G3 latency 21.96µs at N=64, G4 0 allocs/100 calls, G5 sigmoid-not-softmax
// lonely-query correctness. riir-ai runtime G6 fusion cosine sim <0.95 (fusion
// adds value over identity), G7 crowd stability <5% drift over 100×2000 ticks,
// G9 production latency 75.7µs mean/tick at 100 NPCs. G8 collective inference
// FAILED (Super-GOAT→GOAT) — averaging cannot amplify detection; use-case
// limitation, NOT a primitive defect. Validated selling point: crowd coherence.
#[cfg(feature = "set_attention")]
pub mod set_attention;
#[cfg(feature = "set_attention")]
pub use set_attention::{
    SetAttentionConfig, SetAttentionError, identity, identity_into, identity_projection,
    identity_projection_into, set_sigmoid_attention_into,
};
#[cfg(feature = "clr_weighted_set_attention")]
pub use set_attention::{clr_reliability_scores, clr_weighted_set_attention_into};

// Depth-Invariance Diagnostic + Magnitude-Regularized Residual — the
// root-cause counterpart to four symptom-only detectors (BeliefRankPruner,
// GainCostLoopHalter, latent_functor/reestimation.rs,
// micro_belief/coherence_bench.rs). Modelless math, no game semantics.
// Classifies recursive latent-state chains as DepthInvariant /
// DepthSpecificRefinement / Collapsed / Insufficient. The MagnitudeReg
// wrapper is the modelless fix for kernels we own (HLA, functor,
// micro_belief, engram, Raven); for frozen MLPs (BeliefDrafter) the fix
// requires retraining → riir-train.
// Plan 306 Phase 1+5; Research 286; arXiv:2605.09992 Eldenk et al.
// DEFAULT-ON (Plan 306 T7.4, 2026-06-23): G1 8/8 PASS + G4 re-spec'd to absolute-latency PASS.
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance;
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance::{
    DepthInvarianceConfig, DepthInvarianceDiagnostic, DepthInvarianceKind, MagnitudeRegularization,
    Scratch, apply_magnitude_regularization, classify_chain, classify_chain_batched,
};

// Shared linear-algebra kernels. Originally extracted for `karc`'s ridge-style
// solvers (Plan 308); the f32 Cholesky/ridge path lives here as a standalone
// extraction of the PEIRA `(N + λI)⁻¹` pattern — see the module note for why
// PEIRA's f64 path is left untouched. Plan 319 (Clifford geometric product)
// and Plan 326 (Tucker/HOSVD tensor factorization) ship peers under `linalg::`
// — each must also gate this `pub mod` so the crate compiles when only that
// feature is on. Issue 684 (`svd_cca`) joins the same rule for its
// `symmetric_eig` consumption.
#[cfg(any(
    feature = "karc_forecaster",
    feature = "geometric_product",
    feature = "tucker_factorization",
    feature = "svd_cca",
    feature = "twist_smc",
    feature = "mi_est",
    // Both joined the rule late and broke it silently: each consumes
    // `linalg::ridge_solve` and neither gated this `pub mod`, so
    // `--no-default-features --features <either>` failed E0433. Found by the
    // Issue 701 R1b default-on isolation sweep (2026-09-01) — the FIRST run of
    // that scope, and it found them in 4.1 minutes. Nothing else catches this
    // class: `--all-features` compiles the union, where some other consumer
    // always brings `linalg` in.
    feature = "velocity_field_ensemble",
    feature = "hebbian_kernel_memory",
    // Issue 707: `tpr::als` consumes `ridge_solve_direct_f32` / `cholesky_f32`
    // / `chol_solve_f32` for its four closed-form ALS blocks and the cached
    // projection factor — same rule, joined at birth rather than late.
    feature = "tpr"
))]
pub mod linalg;

// KARC — Kolmogorov-Arnold Reservoir Computing delay-basis-ridge forecaster
// (Plan 308, Research 288, arXiv:2606.19984). Modelless, inference-time
// trajectory forecaster: delay-embedding × sealed KarcBasis (Fourier/Chebyshev/
// BSpline) × closed-form ridge readout, with a zero-alloc forecast matvec.
// DEFAULT-ON (Phase 22, 2026-07-21, Plan 308 promotion under Issue 186 Path D3
// split-config G1 gate contract — see Cargo.toml default block + Bench 308).
#[cfg(feature = "karc_forecaster")]
pub mod karc;
#[cfg(feature = "karc_forecaster")]
pub use karc::{
    BSplineBasis, ChebyshevBasis, DelayRing, FitError, FourierBasis, KarcBasis, KarcForecaster,
    KarcScratch, LowRankFitScratch, chunked_gram_into, feature_expand, feature_expand_higher_order,
    forecast_low_rank_apply, higher_order_feature_count, low_rank_fit, low_rank_fit_warm_start,
};

// KARC Regime Gate re-export — Plan 556 Phase 1 (2026-07-20).
// The gate itself is a sibling submodule under `karc::`; the public surface
// is re-exported here for caller ergonomics so consumers don't need to
// reach into the `karc::regime_gate` path.
#[cfg(feature = "karc_regime_gate")]
pub use karc::regime_gate::{KarcRegime, KarcRegimeGate, RegimeVerdict, WelfordVariance};

// Plan 556 Phase 2 — KARC Batched MatVec. SIMD-batched forecast across N
// forecasters of identical (D, M, K) shape. Crowd-scale perf primitive
// (unblocks Plan 514 Phase 3 octree-batched dispatch). Pure modelless
// (linear algebra only). Re-exported here for caller ergonomics.
#[cfg(feature = "karc_batched_matvec")]
pub use karc::batched::{KarcBatchForecaster, karc_batched_matvec_into};

// Plan 556 Phase 3 — KARC LOD Tier. Config tag + tier-promotion Wout
// projection. Three nested tiers map to different KarcForecaster configs;
// promotion is a pure index remap (no re-fit required for forecast continuity
// on the surviving features). Re-exported here for caller ergonomics.
#[cfg(feature = "karc_lod_tier")]
pub use karc::lod_tier::{KarcLodTier, is_identity_projection, project_wout_lod_into};

// KarcShard DP Output Perturbation (Issue 370 T4) — post-hoc Gaussian noise
// on a fitted ridge Wout matrix to provide formal (ε,δ)-DP for the committed
// KarcShard parameters. Defends PARAMETER-INSPECTION MI (attacker reads Wout
// to detect memorized patterns). Does NOT defend Yeom loss-threshold MI —
// see karc_dp module docs and riir-ai/.benchmarks/399 for the structural
// insufficiency analysis. Modelless (post-hoc noise on a closed-form solve).
// Gated on karc_forecaster since it operates on the Wout produced by
// KarcForecaster::fit_ridge.
#[cfg(feature = "karc_forecaster")]
pub mod karc_dp;
#[cfg(feature = "karc_forecaster")]
pub use karc_dp::{KarcDpNoiseConfig, apply_dp_noise_to_wout};

// HOPE — Hilbert-Schmidt Capacity Kernel + Optimal Rank-1 Parent (Plan 469,
// Research 454, arXiv:2607.21366 Mobahi & Bartlett, Google DeepMind 2026-07-24).
// Closed-form math for scale-invariant capacity metric + optimal rank-1 merge
// of two PH-1 neurons modeled as rank-1 Hilbert-Schmidt operators.
// Distilled modelless core (ReLU self/cross kernels, principal eigenvector of
// rank-2 AᵀA, optimal scale s*=(a+b·E_rem)/(2·E_rem+b), capacity + prune/merge/
// block-eviction costs, Dantzig greedy selection). Fixes the AM single-query
// rank-1 collapse failure documented in Issue 001 / Plan 319 T5.6 G5 FAIL.
// Pure modelless — no training, no gradient descent. Zero runtime cost unless
// a caller invokes hope_capacity / optimal_rank1_parent / hope_greedy_select.
// DEFAULT-ON (Phase 23, 2026-07-24, Plan 469 Phase 4 T4.6 promotion — G1+G2+G3+G4
// ALL PASS per bench_469_hope_kernel_goat; see Cargo.toml default block).
#[cfg(feature = "hope_capacity")]
pub mod hope;

// Hebbian Kernel Memory — Closed-Form Fact-Storing MLP Construction + MLP Swap
// (Plan 559, Research 455, arXiv:2607.10034 Garcia et al., Stanford/UB 2026-07-10).
// Bilinear sketched-K₂ feature map + ridge-whitened readout achieves the
// information-theoretic optimal fact-storage capacity W=Θ(F·log F). Closed-form,
// no GD. HOPE × this primitive = Super-GOAT dual (HOPE measures capacity;
// this constructs it). DEFAULT-ON (Plan 559 Phase 3, 2026-07-25): G1+G2+G3+G4
// ALL PASS in Phase 1 (bench_559_hebbian_kernel_memory_goat); G5 Super-GOAT
// quality axis PASS (Bench 462 riir-neuron-db — Constructed=GD=1.000 edit_score
// at 2/5/10% edits vs Frozen 0.000; easy-regime caveat noted). Layer split
// (feature-gate-audit Defense 3): the IP-bearing private bridge
// `hebbian_fact_store` in riir-neuron-db STAYS opt-in (shard-specific value
// table source + BLAKE3-committed audit sidecar).
#[cfg(feature = "hebbian_kernel_memory")]
pub mod hebbian_kernel_memory;

// spectral_pencil — the affine matrix pencil scalar gate f(x) =
// λk(A₀ + Σ xᵢAᵢ) (Issue 676, Research 495, arXiv:2608.08003 "The
// Spectral Neuron"). Shape-by-construction + coefficient transparency
// + seeded γk ≥ ½ init. Opt-in; see the module doc for the determinism
// policy (pinned Jacobi/Sturm/QR — no library eigensolver on committed
// paths).
#[cfg(feature = "spectral_pencil")]
pub mod spectral_pencil;

// orthogonal_factorization — Issue 687 (Research 504, arXiv:2608.20065
// "Orthogonal JEPA" Path 0): orthonormalize (twice-reorthogonalized
// modified Gram–Schmidt + input L_orth defect — the one-shot redundancy
// audit production direction sets never had), per-(factor,coordinate)
// activity variance hinges (Welford, γ ≥ max(γ_min, c/√n) estimator-noise
// schedule), Parseval runtime invariants + exact truncation certificates
// (Hadamard integer-core bases, dyadic-exact at d=64), and construction-
// time head conditioning certificates via spectral_pencil (κ(B)=1 by
// construction — the paper's caveat converts to a certificate). Pure
// modelless closed-form linear algebra; scalar f64 reductions (bit-identical
// cross-platform); zero steady-state alloc. Opt-in until a consumer promotes
// (GOAT .benchmarks/676).
#[cfg(feature = "orthogonal_factorization")]
pub mod orthogonal_factorization;

// CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall (Plan 567,
// Research 466, Galitski "High-Capacity Generalized Hopfield Networks",
// alphaXiv 2607.hopfield-networks, JQI/UMD 2026-07-31).
//
// Associative-memory recall on the symmetric space CP^(d-1) = SU(d)/U(d-1)
// instead of the sphere. The memory kernel K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ| is a
// d×d Hermitian SPIKED random matrix and recall aligns the neuron with its TOP
// EIGENVECTOR, which is BBP-protected against GUE crosstalk. That gap is why
// CP^(d-1) capacity grows with d (α_c ≈ 0.62 at d=3, 2.41 at d=4) where gapless
// vector alignment on S^(n-1) decays as 4/(27n) — and it is the mechanism the
// Plan 276 AttractorKernel lacked when it failed G2.1 at random init.
//
// Pure modelless: closed-form SU(d) basis construction + Hebbian outer-product
// kernel + Rayleigh-quotient ascent. No gradient descent, so memories load from a
// frozen snapshot (freeze/thaw Path 1) rather than being trained.
//
// OPT-IN pending the Plan 567 GOAT gate. Load-bearing gates are G5 (the Plan 276
// modelless unblock) and G7 (whether the BBP gap survives at N=8/N=64 rather than
// only asymptotically in N). Zero runtime cost unless a caller constructs a
// CpHopfieldRecaller.
#[cfg(feature = "cp_hopfield")]
pub mod cp_hopfield;

// Transformer Inversion — SipIt open primitive (Plan 561, Research 232
// Gain-Redirects, arXiv:2510.15511 Nikolaou et al. ICLR 2026). Modelless,
// O(T·|V|) exact prompt recovery from a layer-ℓ hidden state matrix via
// per-position vocabulary search. Public-engine adoption hook for
// transparency / interpretability / audit tooling on standard decoder-only
// text transformers; NOT applicable to HLA (sigmoid-bounded 8-dim belief
// kernel, not a real-analytic text transformer) and NOT a sync-boundary
// compression primitive (transmitting a T×d hidden-state matrix is a
// ~7000× bandwidth INCREASE over the 20-byte scalar sync).
//
// Phase 1 (this module): skeleton + RandomPolicy (uniform-without-
// replacement) + G1 exact-recovery tests on a toy 2-layer GELU transformer.
// Phase 2 (gated on `grad_policy` sub-feature): GradientGuidedPolicy
// (paper Alg 3). Phase 3 (deferred): G2 latency + G4 alloc-free benches.
//
// Gain-tier parking rationale: §1.55 mandates the verdict (modelless +
// not shipped → Gain). The Gain tier routing "Plan only, behind feature
// flag" implies code ships feature-gated; promotion to default-on
// requires a concrete consumer that demonstrates a gain at the GOAT gate
// (Plan 561 T5.1 — unmet as of 2026-07-29: zero production consumers across
// the 7-repo stack; re-verified via grep for `transformer_inversion` /
// `katgpt_core::inversion` across all 7 repos — only self-references in
// this module + bench_561 + this lib.rs export + the
// `examples/transformer_inversion_01_forensics.rs` reference harness).
// The feature ships as adoption-hook infrastructure; the forensics demo
// (2026-07-29) closes the documentation gap (every public primitive in
// katgpt-rs ships an example harness) but is NOT a production consumer.
// Re-evaluate at the 2026-10-26 timeout (T5.2).
#[cfg(feature = "transformer_inversion")]
pub mod inversion;
#[cfg(feature = "grad_policy")]
pub use inversion::InversionGradient;
#[cfg(feature = "transformer_inversion")]
pub use inversion::{
    AcceptanceRegion, InversionConfig, InversionError, InversionForward, InversionPolicy,
    InversionResult, ObservedStates, RandomPolicy, accept_observation, accept_observation_into,
    invert_sequence, invert_sequence_into,
};

// Similarity Inference — endogenous correlation device for embedded equilibrium
// (Plan 526, Research 471, arXiv:2608.03958 Meulemans et al. *Paradigms of
// Intelligence* Aug 2026). Per-focal `SimilarityPosterior ω ∈ (0,1)` updated
// incrementally from joint-action history via the paper's §H.2 closed form
// `ω_T = α/(α+(1−α)·|A|^(−T))`; `embedded_best_response` switches from
// competitive-best-response (Nash) to cooperative-best-response (CCE) when ω
// crosses a payoff-derived threshold (exactly 0.5 for canonical PD). Pure
// modelless closed-form math; O(1) observe, O(A²) best-response, zero allocs
// after construction. The mechanism is genuinely novel per R471 §3.5: the
// shipped `CceLp<N,A>` (Plan 295, DEFAULT-ON) uses an *exogenous*
// designer-set correlation device ζ; this primitive *infers* an *endogenous*
// correlation device ω from interaction history. NOT a sync-boundary primitive
// (ω is latent per-focal; only the final cooperate/defect u8 action crosses).
// Sigmoid, not softmax (ω is a posterior probability, not a categorical).
// DEFAULT-ON (2026-08-11) — Plan 526 Phase 1-5 GOAT G1–G8 ALL PASS (Bench 579:
// G1 closed-form reproduction, G2 emergent cooperation, G5 indirect inference,
// G7 UQ floor, G8 PD threshold); see the Cargo.toml feature-def comment for the
// full gate record.
#[cfg(feature = "similarity_inference")]
pub mod similarity_inference;
#[cfg(feature = "similarity_inference")]
pub use similarity_inference::{
    JointActionHistory, PayoffMatrix, SimilarityError, SimilarityPosterior, canonical_pd,
    embedded_best_response, embedded_best_response_into,
};

// ARG Protocol Primitives — open half of the ARG × Latent Substrate Super-GOAT
// fusion (Plan 327 Phases 1-3, Research 309, Guide 160 private). Five generic
// protocol primitives distilled from the ARG Standard
// (https://protocol.airistech.ai/arg-core.html, Iris Technologies 2026):
// `PolicyEnvelope` (Step 1 hard gate), `TaxonomyValidator` (Step 3 deterministic
// label-set validator), `LifecycleState` + `RedirectTable` (Step E ontology
// lifecycle continuity), `TypedOfflineCandidate` + `CandidateIntent` (Step C
// typed offline candidate), `OfflineCandidateScorer` (Step C scoring with the
// G5 silence-bias penalty), `InfoRegistry` (Step 9 + Step C two-phase dedup
// with grey-zone review). Private runtime composition with HLA / Entity
// Cognition Stack / VMG / Sub-Goal Compaction lives in riir-ai Plan 337.
// No game/chain/shard semantics. DEFAULT-ON (Plan 327 Phase 4, 2026-06-25):
#[cfg(feature = "arg_protocol")]
pub mod arg;
#[cfg(feature = "arg_protocol")]
pub use arg::{
    AccessScope, CandidateIntent, CandidateKind, CompareFn, CompareResult,
    DEFAULT_AUTO_COMMIT_THRESHOLD, Evidence, EvidenceId, GainComponents, InfoKey,
    InfoOutcomeStatus, InfoRegistry, InfoType, InfoUnit, LabelId, LabelSet, LabelSignature,
    LifecycleState, MatchResult, MatchScratch, OfflineCandidateScorer, PayloadHash,
    PayloadHashCompare, PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, Provenance,
    RedirectTable, ResponseMode, ScoredCandidate, ShouldProceed, TaxonomyKind, TaxonomyNode,
    TaxonomyValidator, TypedOfflineCandidate, ValidationError, ValidationResult, ValidationScratch,
};

// Non-Interference Memory Branches — Super-GOAT fusion (Plan 329, Research 310,
// arXiv:2606.20638 Goel et al. Oxford Jun 2026). Five generic open primitives:
// BranchBank (bounded persistent CognitiveBranch bank with spawn/merge/prune
// lifecycle), BranchRouter (dot-product snap + Jaccard fallback), VerifierGate
// (reward + curiosity + centroid-quarantine write gate, composes with CLR),
// NonInterferenceProjection (orthogonal latent subspace per branch),
// BudgetCompiler (priority-cascade context compiler under fixed budget). Fuses
// BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience into a new
// capability class: per-NPC continual adaptation without catastrophic
// forgetting. Composes with arg_protocol LifecycleState when both features on.
// DEFAULT-ON (Plan 329 Phase 3, 2026-06-26): G1–G5 ALL PASS.
#[cfg(feature = "non_interference_branches")]
pub mod branching;
#[cfg(feature = "non_interference_branches")]
pub use branching::{
    AssignError, AssignResult, BranchBank, BranchId, BranchLifecycle, BranchRouter, BranchStats,
    BudgetCompiler, CognitiveBranch, CompiledContext, CompiledItem,
    DEFAULT_ASSIGN_MAX_INTERFERENCE, DEFAULT_BUDGET_BYTES, DEFAULT_MAX_BRANCHES,
    DEFAULT_ORTHOGONAL_EPSILON, DEFAULT_PROJECTION_DIM, DEFAULT_QUARANTINE_CENTROID_THRESH,
    DEFAULT_TAU_CURIOSITY, DEFAULT_TAU_JACCARD, DEFAULT_TAU_SNAP, DEFAULT_TAU_SPAWN,
    DEFAULT_TAU_WRITE, EpisodicEntry, FailureEntry, NonInterferenceProjection, PriorityTier,
    ProceduralRule, RetrievedMaterials, RouteMode, RouteResult, VerifierGate, WriteDecision,
    max_orthogonal_branches,
};

// Post-Candidate Branch Router — distilled from Local Branch Routing
// (arXiv:2606.25354, Yin et al. June 2026). The modelless inference mechanism
// distilled to its open primitive: forward K candidate next-tokens, score each
// post-candidate hidden state by dot-product onto a frozen direction, commit
// the argmax (or perturbed-argmax sample with Logistic noise — the sigmoid
// analog of Gumbel-max).
//
// Generalizes the shipped ColliderPruner::batch_is_valid_with_hidden from
// binary prune/keep to relative route-and-commit. PoC-confirmed modelless
// quality gain of +9pp to +26pp across 5 noise cells (Plan 377 Phase 1,
// riir-ai/crates/riir-poc). Set-attention variant adds zero modelless value
// (PoC §8 — within ±1pp of the dot-product router across v1 and v2) and stays
// a riir-train follow-up (needs trained Q/K/V projections).
//
// Sigmoid (NEVER softmax) per AGENTS.md §2: sampling uses Logistic(0, β)
// noise whose CDF is sigmoid(x/β), making the categorical sample a
// sigmoid-family operation without any exp/softmax normalization.
//
// DEFAULT-ON (Plan 377 Phase 3, 2026-07-04): GOAT PASS — G1 correctness ≥90%, G2 router
// latency <1µs at K=3 D=64, G3 K=1 bit-identical to standard decode, G4
// alloc-free hot path, G5 modelless, G6 sigmoid-not-softmax).
#[cfg(feature = "local_branch_routing")]
pub mod branch_routing;
#[cfg(feature = "local_branch_routing")]
pub use branch_routing::{
    ColliderRouterAdapter, DotProductRouter, PostCandidateRouter, PreservationScorer,
};

// Sleep-Time Query Anticipator — open primitive for offline query anticipation
// (Plan 334, Research 318, arXiv:2504.13171 Lin et al. Letta/Berkeley).
// Implements the open math half: SleepTimeAnticipator orchestrates per-direction
// sleep-time compute → emits reusable AnticipatedQuerySet (the c' artifact,
// BLAKE3-committed) → wake-time consume() does cheap dot-product + sigmoid-
// gated lookup, falling through to fresh compute on low-predictability queries.
// PredictabilityScorer trait + DotPredictabilityScorer default
// (p = sigmoid(α·dot(c,dir)+β)); AmortizationCostModel operationalizes the
// paper's §5.3 cost model. Game-specific direction-vector catalogs, NPC tiering,
// HLA wiring, and chain commitment live in riir-ai Plan 341 (private).
// Phase 1 ships traits + types + IdentityFunctorOp (synthetic-test default);
// Phase 2 ships synthetic gates G1/G2/G5/G6/G7. G2/G3/G4 quality gates require
// a real predictability-labeled corpus → deferred to riir-ai Plan 341.
// Opt-in until G1–G5 GOAT gate passes; promotion to default-on requires
// Plan 341 G1–G5 to clear on a real game corpus.
//
// Substrate lives in the katgpt-sleep crate (Issue 007 Phase E Tier 2 #6,
// 2026-06-28). Re-exported here as `katgpt_core::sleep_time` for backwards
// compatibility — all `crate::sleep_time::*` paths resolve unchanged. The
// `sleep_time_anticipation` Cargo feature turns on the `dep:katgpt-sleep`
// dependency; the substrate compiles unconditionally inside the crate itself.
#[cfg(feature = "sleep_time_anticipation")]
pub use katgpt_sleep as sleep_time;
#[cfg(feature = "sleep_time_anticipation")]
pub use sleep_time::{
    AmortizationCostModel, AnticipatedQueryDir, AnticipatedQuerySet, AnticipatedSlot,
    ConsumeMatchMode, DEFAULT_LATENCY_PREMIUM, DotPredictabilityScorer, IdentityFunctorOp,
    PredictabilityScorer, SLEEP_TIME_DEFAULT_K, SleepTimeAnticipator, SleepTimeComputeOp,
    SleepTimeScratch, commit_direction, consume, consume_gate, consume_gate_with_match_mode,
    consume_with_match_mode,
};

// PairedLossGap — generic modelless paired token-level loss gap diagnostic
// (Plan 335, Research 319, arXiv:2606.20936 Li & Merrill AI2). Pure
// measurement tool: given two log-prob traces over the same prefixes, compute
// per-token Δ_i = ℓ_A − ℓ_B, stratify by token class, report filtered
// aggregates (ALL / TOP-K∩NO-COPY / COPY-N-ONLY) that amplify small
// architecture gaps aggregate loss hides. ClassSizeBound exposes Proposition 1
// (DKL ≤ log|V_τ|) — the volume-of-support bound justifying raw-vs-latent
// sync. Generic math, no game/chain/shard semantics — legitimately public.
// NOT an inference mechanism (measurement tool only) → not Super-GOAT.
// STAYS OPT-IN — G1 + G2 + G2-alloc + G3 + G4 GOAT gate ALL PASS (Bench 335,
// 2026-06-27); not promoted to default because it's a measurement tool by
// nature — opt-in is the right shape (consumers opt in when running A/B).
#[cfg(feature = "paired_loss_diagnostic")]
pub mod paired_loss;
#[cfg(feature = "paired_loss_diagnostic")]
pub use paired_loss::{
    ClassGapReport, ClassGapRow, ClassSizeBound, CopyNGramTagger, FilterKind, FilterScratch,
    PairedLossGap, TokenClass, TokenTagger,
};

// TEMP — Perturbed-Loss-Vector Diversity Fingerprint (Plan 341, Research 323,
// arXiv:2606.26797 Jin et al. ICML 2026). Modelless diversity selector: given
// two committed snapshots S_0, S_1, extrapolate K checkpoints along v = S_1 − S_0,
// compute per-candidate short-prefix loss vectors, and select the K-subset with
// maximal Lipschitz-bound spread — gradient-diversity ranking without gradients.
// Theorem 3.1 modelless reframe: similar loss vectors across K extrapolated
// checkpoints ⇒ similar gradients along v during the next weight-mutation cycle.
// Composes with ac_prefix::ConditionalLogprob, HLA surprise, RavenSlotLossKernel
// (riir-neuron-db Plan 005). DEFAULT-ON (Plan 341 Phase 2, 2026-06-29): G1–G5 ALL PASS.
#[cfg(feature = "temp_loss_fingerprint")]
pub mod diversity;
#[cfg(feature = "temp_loss_fingerprint")]
pub use diversity::temp::{
    LossKernel, extrapolated_snapshot_schedule, lipschitz_gradient_bound, pairwise_bound,
    perturbed_loss_vector, select_diverse_subset,
};
// Plan 367 Fusion C — QMC variant of `extrapolated_snapshot_schedule`.
// Low-discrepancy noise coverage → more diverse loss vectors per unit K.
// Requires both TEMP substrate and the QMC source trait.
#[cfg(all(feature = "temp_loss_fingerprint", feature = "qmc_sampling"))]
pub use diversity::temp::extrapolated_snapshot_schedule_qmc;

// Manifold Bandits — Latent Task Tree + Hierarchical Thompson Sampler +
// BayesianFilterArm (Plan 370, Research 370, arXiv:2606.19750 McKenzie et al.
// UCSD 2026). Modelless inference-time routing primitive: frozen, BLAKE3-
// committable hierarchical clustering of an arm space + top-down Beta posterior
// descent + per-arm non-stationary Bayesian filtering. Closes the contextual +
// non-stationary bandit gap (Plans 030/032/025). The BMC training curriculum
// routes to riir-train; this ships the modelless inference-time routing
// primitive. DEFAULT-ON (Plan 370 Phase 2, 2026-07-03): G1+G3+G4+G5 PASS; G2 FAIL is plan-level expectation (curriculum-learning-specific).
#[cfg(feature = "manifold_bandit")]
pub mod manifold_bandit;

// Mean-Field Crowd Oscillation Regime Classifier — crowd-level (κ, κ_a, Q)
// order-parameter aggregator + closed-form 2×2 Jacobian Hopf boundary check +
// four-way regime taxonomy (Static / NoiseSustainedOscillation /
// IrregularSwitching / GlobalLimitCycle). Distilled from Zheng, Miller, Fiete
// (arXiv:2606.30366, MIT, Jun 2026). The paper's algorithmic content is ~80%
// covered by shipped primitives (LinOSS, `subspace_phase_gate`, `temporal_deriv`,
// `MicroRecurrentBeliefState`, `ict::BranchingDetector`); this ships the
// missing 20% — the crowd-scale mean-field view + oscillatory-instability
// detector + regime taxonomy. Extends Plan 301's `subspace_phase_gate` from
// real-eigenvalue phase transitions (`N ≥ d` input sufficiency) to complex-
// eigenvalue (Hopf) phase transitions. DEFAULT-ON (Plan 371 Phase 6, 2026-07-03): G1+G2+G3+G4+G5 PASS +
// mandatory defend-wrong PoC (Plan 371 Phase 5 T5.1) pass.
#[cfg(feature = "mean_field_regime")]
pub mod mean_field;
#[cfg(feature = "mean_field_regime")]
pub use mean_field::{
    DEFAULT_CLASSIFIER, HopfParams, MeanFieldOverlap, Regime, RegimeClassifier, hopf_boundary,
    static_boundary,
};

// Factorized Transition Action Abstraction — modelless compositional action
// latent primitive distilled from Nam et al., *Latent Actions from Factorized
// Transition Effects under Agent Ambiguity* (arXiv:2606.30544, Brown, 2026-06-30).
// Research 374, Plan 375. The factorized/compositional cousin of the shipped
// monolithic `latent_functor` (riir-ai Plan 273): frozen codebook of K D-dim
// effect primitives + Top-1 patch assignment + sigmoid relevance gate +
// normalized weighted average → compact action latent. Codebook constructed
// modellessly via Lloyd's k-means (Path 2 of AGENTS.md §3.5 — deterministic,
// no gradient descent). Sigmoid gating throughout (NEVER softmax per AGENTS.md
// §2, verified in `otf_lam/model.py::GateNetwork.forward()`). Opt-in until the
// G1–G6 GOAT gate (bench_375_factorized_action_goat) passes.
#[cfg(feature = "factorized_action")]
pub mod factorized_action;
#[cfg(feature = "factorized_action")]
pub use factorized_action::{
    AggregatorType, EffectCodebook, FactorizedActionLatent, FilmProjectionBank, MAX_K, MAX_PATCHES,
    TransitionFactors, aggregate_action_latent_into, factor_token_into, finalize_factors,
    fit_codebook_kmeans_into, motion_input_velocity_into, patchify_1d, relevance_score,
};

// Velocity-Field Ensemble — Algebraic Combination of Pre-Trained Models
// (Plan 376, Research 375, arXiv:2602.20070 Coeurdoux et al. ICML 2026 SPIGM).
// Combine P frozen pre-trained velocity fields (any forward model: LLM
// drafter, HLA forecaster, KARC forecaster, archetype operator field) into a
// single regression-optimal combined drift b̂(x) = Σ_i η_i · b_i(x), where η
// is solved once from N data pairs via the existing linalg::ridge_solve
// P×P Cholesky path (the SAME math KARC uses — KARC's basis is delay-embedded
// features; this primitive's basis is P frozen model forward outputs).
//
// The contribution is the *basis construction*, NOT the ridge solve — anyone
// reviewing should grep `ridge_solve_direct_f32` and confirm KARC's `fit_direct`
// is the same linear-algebra operation. No duplicate math; pure DRY reuse.
//
// η CAN be negative (signed combination, not probabilistic mixture). No
// softmax anywhere; no sigmoid on η either (η is regression-solved, not
// projected). The sigmoid-not-softmax rule applies to *gating*, not to
// regression-optimal weights.
//
// Includes the optimal-diffusion SDE integrator (paper Algorithm 1, eq. 14
// with D*_t = α_t γ_t / β_t) as a decoupled utility — composes with any drift
// source, not just the ensemble.
//
// DEFAULT-ON (Plan 376 Phase 3, 2026-07-04): G1–G4 ALL PASS. G2 (cross-domain
// quality) is the make-or-break gate — the paper proves cross-domain
// composition for image generation only; Phase 2 PoC is mandatory before any
// quality-parity claim for game AI.
#[cfg(feature = "velocity_field_ensemble")]
pub mod velocity_field_ensemble;
#[cfg(feature = "velocity_field_ensemble")]
pub use velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, Schedule, VelocityField, VelocityFieldEnsemble,
    accumulate_pair_into, stochastic_interpolant_step_into,
};

// VFD — Velocity-Field Disagreement score (Plan 432, Research 420).
// Modelless: consumes the same M frozen velocity fields as VelocityFieldEnsemble,
// but integrates each member independently and measures pairwise disagreement
// weighted by kappa_s = s/(1-s).
//
// Ships as an OPT-IN NON-UQ disagreement score. Phase 2 GOAT gate ran on
// 2026-07-13 (G1✅ G2❌ G3✅ G4✅ G5✅) — the make-or-break G2 UQ floor
// (per Issue 010) FAILED: optimal λ*=0 on both AR(1) + bimodal corpora means
// VFD's epistemic scaling adds zero calibrated-UQ value over the conformal-naive
// floor. VFD does NOT activate Plan 376 Phase 6's deferred UQ gate — the
// ensemble remains UQ-bearing on its own (Plan 376 Phase 6); VFD does not
// upgrade it. Useful for CLR L1 gating, sleep-time prioritization, and runtime
// failure detection (paper §6.4), but carries NO calibrated-UQ claim. Canonical
// GOAT record: `.benchmarks/432_vfd_goat.md`.
#[cfg(feature = "velocity_field_disagreement")]
pub mod velocity_field_disagreement;
#[cfg(feature = "velocity_field_disagreement")]
pub use velocity_field_disagreement::{VfdScore, VfdScratch, VfdVarianceSignal, vfd_score_into};

// ── Phase 10 absorption (Proposal 003, 2026-07-04): modules moved from katgpt-rs/src/.
// Always-on (no feature gate):
pub mod alloc; // Debug-only TrackingAllocator (consumer gates via #[cfg(debug_assertions)])
pub mod cumprodsum; // Cumprodsum primitive (Plan 263) — always-on
pub mod trigger_gate; // Compute-tier trigger gate — always-on
// ── Phase 12 absorption (Proposal 003, 2026-07-04): more modules moved from katgpt-rs/src/.
// Feature-gated (mirror root feature names):
#[cfg(feature = "critical_interval_gate")]
pub mod dllm_solver; // Discrete Critical Interval Solver Switching (Plan 222)
#[cfg(feature = "modality_pruned_load")]
pub mod pipeline_pruner; // Pipeline Pruner — modality-aware inference pipeline selection (Plan 227 Phase 3)
// ── Phase 12 T4.3: folder moves from katgpt-rs/src/.
#[cfg(feature = "breakeven_routing")]
pub mod breakeven;
#[cfg(feature = "closed_unit_compaction")]
pub mod compaction; // Closed-Unit Compaction Gate — CUCG (Plan 333)
#[cfg(feature = "cubical_nerve")]
pub mod cubical_nerve; // CubicalNerve CAT(0) cubical complexes (Plan 252 Phase 3)
#[cfg(feature = "mux_latent_context")]
pub mod mux_latent; // MUX-Latent Context Compression (Research 211, Plan 238) // Breakeven complexity cost-aware routing (Plan 250)
// Feature-gated (mirror root feature names):
#[cfg(feature = "cce_moderator")]
pub mod cce;
#[cfg(feature = "llmexec_guard")]
pub mod llmexec_guard;
#[cfg(feature = "memory_soup_lora")]
pub mod memory_soup_lora;
#[cfg(feature = "mux_demux")]
pub mod mux_demux;
#[cfg(feature = "salience_tri_gate")]
pub mod salience;
#[cfg(feature = "salience_tri_gate")]
pub use salience::{
    DelegateToken, FoldbackTarget, SalienceDecision, SalienceTriGate, SilenceToken,
};
#[cfg(feature = "channel_simd_align")]
pub mod channel_simd;
#[cfg(feature = "skill_opt")]
pub mod skill_opt;
#[cfg(feature = "ssd_block")]
pub mod ssd_block;

// GDN Rollback-Free Tree Verification — masked triangular solve for delta-rule
// speculative trees (Plan 424, Research 407, arXiv:2607.06763 §3.4). Reduces
// tree verification for GDN recurrent layers to (I+X)U=βV, eliminating state
// rollback entirely. Pure-math substrate: flat &[f32] slices, no Gdn2State/Config
// dep. STAYS OPT-IN — G1–G4 GOAT gate PASS (Bench 424, 2026-07-10; G2 wins on
// deep draft trees); not promoted to default because it only activates on
// opt-in GDN/QwenDeltaNet configs and provides significant speedup only on
// deep trees.
#[cfg(feature = "gdn_tree_verify")]
pub mod gdn_tree_verify;

// TILR — Trajectory-Invariant Latent Refinement (alignment-gated subspace
// correction). Plan 425, Research 408, arXiv:2606.29164 (ICML 2026 Mech Interp
// Workshop). The alignment-gated member of the subspace-projection family:
// projects a contrastive direction onto a frozen SVD basis, modulates the step
// size by the alignment fraction γ = ‖Πd‖/‖d‖ so that γ→0 bit-recovers the
// uncorrected input (strict no-harm guarantee). Pure linear algebra — flat
// &[f32] slices + SIMD dot products, zero `crate::` deps. Consumes a
// pre-computed SVD basis (Plan 301 thin_svd_into); does not compute it.
// DEFAULT-ON (2026-07-09): G1–G4 ALL PASS — see .benchmarks/425_tilr_goat.md.
#[cfg(feature = "tilr_invariant_subspace")]
pub mod tilr;
#[cfg(feature = "tilr_invariant_subspace")]
pub use tilr::{
    TilrError, TilrScratch, check_orthonormal, tilr_refine, tilr_refine_apply, tilr_refine_into,
};
// Phase 3 calibration helper needs Plan 301's thin SVD — gated on both features.
#[cfg(all(feature = "tilr_invariant_subspace", feature = "subspace_phase_gate"))]
pub use tilr::discover_invariant_subspace;

// Plan 426: MANCE — Manifold-Aware Concept Erasure. Local tangent + spectral
// weighting + trust-bounded erasure. Pure modelless linear algebra (k-NN +
// thin SVD + dot-product projections). Gated on subspace_phase_gate for the
// local tangent SVD (Plan 301 thin_svd_into).
#[cfg(all(feature = "manifold_erasure", feature = "subspace_phase_gate"))]
pub mod manifold_erasure;

// Issue 565 / Research 463: Quantization-Error Compensating Reader-LoRA —
// deterministically-constructed low-rank (weight-space SVD, output-space
// data-aware SVD) or sparse (top-K COO bypass) correction for quantized
// weight matrices. Pure modelless (closed-form SVD / partial-sort). Gated on
// subspace_phase_gate for the thin SVD machinery (Plan 301).
#[cfg(all(feature = "quant_error_lora", feature = "subspace_phase_gate"))]
pub mod quant_error_lora;
#[cfg(all(feature = "manifold_erasure", feature = "subspace_phase_gate"))]
pub use manifold_erasure::{
    ManceConfig, ManceError, ManceScratch, ManceStepInfo, ManceTangentCache,
    covmatch_second_moment_into, leace_first_moment_into, mance_plus_plus_step_into,
    mance_plus_step_into, manifold_erasure_loop_cached_into, manifold_erasure_loop_into,
    manifold_erasure_step, manifold_erasure_step_cached_into, manifold_erasure_step_into,
};

// Lifelong LaCAM Multi-Agent Pathfinding Substrate (Plan 440, Research 424,
// arXiv:2605.16855). Paper-faithful LLLG with four pluggable seams
// (CostFn, LocalGuidanceSource, WarmStartScheme, HindranceEstimator) for the
// Super-GOAT fusion (riir-ai/318: HLA × Crowd MCGS × P350). Pure modelless
// (heuristic only, no training). Opt-in until GOAT gate G1–G4 pass.
#[cfg(feature = "multi_agent_path")]
pub mod multi_agent_path;

// Plan 449: Poincaré Adapter — closed-form latent navigation primitive.
// Distillation of arXiv:2607.14228 (Chen et al., *SeeSE3: Emergence of 3D
// Space in Vision Features*, DeepMind, 15 Jul 2026). Research 449 ran the
// novelty gate (4/4 Super-GOAT). The open primitive ships a frozen
// `PoincareAdapter` Pod + the closed-form navigator `poincare_navigate_into`
// + the multi-step variant + an offline closed-form ridge fit. Private game-
// runtime selling point (NPC imagination) lives in riir-ai/.research/319.
//
// Pure modelless (closed-form PCA + ridge + SVD pseudoinverse; no gradient
// descent). Reuses Plan 301's `thin_svd_into` + Plan 308's
// `ridge_solve_direct_f32`. Gated on `subspace_phase_gate` for the SVD. DEFAULT-ON
// (Plan 449 Phase 3, 2026-07-18): G1–G7 ALL PASS — see .benchmarks/449_poincare_goat.md.
#[cfg(all(feature = "poincare_navigator", feature = "subspace_phase_gate"))]
pub mod poincare;
#[cfg(all(feature = "poincare_navigator", feature = "subspace_phase_gate"))]
pub use poincare::{
    FitConfig, LATENT_DIM_MAX, PHI_HIDDEN_DEFAULT, PHI_OUT_DEFAULT, PoincareAdapter,
    PoincareFitError, RIDGE_ALPHA_DEFAULT, TARGET_DIM_MAX, accumulate_pinv_into, eval_phi_into,
    fit_poincare_adapter, poincare_multi_step_into, poincare_navigate_into,
};

// Plan 571: Phase Separation Probe — per-entity minimum circular distance on
// a phase circle, distilled from the Lonely Runner Conjecture (Barajas & Serra
// 2007, arXiv:0710.4495; proven for N≤7, conjectured beyond). The LRC
// guarantees every entity cycles through phase_separation ≥ 1/N — a coverage
// guarantee no existing primitive provides. Three modelless paths (O(N),
// O(N²), O(N log N)) + two bridge helpers (raw time-phase, latent projection).
// Pure modelless (closed-form modular arithmetic + sigmoid + dot-product).
// DEFAULT-ON (2026-08-07, Plan 571) — G1–G4 GOAT gate ALL PASS
// (bench_571_phase_separation_goat).
#[cfg(feature = "phase_separation")]
pub mod phase_separation;
#[cfg(feature = "phase_separation")]
pub use phase_separation::{
    circular_distance, from_latent_projection, from_speeds_and_tick, phase_separation,
    phase_separation_all, phase_separation_sorted,
};

// Issue 680: Signed-Coupling Opinion Dynamics — Glauber (heat-bath) update on
// a SIGNED social graph plus the three crowd order parameters, distilled from
// "Physics of Agents" (El et al., arXiv:2608.16578; Research 497). The kernel
// is CLR set attention's sibling — the same σ(gated weighted sum) shape, but
// with signed, tie-typed couplings on a stance instead of unsigned relevance
// weights: h_i = β⁺Σ J⁺s + β⁻Σ J⁻s + β₀Σ|J|s + g_i, collapsed to one
// branch-free O(edges) pass over a CSR row. Ships the three reducers nothing
// else in the stack had: net_opinion (mean), crowd_conviction (mean of
// squares — genuinely new), and the χ = N·Var_t(|n|) susceptibility
// accumulator whose peak over a temperature sweep locates the critical social
// temperature. Pure modelless (the paper's only gradient descent fits ~19
// scalars to real LLM transitions; a game crowd AUTHORS its couplings, so the
// paper's fitted ranges become designer-facing defaults). OPT-IN — promotion
// waits on a production consumer, the CLR precedent.
#[cfg(feature = "signed_coupling_dynamics")]
pub mod signed_coupling;
#[cfg(feature = "signed_coupling_dynamics")]
pub use signed_coupling::{
    Couplings, InformedCouplings, PAPER_BETA_MINUS_RANGE, PAPER_BETA_PLUS_RANGE,
    PAPER_BETA_ZERO_RANGE, PAPER_TRUTH_GAP_RANGE, SignedGraph, SignedGraphError,
    SusceptibilityAccumulator, crowd_conviction, net_opinion, sample_states_into,
    signed_coupling_update_informed_into, signed_coupling_update_into,
};

// Plan 568: Recurrent Residual Quantization (RRQ) — single-checkpoint
// multi-precision weight representation via iterated 2-bit RTN residual
// corrections (Luo et al. Intel, arXiv:2608.04048 Aug 2026; Research 467).
// W̃(t) = Ŵ0 + Σ residuals — base + N stages, each 2-bit RTN with per-group
// f16 scale + zero-point. Default 1+3 → 2/4/6/8-bit prefixes. Pure modelless
// PTQ (no Hessian, no calibration). prefix_dot_into exploits matmul linearity.
// Verdict: Gain (not Super-GOAT — no concrete consumer today). Opt-in until a
// multi-precision LLM / per-NPC expert base / incremental-upgrade consumer lands.
#[cfg(feature = "rrq_quant")]
pub mod rrq_quant;
#[cfg(feature = "rrq_quant")]
pub use rrq_quant::{
    peak_to_mean_ratio, select_quant_strategy, BITS_PER_STAGE, CODES_PER_BYTE,
    DEFAULT_DIRECT_RTN_BITS, DEFAULT_GROUP_SIZE, DEFAULT_N_STAGES, KS_FLAG_THRESHOLD,
    LEVELS_PER_STAGE, PMR_THRESHOLD_2_2, QuantStrategy, RrqStage, RrqWeights,
};

// Selection-Set Fixpoint Propagation — KEEP M3 in house operator vocabulary
// (Issue 655 / Research 483, KEEP arXiv:2602.23592; HippoRAG PPR class). The
// one genuinely-unshipped composition from the Research 483 audit: a
// query-seeded importance propagation iterated until the top-r selected set
// stabilizes (membership fixpoint). Sigmoid-gated membership, CLR-reliability
// edge weighting, zero-alloc caller scratch, deterministic CSR order.
// Opt-in — promotion depends on the Issue 655 G1 head-to-head vs the shipped
// BFS-decay traversal (riir-rag fuse_graph_candidates) + downstream consumers.
#[cfg(feature = "selection_propagation")]
pub mod selection_propagation;
#[cfg(feature = "selection_propagation")]
pub use selection_propagation::{
    PropagationBlend, PropagationConfig, PropagationOutcome, SelectionPropagationScratch,
    propagate_selection_to_fixpoint_into,
};

// Sterling-derived modelless primitives (Issue 672 / Research 491,
// arXiv:2608.07594 Steerling-8B): ReLU-gated logit suppression (the naive
// subtraction promotes anti-aligned tokens), exact-decomposition readout
// (Σ parts + residual == fused, bit-identical by fixed summation order),
// lift-set steering targets (two-pass corpus statistic), γ=τ/peak logit-space
// calibration, + the HSIC-style cross-covariance gauge (measure-only). The
// noisy-OR rider lives UNGATED at the crate root (`noisy_or` / `noisy_or_stable`)
// because the riir-games-civ salience gate delegates to it under the DEFAULT
// feature set. Opt-in — promotion requires a consumer GOAT (riir-ai Issue 732,
// the exact-emotion-ledger NPC decision surface, is the first candidate).
#[cfg(feature = "sterling_primitives")]
pub mod sterling;

// Recirculation — cross-step residual mixture operator (Issue 673 Phase 1 /
// Research 492, arXiv:2608.17981 Mozer et al. DeepMind): leaks a convex,
// norm-matched mixture of the previous step's deep-layer state into a
// shallow destination layer at the NEXT input step. Sibling of RelocateOp
// (R417/Plan 431 — whose defend-wrong PoC refuted the overwrite semantics
// this mixture answers). Opt-in — promotion requires the Phase 2 defend-wrong
// PoC on gemma-2-2b (ppl reduction > 0 on ≥2 datasets AND strictly safer than
// the overwrite at equal layer pairs); default stays OFF until then.
#[cfg(feature = "recirculation")]
pub mod recirculation;

// Contrastive scope gate (Issue 674 / Research 493, arXiv:2608.13545
// LittleLearner): two-corpus log-odds table + Naive-Bayes log-LLR document
// scope score D(x) + epistemic haircut ĉ = c·sigmoid(−κ·D) + decline wiring
// + the paired OOS probe battery (Report-the-Floor extension). "A relevance
// check is not a scope check." Opt-in POC — per the issue's own T5 rule,
// promotion requires a consumer adoption (riir-clippy L4 2D gate / riir-ai
// engram gates); otherwise record negative and close.
#[cfg(feature = "contrastive_scope")]
pub mod contrastive_scope;

// Bounded-target correction + realization-gap triage primitives (Issue 695
// / Research 432, arXiv:2608.24646 DiffusionOPSD, Zhou et al.): the OPSD
// recipe's modelless half — one-measurement SPSA direction (unit by
// construction: with Rademacher Δ the normalized estimate collapses to
// sign(dq)·Δ/√D), bounded ±pairs/corrections with a type-level ‖Δ‖≤ε
// contract, the 5-eval ε-ladder with an honest monotone flag, the
// scorer-vitality canary, the multiplicative-step fixpoint + budget law
// (k > 3/η ⇒ re-anchor), and the ρ̂(k,η,ε) = (1−(1−η)^k)(1−cε²) realization
// model with FittingStarved/TargetStarved/OnModel triage. Consumers
// (riir-train Plan 360 T3.1, riir-clippy score-bench promised-vs-realized
// axis, riir-ai self-adaptive loops) file consumer-side at adoption.
// Zero-alloc (fixed [f32; 64] cap); c in ρ̂ is landscape-dependent —
// calibrate on frozen fixtures, DEFAULT_C is a prior. Opt-in POC.
#[cfg(feature = "bounded_target")]
pub mod bounded_target;
#[cfg(feature = "bounded_target")]
pub mod realization_gap;


// RVM modelless extraction (Issue 696 / Research 433, riir-train,
// arXiv:2608.23664 — anchored reward-weighted velocity regression, Choi et
// al.): the DT2 ANTI-COMMON-MODE scalar gate (peak-quantile statistic +
// context-scaled threshold + median subtraction over the hack-carrying
// population + band window — resists capture by the population-dominant
// degenerate mode; the CLR crowd-panic failure is the named consumer, T3
// PoC pending) and the ANCHORED SIGNED-REACH blend operator (out = anchor
// + A·(cand − anchor); five regimes clamp/blend/adopt/overshoot/repel;
// bit-identical pole fast paths at A ∈ {0,1}; A(r) schedules linear /
// 2σ(kr)−1 / (2r−1)/β̄). Both zero-alloc, modelless,
// sigmoid-not-softmax by construction. Opt-in POC — promotion only via the
// issue's consumer PoC gates (T3 headline; T4 operator A/Bs).
#[cfg(feature = "anti_common_mode")]
pub mod anti_common_mode;
#[cfg(feature = "anchored_reach")]
pub mod anchored_reach;

// Numeric-deviation contextualization probe (Issue 697 Phase 1 / Research
// 515, arXiv:2405.02803 "Is Flash Attention Stable?"): the f64
// mantissa-truncation format emulator + the two-surface DeviationReport
// (elementwise max_diff + 1-D Wasserstein delegated to mag::transfer's
// quantile-grid core) + the reference-band acceptance rule — R1 two-draw
// init divergence, R2 quantize→dequant round-trip labeled a SINGLE-STEP
// LOWER BOUND (doc-truth tripwired). The margin is an explicit caller
// parameter: the paper's 2–5× is context-specific, never a default. Scope
// limit: divergence similarity, NOT training stability (arXiv:2510.04212
// owns the mechanism). Zero-alloc *_into hot paths; no new deps; NaN
// rejected at the boundary. Opt-in — Phase 2 (perturbable attention lab) +
// T3.2/T3.3 open; first consumer: riir-ai gate layer; riir-train: Issue 492.
#[cfg(feature = "numeric_stability")]
pub mod numeric_stability;


// Kinematic rollout primitive (Plan 578 / Research 506, arXiv:2608.09926 —
// LDR, Li et al.): the modelless core of latent dynamics reasoning —
// finite-difference state (order ladder 0→3), O(1) Newton-backward closed-
// form k-step rollout exact on degree ≤ 3 motion (the provable ID-OOD gap ≡ 0
// strengthening of the paper's empirical ~20×), deterministic jerk/drag
// schedules, looming time-to-contact, regime predicates with sigmoid
// hysteresis, residual-surprise events (z/CUSUM/impulse + restitution),
// two-body closest approach/intercept, and the UQ-bearing extrapolation-
// horizon admission bound (RANK-ONLY verdict vs the conformal floor —
// .benchmarks/677). Pure f32 math, zero deps, zero allocs, #[repr(C)] POD
// state. DEFAULT-ON since 2026-08-26 (Bench 677 GOAT G1–G4 ALL PASS — the
// KARC precedent); consumer PoC: riir-ai Issue 757.
#[cfg(feature = "kinematic_rollout")]
pub mod kinematics;

// Stale-residual speculative layer pipelining — the modelless analysis half
// (Issue 691 / Research 508, arXiv:2608.23841 §6.3 Approach A+B — the
// paper's own UNTESTED hypothesis): residual-dominance ratios ‖δℓ‖/‖x_in‖
// + the paper's viability bar (>50% of layers, median ratio < 0.05), the
// accept/rollback threshold gate, the KL/top-1 SpecOutcome metrics + θ-sweep
// reduction, and the stream-ratio-aware (C+IO)/max(C,IO) overlap latency
// model (bits/weight-parameterized so ternary 1.58 b/w and Q4 4.6 b/w
// project from one code). Pure analysis over captured activations — the
// K3-0.40B simulator (which actually executes layers on stale residuals)
// lives in the root crate's `kimi_k3` module; Bonsai/Gemma trace producers
// in riir-ai feed `residual_dominance_from_trace`. Opt-in pending the Issue
// 691 verdict.
#[cfg(feature = "stale_residual")]
pub mod stale_residual;

// Conditioning-consistency audit (Issue 719 / Research 528, arXiv:2609.00865
// "MemoryWalker"): per-junction forward-KL between a compressed-conditioned
// (student) forward and a full-context (teacher) forward over the same decode
// positions + the unconditional Pinsker TV verdict `TV <= sqrt(eps_KL/2)` +
// the greedy-stream flip counter + the calibrated-zero (compression-off)
// control arm. Modelless pure-f32 arithmetic; the per-junction KL DELEGATES
// to `stale_residual::kl_logits` (substrate composition, no duplicate
// numeric core). Opt-in — NO live consumer (every shipped numeric-compression
// surface is gated stronger at bit-identity); T2 Gemma-4 ring / T3 packer /
// T4 H2O stay trigger-gated. No default promotion, no GOAT claim until a
// consumer exists.
#[cfg(feature = "cond_audit")]
pub mod cond_audit;

// TPR (Tensor Product Representation) binding algebra — the modelless rank-m
// generalization of the single-direction-vector latent ops (Issue 707,
// Research 527, arXiv:2608.29530 McCoy/Soulos/Linzen/Smolensky 2026). Four
// zero-alloc runtime ops (bind / unbind / constituent surgery / structural
// projection) over a frozen BLAKE3-committed artifact, fitted offline by
// closed-form ridge-ALS (no gradient descent anywhere). Ships its own GOAT
// instruments: the atomic-dictionary null, the withheld-pair OOD eval, the
// BoW structure router and BIC scheme selection. Sibling to the R299 Clifford
// wedge (`linalg::geometric_product`) and the R491/R389 steering families.
// Opt-in: the Issue 707 gates ALL PASS (Bench 698); promotion waits on a
// consumer that wants it on ITS default path (no-default-consumer rule).
#[cfg(feature = "tpr")]
pub mod tpr;

// Test-only `#[global_allocator]` so `alloc::tests::*` pass when running
// `cargo test -p katgpt-core --lib`. Downstream consumers (katgpt-rs root,
// riir-engine, etc.) install their OWN `#[global_allocator]`; this static is
// `cfg(test)` so it does not exist when katgpt-core is consumed as a library
// dep — no double-declare conflict. Mirrors the root crate's
// `static GLOBAL_ALLOC: TrackingAllocator` (src/lib.rs:356).
#[cfg(all(test, debug_assertions))]
#[global_allocator]
static TEST_GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;
