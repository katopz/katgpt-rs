# Plan 292: Future Probe Controlled Generation (FPCG) — Sentence-Level Behavior Steering

**Date:** 2026-06-18
**Research:** [`.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md`](../.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md)
**Source paper:** [openreview 48NnVTsirb](https://openreview.net/forum?id=48NnVTsirb) — Kortukov et al., NeurIPS 2026 / Mech Interp Workshop at ICML 2026
**Reference impl:** <https://github.com/kortukov/future_probes>
**Target:** `katgpt-rs/src/pruners/future_probe.rs` (new module) + `katgpt-rs/src/pruners/feature_class.rs` (vocabulary tag) + Cargo features `future_probe`, `fpcg_selector`
**Status:** Active — Phase 1 ✓ / Phase 2 ✓ / Phase 3 ✓ / Phase 4 partial (G5 ✓ / G6 partial / G7 ✓ measured in pure Rust; G1–G4 blocked on offline training — see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md)) / Phase 5 deferred

---

## Goal

Ship the **detection-vs-prediction vocabulary** in the trait stack and a **`FutureBehaviorProbe` primitive** + **`SentenceCandidateSelector`** that implement the FPCG algorithm from Research 267: sample M candidate next-utterance-spans, score each by a sigmoid probe on the mid-layer residual stream, select argmax/min. The probe direction vector is a **frozen, BLAKE3-committed artifact** loaded at init (no online training, freeze/thaw-compatible per `AGENTS.md` rules).

**GOAT gate:** the headline metric is **perplexity-vs-steering-strength Pareto frontier** vs our existing detection-side primitives (`EmotionDirections` Plan 162, CNA Plan 087). FPCG must dominate on at least one behavior class (same or better steering Δpp, strict perplexity win) to promote the selector to default-on. If GOAT fails on quality: demote selector to opt-in, keep only the vocabulary tag (Phase 1) as the always-shippable output.

**Why this is a GOAT not a Super-GOAT:** every component (linear projection, candidate-sample-score-select, calibrated linear forecast) has a strong cousin already shipped. FPCG combines them at a new abstraction layer (sentence-atomic future-behavior probe). See Research 267 §3 for the full novelty gate.

---

## Constraints (non-negotiable)

1. **Modelless.** Probe direction vectors are frozen artifacts. No gradient updates, no backprop. Offline training (if needed for benchmark) lives in `riir-train` or as a one-off Python script — never in `katgpt-rs`.
2. **Latent-to-latent with sigmoid.** Probe readout is `σ(w · act + b)`, never softmax. The scalar probability is the only thing that crosses to the selector.
3. **Freeze/thaw compatible.** Direction vector is a versioned, BLAKE3-hashed artifact reloadable at runtime via atomic swap. Pattern matches `LoRAWeightVersion` / snapshot conventions.
4. **Zero-alloc hot path.** The probe `forecast()` is a single `simd_dot_f32` over `d_model` + one sigmoid. Reuse the `EmotionDirections::project` template — it's the canonical pattern.
5. **Read-only at the LLM level.** FPCG **never modifies the residual stream**. The intervention is at the sample selector, not the activation. This is the *whole point* of FPCG (perplexity preservation) — breaking this rule voids the GOAT gate.
6. **3-repo discipline.** Generic math (probe, selector, vocabulary tag) → `katgpt-rs`. Game-side NPC dialogue steering → `riir-ai` (deferred, post-GOAT). No game semantics leak to the engine.

---

## Phase 1 — Vocabulary in the Trait Stack (CORE, ALWAYS SHIPS)

The smallest, highest-value output of this plan. The detection-vs-prediction distinction lets us tag every activation-reading primitive with whether it's safe to *steer on* (prediction-side) vs only *monitor with* (detection-side).

### Tasks

- [x] **T1.1** Create `src/pruners/feature_class.rs` with `FeatureClass` enum:

  ```rust
  /// Tags how a primitive reads model activations.
  ///
  /// Detection: reads features that *already* describe behavior in the
  ///   generated text (e.g. emotion vectors extracted from contrastive
  ///   final-answer pairs, CNA circuits). Safe for *monitoring* and
  ///   *intervention that mutates behavior downstream of the read*.
  ///   Risky to use as a *direct steering target* — pushing activations
  ///   along detection directions pushes the model off-manifold.
  ///
  /// Prediction: reads features that *forecast* future behavior probability
  ///   from intermediate reasoning state (e.g. FPCG's future probe).
  ///   Safe as a *non-invasive steering target* via candidate selection.
  ///
  /// See Research 267 §1.1 for the empirical distinction.
  #[repr(u8)]
  #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
  pub enum FeatureClass {
      Detection = 0,
      Prediction = 1,
  }
  ```

- [x] **T1.2** Add `fn feature_class(&self) -> FeatureClass` to `ScreeningPruner` trait with default `FeatureClass::Detection` (non-breaking). All existing pruners inherit Detection by default — no migration.
- [x] **T1.3** Add `FeatureClass::Detection` annotation to `EmotionDirections`, CNA's pruner, `FaithfulnessProbe`, `RegimeTransition`. (Document-only — they already default to Detection; this makes it explicit.)
- [x] **T1.4** Add a doc cross-reference in `crates/katgpt-core/src/traits.rs` pointing to Research 267 for the rationale.
- [x] **T1.5** Unit test: assert `EmotionDirections::feature_class() == Detection`. Assert that a new `FutureBehaviorProbe` (Phase 2) returns `Prediction`. Assert the default impl returns `Detection`.
- [x] **T1.6** Add `feature_class` field to `ReviewMetrics` for telemetry (count of detection-side vs prediction-side reads per session).

**Phase 1 ships independently.** Even if Phase 2–5 fail GOAT, the vocabulary tag is the durable architectural output. Mark Phase 1 complete when the trait compiles, all existing pruners still work, and the docstring is in place.

---

## Phase 2 — FutureBehaviorProbe Primitive

The forecast-side primitive. Single new file: `src/pruners/future_probe.rs`. Mirror `src/pruners/emotion_vector.rs` for the projection pattern.

### Tasks

- [x] **T2.1** Define types in `src/pruners/future_probe.rs`:

  ```rust
  /// Frozen direction vector for forecasting future behavior probability.
  ///
  /// The direction `w_B` is trained offline (typically via logistic regression
  /// on (mid-layer activation, future-behavior-probability) pairs gathered by
  /// resampling, following Kortukov et al. 2026). The artifact is BLAKE3-hashed
  /// and atomic-swappable at runtime (freeze/thaw compatible).
  pub struct FutureBehaviorProbe {
      /// Direction vector [d_model]. Project via simd_dot_f32.
      direction: Vec<f32>,
      /// Bias term.
      bias: f32,
      /// BLAKE3 hash of `direction` + `bias` bytes (commitment).
      artifact_hash: [u8; 32],
      /// Layer index this probe was trained against.
      layer: usize,
      /// Behavior label (free-form short string, e.g. "refusal", "aggression").
      behavior: Box<str>,
  }
  
  /// A read of the future-behavior probability. The ONLY thing that crosses
  /// the primitive boundary — a scalar in [0, 1]. Never the activation or the
  /// direction vector.
  #[derive(Copy, Clone, Debug, Default, PartialEq)]
  pub struct BehaviorForecast {
      /// σ(w · act + b) — probability the model will exhibit behavior B.
      pub probability: f32,
  }
  ```

- [x] **T2.2** Implement `FutureBehaviorProbe`:

  ```rust
  impl FutureBehaviorProbe {
      pub fn new(direction: Vec<f32>, bias: f32, layer: usize, behavior: impl Into<Box<str>>) -> Self;
      
      /// O(d) forecast via simd_dot_f32 + sigmoid. Zero-allocation, #[inline(always)].
      /// `activation` is the residual stream at `self.layer` at the sentence-end token.
      #[inline(always)]
      pub fn forecast(&self, activation: &[f32]) -> BehaviorForecast;
      
      /// BLAKE3 hash for freeze/thaw commitment checks.
      pub fn artifact_hash(&self) -> [u8; 32];
      
      /// Atomic swap of the direction vector. Lock-free read path via ArcSwap.
      /// Used for runtime hot-swap of probe versions.
      pub fn swap_direction(&self, new: FutureBehaviorProbe);
  }
  ```

- [x] **T2.3** Implement `ScreeningPruner` for `FutureBehaviorProbe`. `feature_class()` returns `Prediction`. `relevance()` returns the forecast probability (so it composes with the rest of the screening stack).
- [x] **T2.4** Add loader: `FutureBehaviorProbe::load_from_safetensors(path)` and `load_from_bytes(&[u8])`. Both compute BLAKE3 on load and refuse to serve if hash mismatches a manifest entry.
- [x] **T2.5** Add feature flag `future_probe` in root `Cargo.toml`. No default-on — opt-in until Phase 4 GOAT gate passes.
- [x] **T2.6** Unit tests:
  - zero direction → forecast = σ(bias) (deterministic)
  - orthogonal direction → forecast = σ(bias) (no signal)
  - aligned direction → forecast → 1.0
  - anti-aligned direction → forecast → 0.0
  - BLAKE3 hash stable across runs
  - swap is atomic (concurrent readers never see torn state)
- [x] **T2.7** Example `examples/future_probe_01_basic.rs`: load a synthetic probe (random direction), forecast on a synthetic activation, print the probability. Mirrors `examples/emotion_vector_demo.rs`.

---

## Phase 3 — SentenceCandidateSelector

The candidate-sampler + score + select loop. Mirrors the CGSP Conjecturer→Guide shape but at sentence granularity.

### Tasks

- [x] **T3.1** Define `SentenceCandidateSelector` trait (in `src/pruners/future_probe.rs` or a new `src/pruners/fpcg_selector.rs`):

  ```rust
  /// Generates M candidate next-utterance-spans for FPCG.
  ///
  /// Atomic unit: a span of generated text terminated by a sentence boundary
  /// (period, newline, EOS). This is the paper's CoT-understanding atomicity
  /// claim (Bogdan 2025, Macar 2026): decisions crystallize at sentence level.
  pub trait SentenceCandidateGenerator {
      /// Generate `n` candidate next-sentence spans given the current prefix.
      /// Implementations may use temperature sampling, beam search, or
      /// nucleus sampling — the selector is generator-agnostic.
      fn generate_candidates(
          &mut self,
          prefix: &str,
          n: usize,
          rng: &mut impl Rng,
      ) -> Vec<String>;
  }
  
  pub struct FpcgSelector<G: SentenceCandidateGenerator> {
      generator: G,
      probe: Arc<FutureBehaviorProbe>,
      /// Steering direction. Positive = maximize behavior, Negative = minimize.
      direction: SteeringDirection,
      /// Number of candidates per step (paper default: 10).
      num_candidates: usize,
  }
  
  #[repr(u8)]
  pub enum SteeringDirection { Positive = 0, Negative = 1 }
  ```

- [x] **T3.2** Implement `FpcgSelector::step(prefix) -> String`:
  1. Generate `num_candidates` next-sentence spans.
  2. For each: run forward pass up to `probe.layer`, extract activation at sentence-end position, call `probe.forecast()`.
  3. Return argmax (Positive) or argmin (Negative) candidate.
  - **Zero-alloc hot path**: pre-allocate `Vec::with_capacity(num_candidates)` once, `clear()` + reuse across steps.
- [x] **T3.3** Implement `FpcgSelector::run(prompt, max_sentences) -> String`: top-level loop calling `step` until EOS or `max_sentences` reached.
- [x] **T3.4** Add feature flag `fpcg_selector` depending on `future_probe`. Opt-in.
- [x] **T3.5** Provide a default `TemperatureCandidateGenerator` impl that wraps the existing model forward pass at temperature T=1.0 (matches paper setup). Other generators (nucleus, beam) can be added later.
- [x] **T3.6** Unit tests on a stub model:
  - selector with `num_candidates=1` is equivalent to unsteered generation
  - selector with `direction=Positive` picks the highest-probability candidate
  - selector with `direction=Negative` picks the lowest-probability candidate
  - selector terminates at EOS
  - hot-path is zero-alloc across 1000 steps (assert via `Vec::capacity` stable)
- [x] **T3.7** Example `examples/fpcg_01_basic.rs`: stub model producing synthetic candidates, selector picks based on a synthetic probe, prints the resulting trajectory.

---

## Phase 4 — GOAT Gate

Benchmark vs the existing detection-side primitives. The headline is the **perplexity-vs-steering-strength Pareto frontier**.

### Tasks

- [ ] **T4.1** Set up a small test corpus (synthetic behaviors + resampling labels, or reuse a small open-source prompt set — Refusal-style binary behaviors are simplest). Generate ground-truth behavior labels via the paper's resampling recipe (S=10 base × M=10 completion per sentence).
  - **BLOCKED on offline training, see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md).** Corpus prep is external data work, not engine code.
- [ ] **T4.2** Train a `FutureBehaviorProbe` direction vector offline (Python script in `scripts/train_future_probe.py` or `riir-train/`). Logistic regression on (mid-layer activation, behavior-probability label). Save as safetensors with BLAKE3 manifest.
  - **BLOCKED on offline training, see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md).** Per `AGENTS.md` constraint #1, offline training lives in `riir-train`, never in `katgpt-rs`.
- [ ] **T4.3** Run FPCG selector on test corpus. Record: behavior-fraction shift (pp), perplexity delta, format-filter rate, mean tokens generated.
  - **BLOCKED on T4.1 + T4.2 + a real-model `ActivationExtractor` wiring, see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md).** Feeds G1/G2/G3.
- [ ] **T4.4** Run baselines on same corpus:
  - Unsteered (control)
  - `EmotionDirections` desperation-style modulation (current detection-side)
  - CNA modulation (Plan 087, sparse-neuron steering)
  - **BLOCKED on T4.1 + T4.2 + engine wiring, see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md).** Feeds G4.
- [~] **T4.5** GOAT gate table (G1–G4 BLOCKED; G5/G6/G7 measured in pure Rust — see [`.benchmarks/292_fpcg_goat.md`](../.benchmarks/292_fpcg_goat.md)):

  | Gate | Target | How measured | Status |
  |---|---|---|---|
  | G1 Steering strength | FPCG achieves ≥ 30pp behavior shift on at least one behavior | behavior fraction Positive vs Negative | **BLOCKED** — see `.issues/032_fpcg_phase4_training_blocker.md` |
  | G2 Quality preservation | FPCG perplexity delta < 5% vs unsteered | mean PPL on test corpus | **BLOCKED** — see `.issues/032_fpcg_phase4_training_blocker.md` |
  | G3 Format integrity | FPCG format-filter rate < 10% | fraction of outputs failing regex/format check | **BLOCKED** — see `.issues/032_fpcg_phase4_training_blocker.md` |
  | G4 Pareto dominance | FPCG dominates EmotionDirections or CNA on at least one behavior class | plot PPL vs steering-Δpp | **BLOCKED** — see `.issues/032_fpcg_phase4_training_blocker.md` |
  | G5 Zero-alloc hot path | `Vec::capacity` stable across 1000 selector steps | instrumentation test | **PASS ✓** — capacity = 10 stable across 1000 steps; `tests/fpcg_goat_gate.rs::g5_*` + `pruners::fpcg_selector::tests::hot_path_is_zero_alloc_across_many_steps` |
  | G6 Latency | `forecast()` < 200ns per call (matches EmotionDirections) | `cargo bench` | **PASS ≤2048 / FAIL @4096 (absolute bar); PASS (relative) all sizes** — `forecast()` is 0.32–0.70× the cousin at every size (better-SIMD `simd_dot_f32`); d=64: 11.13ns … d=2048: 157.77ns PASS, d=4096: 309.54ns FAIL. `benches/fpcg_probe_forecast_bench.rs`. Real numbers in `.benchmarks/292_fpcg_goat.md`. |
  | G7 BLAKE3 commitment | Probe reload from tampered bytes refuses to serve | unit test | **PASS ✓** — clean load serves + reproduces forecast; tampered direction byte → `ProbeLoadError::HashMismatch`. `tests/fpcg_goat_gate.rs::g7_*` + `pruners::future_probe::tests::load_rejects_tampered_bytes`. |
- [x] **T4.6** Write benchmark report `katgpt-rs/.benchmarks/292_fpcg_goat.md` with the table, plots (PPL vs Δpp), and a clear PASS/FAIL verdict per gate.
  - Done as a **scaffold report**: G5/G6/G7 carry real measured numbers; G1–G4 rows marked BLOCKED with the methodology documented for the unblock session; Pareto plot deferred until G4 can run. No fabricated numbers.
- [x] **T4.7** If any of G1–G4 fails: file an issue in `.issues/` documenting the failure mode, do not promote to default. Continue to Phase 5 with the demotion branch.
  - Filed as [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md). G1–G4 are not *failed* (no negative result) — they are *blocked* on offline training. No promote/demote decision is possible until they run (Phase 5 deferred, not demoted).

---

## Phase 5 — Promotion / Demotion Decision

- [ ] **T5.1** If **G1+G2+G3+G4 all PASS**:
  - Promote `future_probe` feature flag to default-on (the probe primitive is zero-cost when unused — feature-gated only to keep the dependency graph clean).
  - Keep `fpcg_selector` opt-in (it costs M forward passes per step, not zero-cost).
  - Update `katgpt-rs/README.md` Feature Showcase with a new entry for FPCG.
  - Update `katgpt-rs/.docs/01_overview.md` Feature Flags table.
  - Cross-link Research 267 and Plan 292 in `katgpt-rs/.docs/` indices.
  - **DEFERRED — blocked on G1–G4, which are blocked on offline training (see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md)). Not executed this session.**
- [ ] **T5.2** If **G1 or G2 fails**:
  - Demote `future_probe` to opt-in permanently.
  - Keep Phase 1 (vocabulary tag) as the always-on shippable output.
  - Write a post-mortem in `.benchmarks/292_fpcg_goat.md` explaining why FPCG underperforms our existing detection-side primitives on this corpus. (Likely: our baseline is already read-only, so the perplexity-preservation advantage is muted.)
  - File follow-up issue for retry on a different behavior class (e.g., game-side NPC dialogue steering — but that's deferred to riir-ai).
  - **DEFERRED — blocked on G1/G2 running. Not executed this session.**
- [ ] **T5.3** If **G4 fails specifically** (FPCG works but doesn't dominate any baseline):
  - Keep both as opt-in.
  - Document FPCG as a *complementary* technique (paper's headline is complementarity, not dominance). Note in README: "use FPCG when activation steering breaks outputs, use Emotion Vector when raw strength matters."
  - **DEFERRED — blocked on G4 running. Not executed this session.**
- [ ] **T5.4** Commit. Use `feat:` prefix for shipping code, `docs:` for the research note + plan + benchmark report. Stay on `develop` branch (per `AGENTS.md`). Rebase non-interactive or merge fast-forward.
  - **NOT executed this session** — per task instructions, commits are left to the parent agent that validates the integrated state (parallel commits on `develop` are unsafe).

---

## Phase 6 — (Optional, post-GOAT) riir-ai Game-Side Integration

Deferred. Only kicks in if Phase 5 promotes. Sketch:

- [ ] **T6.1** Create `riir-ai/.research/140_Future_Probe_NPC_Dialogue_Steering_Guide.md` — private guide for NPC dialogue steering without voice breakage. Includes the latent-vs-raw boundary audit (probe stays latent; only scalar probability crosses).
- [ ] **T6.2** Create `riir-ai/.plans/3XX_npc_future_probe_dialogue.md` — runtime integration with `riir-engine` adapter stack. Per-NPC probe direction vectors versioned via freeze/thaw.
- [ ] **T6.3** Wire into `riir-games/src/npc/` dialogue generation path. Quest director can request "steer this NPC towards refusing the bribe" via FPCG selector.

**This phase does NOT execute in the current session.** It's a marker for follow-up.

---

## Dependencies

- Phase 1: none (pure trait change).
- Phase 2: `simd` primitives from `crates/katgpt-core/src/simd.rs` (simd_dot_f32), `blake3` crate (already in workspace), `papaya` or `arc-swap` for atomic direction swap (already in workspace).
- Phase 3: Phase 2 + a forward-pass interface (already exists in `inference_backend.rs` / `transformer_still.rs`).
- Phase 4: a small test model + corpus. Reuse the smallest config in `katgpt-rs/Cargo.toml` presets (`micro`).

---

## Risks

1. **Probe accuracy depends on training data quality.** If the resampling pipeline (Phase 4.1) produces noisy labels, the probe is useless. Mitigation: reuse Kortukov's open-source pipeline structure; verify on a behavior class with clear ground truth (refusal is binary and easy to label).
2. **Sentence-boundary detection is fiddly.** Different tokenizers split sentences differently. Mitigation: use the simplest possible splitter (period + newline + EOS) for v1; refine later.
3. **Per-candidate forward pass cost.** Paper uses `num_candidates=10` which is 10× the forward cost. For our plasma-tier hot path this is too slow. Mitigation: the GOAT gate measures latency; if `forecast()` is fast but the candidate generation is slow, FPCG is Warm-tier only (not Plasma). Document this honestly.
4. **"Detection-side already works for us" failure mode.** If our `EmotionDirections` already does the job read-only, FPCG's quality-preservation advantage is muted. The honest outcome then is: ship Phase 1 vocabulary tag, keep Phase 2–3 opt-in, document as complementary. This is a successful GOAT outcome, not a failure.

---

## Cross-references

- **Research:** [`.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md`](../.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md)
- **Canonical research-note example:** `.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md`
- **Canonical plan example:** `.plans/271_attention_matching_compaction.md`
- **Closest cousin plans:**
  - [`.plans/162_emotion_vector_inference_control.md`](162_emotion_vector_inference_control.md) — projection primitive (detection-side)
  - [`.plans/087_cna_contrastive_neuron_attribution.md`](087_cna_contrastive_neuron_attribution.md) — sparse circuit steering (detection-side)
  - [`.plans/239_posterior_guided_pruner_evolution.md`](239_posterior_guided_pruner_evolution.md) — precision vector (posterior-side)
  - [`.plans/274_curiosity_guided_self_play.md`](274_curiosity_guided_self_play.md) — sample-score-select skeleton
  - [`.plans/277_temporal_derivative_kernel.md`](277_temporal_derivative_kernel.md) — prediction-error channel
  - [`.plans/278_faithfulness_probe_modelless.md`](278_faithfulness_probe_modelless.md) — behavioral delta
- **Closest cousin issue:** [`.issues/023_adaptive_gamma_from_entropy_forecast.md`](../.issues/023_adaptive_gamma_from_entropy_forecast.md) — linear forecast from cheap signal (land first if not already shipped)

---

## TL;DR

Ship (1) the `FeatureClass::{Detection, Prediction}` vocabulary tag in the `ScreeningPruner` trait (Phase 1, always ships, durable architectural output), (2) a `FutureBehaviorProbe` primitive that mirrors `EmotionDirections` but returns a sigmoid future-behavior probability instead of a current-state projection (Phase 2), and (3) a `FpcgSelector` that samples M candidate next-sentences and picks argmax/min by probe score (Phase 3). GOAT gate (Phase 4) is the **perplexity-vs-steering-strength Pareto frontier** vs `EmotionDirections` and CNA. Promote if FPCG dominates on at least one behavior class (Phase 5); demote to opt-in and keep Phase 1 as the always-on output otherwise. **Latent-only**: probe operates on mid-layer residual; only the scalar probability crosses to the selector. **Freeze/thaw**: direction vector is BLAKE3-committed and atomic-swappable. **No residual-stream modification** — that's the entire point of FPCG. **riir-ai game-side integration deferred** pending GOAT proof.
