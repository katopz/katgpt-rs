# Benchmark 292: FPCG GOAT Gate — Perplexity vs Steering-Strength Pareto Frontier

**Plan:** [`katgpt-rs/.plans/292_future_probe_controlled_generation.md`](../.plans/292_future_probe_controlled_generation.md)
**Research:** [`katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md`](../.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md)
**Source paper:** [openreview 48NnVTsirb](https://openreview.net/forum?id=48NnVTsirb) — Kortukov et al., NeurIPS 2026
**Date:** 2026-06-19
**Status:** **Phase 4 IN PROGRESS — G5/G6/G7 measured in pure Rust; G1–G4 blocked on offline training (see [`.issues/032_fpcg_phase4_training_blocker.md`](../.issues/032_fpcg_phase4_training_blocker.md)). Phase 5 (promote/demote) deferred.**

---

## TL;DR

- **G5 (zero-alloc hot path): PASS.** `candidates_buf` capacity stable at **10** across 1000 selector steps. Measured by `tests/fpcg_goat_gate.rs::g5_zero_alloc_hot_path_across_1000_steps` and the Phase 3 unit test `pruners::fpcg_selector::tests::hot_path_is_zero_alloc_across_many_steps`.
- **G6 (forecast latency < 200ns, matches `EmotionDirections`): PASS at d_model ≤ 2048, FAIL at d_model = 4096 on the *absolute* 200ns bar.** Real numbers below. Critically, `forecast()` is **faster** than the `EmotionDirections::project()` cousin at every swept size (0.32–0.70×) because the probe's `simd_dot_f32` is better vectorized than the cousin's hand-rolled 4-wide loop — so the *relative* intent of the gate ("match the cousin") is met everywhere by a wide margin. The absolute 200ns bar is exceeded only at the 4096-dim residual width of the paper's 8B-class models.
- **G7 (BLAKE3 commitment tamper refusal): PASS.** Clean round-trip load serves (forecast probability reproduced bit-for-bit); tampered direction byte → `ProbeLoadError::HashMismatch`. Measured by `tests/fpcg_goat_gate.rs::g7_blake3_commitment_clean_loads_and_tamper_refusal` and the Phase 2 unit test `pruners::future_probe::tests::load_rejects_tampered_bytes`.
- **G1–G4: BLOCKED.** Require a *trained* `FutureBehaviorProbe` direction vector (offline logistic regression, T4.2) and a *behavior-labeled test corpus* (T4.1, paper resampling S=10 × M=10). Both are out of scope for the public `katgpt-rs` engine per `AGENTS.md` ("training lives in `riir-train`"). No numbers fabricated. See issue 032.
- **Phase 5 (promote/demote to default): DEFERRED** — conditional on G1–G4, which cannot run this session.

---

## GOAT Gate Table (Plan 292 T4.5)

| Gate | Target | Status | Measured value | Notes |
|------|--------|--------|----------------|-------|
| **G1** Steering strength | FPCG ≥ 30pp behavior shift on ≥1 behavior | **BLOCKED** | — | Needs trained probe (T4.2) + corpus (T4.1). Offline training lives in riir-train per `AGENTS.md`. See issue `.issues/032_fpcg_phase4_training_blocker.md`. |
| **G2** Quality preservation | FPCG PPL delta < 5% vs unsteered | **BLOCKED** | — | Same blocker as G1. Needs a real-model forward pass exposing mid-layer residual at sentence-end + a PPL evaluator. |
| **G3** Format integrity | FPCG format-filter rate < 10% | **BLOCKED** | — | Same blocker. Needs real-model generations + a format/regex checker. |
| **G4** Pareto dominance | FPCG dominates `EmotionDirections` OR CNA on ≥1 behavior class | **BLOCKED** | — | Needs G1+G2 numbers for FPCG *and* the detection-side baselines on the same corpus. |
| **G5** Zero-alloc hot path | `Vec::capacity` stable across 1000 selector steps | **PASS ✅** | capacity = **10** before and after 1000 steps (no growth) | `tests/fpcg_goat_gate.rs::g5_zero_alloc_hot_path_across_1000_steps`; also the Phase 3 unit test `pruners::fpcg_selector::tests::hot_path_is_zero_alloc_across_many_steps`. |
| **G6** Latency | `forecast()` < 200ns/call (matches `EmotionDirections`) | **PASS ≤2048 / FAIL @4096** (absolute bar); **PASS** (relative "match cousin") at all sizes | d=64: **11.13ns**; d=256: **26.65ns**; d=768: **66.72ns**; d=1024: **85.07ns**; d=2048: **157.77ns**; d=4096: **309.54ns**. Cousin `project()` at same sizes: 15.91 / 64.04 / 180.64 / 244.60 / 482.26 / 971.86 ns. forecast/project ratio 0.32–0.70×. | `benches/fpcg_probe_forecast_bench.rs`, release build, warmup=10⁴, timed=10⁶ iters. See §G6 detail. |
| **G7** BLAKE3 commitment | Probe reload from tampered bytes refuses to serve | **PASS ✅** | clean load → `Ok`, forecast reproduces original (Δ < 1e-6); tampered direction byte → `Err(HashMismatch)` | `tests/fpcg_goat_gate.rs::g7_blake3_commitment_clean_loads_and_tamper_refusal`; also Phase 2 unit test `pruners::future_probe::tests::load_rejects_tampered_bytes`. |

---

## G6 — Latency detail (real measurement, 2026-06-19)

Bench: `benches/fpcg_probe_forecast_bench.rs` (`harness = false`, `std::time::Instant` + `std::hint::black_box`; **criterion is not a katgpt-rs dev-dependency**, so this follows the repo bench convention established by `benches/faithfulness_probe_bench.rs`). Run: `cargo bench --bench fpcg_probe_forecast_bench --features future_probe`. Release build (SIMD requires optimizations per `.contexts/optimization.md`). Warm-up 10 000 iters, timed 1 000 000 iters per size, on the development host (macOS).

```
=== Plan 292 T4.5 G6 — FutureBehaviorProbe::forecast() latency ===
Target: < 200 ns/call (matches EmotionDirections). warmup=10000, timed=1000000 iters.

 d_model   forecast ns/it    project ns/it  probe_verdict          ratio
      64            11.13            15.91         PASS ✅          0.70×
     256            26.65            64.04         PASS ✅          0.42×
     768            66.72           180.64         PASS ✅          0.37×
    1024            85.07           244.60         PASS ✅          0.35×
    2048           157.77           482.26         PASS ✅          0.33×
    4096           309.54           971.86 FAIL ❌ (>200ns)          0.32×

Gate G6 (<200ns for forecast): FAIL at one or more sizes — see per-size rows
```

### Honest interpretation

1. **The probe is *faster* than its detection-side cousin at every dimension.** Despite `forecast()` doing strictly more work (RwLock read + Arc clone + `simd_dot_f32` + sigmoid) than `EmotionDirections::project` (a hand-rolled 4-wide chunked dot product), the probe wins by 1.4× (d=64) to 3.2× (d=4096). The reason: `simd_dot_f32` (from `katgpt-core::simd`) auto-vectorizes better than the cousin's manual unrolled loop. So the gate's *relative* intent — "match `EmotionDirections` latency" — is **satisfied everywhere**.
2. **The absolute 200ns bar holds up to d_model = 2048 and breaks at 4096.** 4096 is the residual width of the paper's target models (DeepSeek-R1-Distill-Llama-8B ≈ 4096, Qwen3-14B ≈ 5120). At 4096, `forecast()` is 309.54ns — over the bar, but still **3.1× faster than the cousin** at the same width (971.86ns).
3. **Verdict on G6:** the *absolute* 200ns gate is **PASS for small/mid models (d ≤ 2048) and FAIL for 8B-class residual widths (d = 4096)**. The *relative* "matches `EmotionDirections`" gate is **PASS at all sizes**. The promote/demote decision should weigh: (a) the probe dominates the cousin on latency at every realistic size, (b) the 200ns absolute bar was a proxy chosen when the probe was assumed ≈ cousin-cost — that assumption is false (probe is cheaper). No code change recommended; this is a reporting/verdict nuance, not a regression.
4. **No fabrication.** These are the actual numbers from the single bench run on 2026-06-19. They will vary by host (CPU, thermal state per `.contexts/optimization.md` "Don't compare across thermal states"); re-run on the target machine before any default-on promotion.

---

## Methodology — how each gate is measured once unblocked

This section documents the procedure so the next session (after the trained probe + corpus land) can run G1–G4 with no ambiguity. **None of G1–G4 were run this session.**

### Prerequisites (block all of G1–G4)

- **T4.1 — Test corpus.** A prompt set with binary-behavior ground-truth labels. Simplest per Plan 292 Risk #1: **refusal** (binary, large effect size). Generate labels via the paper's resampling recipe: **S=10 base responses per prompt**, each split into sentences, then **M=10 completions re-sampled per sentence prefix** to measure the empirical future-behavior probability `B̄(p_{i←r_{j:k}})`. Reference pipeline: <https://github.com/kortukov/future_probes> (`behavior_distribution_analysis.py`). Open behaviors to consider: refusal, prompt-injection, sycophancy (free-form) and myopia/wealth/survival (MCQ).
- **T4.2 — Trained probe.** Logistic regression on `(mid-layer residual-stream activation at sentence-end position, future-behavior-probability label)` pairs. Single layer (the paper shows linear probes capture most of the signal; MLP adds little). Save as the `FPPB` binary format via `FutureBehaviorProbe::save_to_bytes()` with the BLAKE3 manifest hash embedded (G7 already enforces this on load). **Lives in `riir-train`** or a one-off `scripts/train_future_probe.py` — never in `katgpt-rs` (modelless constraint).
- **Engine wiring.** `ActivationExtractor` impl backed by a real model forward pass (likely `src/transformer.rs` / `inference_backend.rs`), exposing the residual stream at `probe.layer()` at the sentence-end token. Not currently wired (Phase 3 ships the trait + a stub).

### G1 — Steering strength (≥ 30pp behavior shift)

1. Load the trained probe (T4.2) into an `FpcgSelector` with `num_candidates = 10` (paper default), `SteeringDirection::Positive`.
2. Run FPCG over the test corpus; classify each output's behavior (refusal classifier or regex for binary behaviors).
3. Compute `fraction_positive = #(behaviors exhibited) / #(prompts)`.
4. Repeat with `SteeringDirection::Negative`; compute `fraction_negative`.
5. **Δpp = (fraction_positive − fraction_negative) × 100.** Gate: **Δpp ≥ 30** on at least one behavior class.

### G2 — Quality preservation (PPL delta < 5% vs unsteered)

1. Compute mean perplexity of **unsteered** generations on the corpus (`num_candidates = 1`, or no selector).
2. Compute mean perplexity of **FPCG-steered** generations (same prompts, `num_candidates = 10`).
3. **Δppl% = (ppl_steered − ppl_unsteered) / ppl_unsteered × 100.** Gate: **|Δppl%| < 5**. (FPCG never modifies the residual stream, so this should be small by construction — the gate exists to catch implementation bugs and candidate-distribution drift.)

### G3 — Format integrity (format-filter rate < 10%)

1. Define a format checker (regex / parse) per behavior — e.g. for MCQ: output must contain a valid `(A)`–`(D)` answer; for refusal: output must be a coherent refusal sentence.
2. Run FPCG over the corpus; flag outputs failing the checker.
3. **format_filter_rate = #(failing) / #(outputs).** Gate: **< 10%**. (Paper §4.2: activation steering filters 10–100% of outputs at effective multipliers; FPCG filters <10% in nearly all settings — this is FPCG's headline quality win.)

### G4 — Pareto dominance (vs `EmotionDirections` and CNA)

1. Run G1+G2 for **three conditions** on the same corpus: (a) FPCG, (b) `EmotionDirections`-based modulation (detection-side, Plan 162), (c) CNA modulation (Plan 087).
2. For each condition × behavior, plot **(Δppl%, Δpp)** — perplexity cost (x) vs steering strength (y).
3. **Gate:** FPCG **dominates** at least one baseline on at least one behavior class — i.e. FPCG's point is up-and-to-the-left (more steering at less PPL cost, or equal PPL with strictly more steering). Plot to `katgpt-rs/.benchmarks/292_fpcg_pareto.png` (plotters is a workspace dep).
4. Note (Plan 292 T5.3): if FPCG works but doesn't dominate, the paper's headline is **complementarity**, not dominance — keep both opt-in and document as complementary.

---

## Pre-ship synthetic gates (already green)

These ran this session and are independent of the training blocker:

- **G5** (zero-alloc hot path): capacity = 10 stable across 1000 steps. `tests/fpcg_goat_gate.rs::g5_*` + `pruners::fpcg_selector::tests::hot_path_is_zero_alloc_across_many_steps`.
- **G6** (latency): real ns/iter above. `benches/fpcg_probe_forecast_bench.rs`.
- **G7** (BLAKE3 commitment): clean load serves, tampered load refuses. `tests/fpcg_goat_gate.rs::g7_*` + `pruners::future_probe::tests::load_rejects_tampered_bytes`.
- **Forecast contract** (Phase 2): zero direction → σ(bias); orthogonal → σ(bias); aligned → p > 0.99; anti-aligned → p < 0.01. `pruners::future_probe::tests::forecast_*`.
- **Selector correctness** (Phase 3): Positive picks highest-prob, Negative picks lowest, EOS terminates, `num_candidates=1` ≡ unsteered, direction flip, probe swap atomic. 12 tests in `pruners::fpcg_selector::tests`.

---

## Promotion / Demotion Decision (Plan 292 Phase 5)

**Status: DEFERRED — blocked on G1–G4.** Cannot promote or demote until the trained probe + corpus land and G1–G4 run.

Current feature-flag state (unchanged this session):

| Feature | Default | Reason |
|---------|---------|--------|
| `future_probe` | opt-in | Phase 4 quality gates (G1–G4) not yet run; G6 PASSES ≤2048 but FAILS @4096 on the absolute bar (though it beats the cousin everywhere). Phase 1 vocabulary tag (`FeatureClass`) ships always-on regardless. |
| `fpcg_selector` | opt-in | Depends on `future_probe`; opt-in until GOAT gate passes. (Selector also costs M forward passes per step — would stay opt-in even after a quality GOAT pass, per Plan 292 T5.1.) |
| `FeatureClass` enum + `ScreeningPruner::feature_class()` default | **always-on** (no feature gate) | Phase 1 ships independently; non-breaking trait addition with default `Detection`. This is the durable architectural output regardless of Phase 4 outcome. |

**Path to promotion** (per Plan 292 T5.1, unblocked once issue 032 resolves):
1. Train a probe on Refusal (binary, large effect — easiest).
2. Wire `ActivationExtractor` to a real forward pass.
3. Run G1–G4 per the methodology above.
4. If G1+G2+G3+G4 all PASS → promote `future_probe` to default-on (selector stays opt-in).
5. If G1 or G2 fails → demote permanently; keep Phase 1 vocabulary tag as the always-on "fallback success".
6. If G4 fails specifically → keep both opt-in, document as complementary (T5.3).

---

## What Shipped (Phases 1–3 + Phase 4 achievable subset)

| Component | Path | Tests |
|-----------|------|-------|
| `FeatureClass` enum + `ScreeningPruner::feature_class()` default | `crates/katgpt-core/src/traits.rs` | 4 |
| `feature_class.rs` re-export shim | `src/pruners/feature_class.rs` | (same 4) |
| `EmotionDirections::feature_class()` explicit override | `src/pruners/emotion_vector.rs` | inherited |
| `FutureBehaviorProbe` primitive (Phase 2) | `src/pruners/future_probe.rs` | 13 |
| `FpcgSelector` + traits + default generator (Phase 3) | `src/pruners/fpcg_selector.rs` | 12 |
| **Phase 4 G5+G7 integration gate (new)** | `tests/fpcg_goat_gate.rs` | 3 (G5, G7, feature-class sanity) |
| **Phase 4 G6 latency bench (new)** | `benches/fpcg_probe_forecast_bench.rs` | — (bench, not test) |
| Examples | `examples/future_probe_01_basic.rs`, `examples/fpcg_01_basic.rs` | — |

---

## Hand-off Checklist for G1–G4 (issue 032)

- [ ] Train a `FutureBehaviorProbe` on Refusal (binary, large effect). Output as `FPPB` via `save_to_bytes()`.
- [ ] Wire `ActivationExtractor` to a real model forward pass (`src/transformer.rs` / `inference_backend.rs`), exposing residual at `probe.layer()` at sentence-end.
- [ ] Acquire a behavior-labeled corpus (paper repo: <https://github.com/kortukov/future_probes>); generate S=10 × M=10 resampling labels.
- [ ] Run G1 (Δpp ≥ 30), G2 (|Δppl%| < 5), G3 (format-filter < 10%), G4 (Pareto vs `EmotionDirections` + CNA).
- [ ] Fill the gate table above with real numbers + Pareto plot.
- [ ] Re-run G6 on the target deployment host (numbers above are dev-host, 2026-06-19).
- [ ] Phase 5 promote/demote per the rules above.
