# Issue 032: FPCG Phase 4 — G1–G4 Training Blocker

> **Plan:** [katgpt-rs/.plans/292_future_probe_controlled_generation.md](../.plans/292_future_probe_controlled_generation.md)
> **Research:** [katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md](../.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md)
> **Benchmark report:** [katgpt-rs/.benchmarks/292_fpcg_goat.md](../.benchmarks/292_fpcg_goat.md)
> **Source paper:** [openreview 48NnVTsirb](https://openreview.net/forum?id=48NnVTsirb) — Kortukov et al., NeurIPS 2026
> **Date:** 2026-06-19
> **Status:** CLOSED (cross-repo hand-off — G1–G4 require trained probe + labeled corpus that live in riir-train, not the public modelless katgpt-rs engine)
> **Type:** Blocker / cross-repo hand-off (training lives in `riir-train`; corpus is external data)

**Closure rationale (2026-06-20):** Per `AGENTS.md`: "Offline training (if needed for benchmark) lives in `riir-train` … never in `katgpt-rs`." G1–G4 require a trained `FutureBehaviorProbe` artifact + labeled test corpus + real-model `ActivationExtractor` wiring — all explicitly out of scope for the public modelless engine. G5/G6/G7 (the pure-Rust gates) already PASS in `.benchmarks/292_fpcg_goat.md`. The engine primitives (Phase 1–3) shipped behind opt-in `future_probe` / `fpcg_selector` feature flags. Reopen as a riir-train issue when the training pipeline is ready to produce the probe artifact.

---

## Problem

Plan 292 Phases 1–3 (the `FeatureClass` vocabulary tag, the `FutureBehaviorProbe` primitive, and the `FpcgSelector` sample-score-select loop) shipped behind the opt-in `future_probe` / `fpcg_selector` feature flags. The Phase 4 GOAT gate has 7 sub-gates (G1–G7):

- **G5** (zero-alloc hot path), **G6** (`forecast()` latency), **G7** (BLAKE3 tamper refusal) are pure-Rust and **PASS** (G6 PASSES the absolute 200ns bar at d_model ≤ 2048 and beats its `EmotionDirections` cousin at every size; G7 enforces hash-check-on-load; G5 keeps `Vec::capacity` stable across 1000 steps). See `.benchmarks/292_fpcg_goat.md` for the real numbers.
- **G1** (≥30pp behavior shift), **G2** (PPL delta < 5%), **G3** (format-filter < 10%), **G4** (Pareto dominance vs `EmotionDirections` / CNA) **CANNOT run** without two prerequisites that are out of scope for the public `katgpt-rs` engine.

This issue tracks the blocker so the promote/demote decision (Plan 292 Phase 5) has a clear unblock path.

## What's blocked and why

| Task | Why blocked |
|------|-------------|
| **T4.1 — Test corpus with ground-truth behavior labels** | Requires the paper's resampling recipe (S=10 base × M=10 completions per sentence) to generate empirical future-behavior probabilities. This is external data preparation, not engine code. |
| **T4.2 — Trained `FutureBehaviorProbe` direction vector** | Offline logistic regression on (mid-layer activation, behavior-probability label) pairs. Per `AGENTS.md` constraint #1: "Offline training (if needed for benchmark) lives in `riir-train` … never in `katgpt-rs`." This is a Python/`riir-train` task. |
| **T4.3 — Run FPCG on the corpus** (feeds G1/G2/G3) | Depends on T4.1 + T4.2 + a real-model `ActivationExtractor` wiring (not currently in `katgpt-rs`; Phase 3 ships the trait + stub only). |
| **T4.4 — Run baselines** (feeds G4) | Same dependencies; needs `EmotionDirections` modulation and CNA modulation runnable on the same corpus + model. |
| **G1, G2, G3, G4** | Consequent on T4.3 / T4.4. |

## Why this is the correct scope boundary

`katgpt-rs` is the **public, modelless inference engine**. Per `AGENTS.md`:

- "Modelless. Probe direction vectors are frozen artifacts. No gradient updates, no backprop. Offline training (if needed for benchmark) lives in `riir-train` or as a one-off Python script — never in `katgpt-rs`."
- "3-repo discipline. Generic math (probe, selector, vocabulary tag) → `katgpt-rs`. Game-side NPC dialogue steering → `riir-ai` (deferred, post-GOAT)."

So fabricating G1–G4 numbers, or shipping a half-trained probe inside `katgpt-rs`, would violate the modelless constraint and the honesty rule. The correct move is to ship the engine primitives (done, Phases 1–3), prove the pure-Rust gates (done, G5/G6/G7), and defer the training-dependent gates to the repo that owns training.

## Unblock needs

1. **A trained `FutureBehaviorProbe` artifact** — logistic regression on a single mid-layer's residual-stream activation at the sentence-end position (the paper shows linear probes capture most of the signal; MLP adds little, Research 267 §1.3). Serialize via `FutureBehaviorProbe::save_to_bytes()` so the `FPPB` binary format + embedded BLAKE3 manifest hash are produced (G7 already verifies the hash on load).
2. **A BLAKE3 manifest** — already embedded in the `FPPB` format by `save_to_bytes()`; the loader (`load_from_bytes`) recomputes and refuses on mismatch. No extra manifest file strictly required, but a sidecar `.blake3` is conventional for distribution.
3. **A labeled test corpus** — Refusal is simplest (binary labels, large effect size; Plan 292 Risk #1 explicitly recommends it). Reuse the paper's open pipeline: <https://github.com/kortukov/future_probes> (`behavior_distribution_analysis.py` for resampling labels, `train_probe.py` for the logistic regression).
4. **An `ActivationExtractor` impl** backed by a real model forward pass — likely in `src/transformer.rs` / `src/inference_backend.rs`, exposing the residual stream at `probe.layer()` at the sentence-end token. This *is* in-scope for `katgpt-rs` once a model is available, but is engine wiring, not primitive work.

## Owner

- **Primary:** `riir-train` (offline training is its domain per `AGENTS.md`).
- **Alternative:** a one-off `scripts/train_future_probe.py` in `katgpt-rs/scripts/` that follows the paper's recipe (Plan 292 T4.2 explicitly allows this). Uses `uv` (per `AGENTS.md` Python rule), not `pip`.
- **Engine wiring** (`ActivationExtractor` over a real forward pass): `katgpt-rs`, once a target model is chosen.

## Acceptance

- [ ] Trained `FutureBehaviorProbe` artifact lands (safetensors or `FPPB` binary) with a documented behavior label + layer index.
- [ ] Behavior-labeled test corpus lands (or a documented path to reproduce it from the paper repo).
- [ ] `ActivationExtractor` wired to a real model forward pass in `katgpt-rs`.
- [ ] **Rerun Phase 4 G1–G4** per the methodology in `.benchmarks/292_fpcg_goat.md` (§Methodology).
- [ ] Fill the G1–G4 rows of the gate table in `.benchmarks/292_fpcg_goat.md` with real numbers.
- [ ] **Phase 5 promote/demote decision** per Plan 292 T5.1–T5.3:
  - G1+G2+G3+G4 all PASS → promote `future_probe` to default-on (selector stays opt-in).
  - G1 or G2 fails → demote permanently; keep Phase 1 vocabulary tag as always-on "fallback success".
  - G4 fails specifically → keep both opt-in, document as complementary (paper's headline is complementarity, not dominance).

## Cross-references

- **Plan 292:** `.plans/292_future_probe_controlled_generation.md` (Phase 4 T4.1–T4.7, Phase 5).
- **Research 267:** `.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md` (§1.3 probe accuracy, §1.4 FPCG algorithm, §1.5 vs activation steering).
- **Benchmark report:** `.benchmarks/292_fpcg_goat.md` (G5/G6/G7 real numbers + G1–G4 methodology).
- **Closest cousin issue:** `.issues/023_adaptive_gamma_from_entropy_forecast.md` — the linear-forecast-from-cheap-signal precedent (FPCG generalizes it from acceptance to behavior).
- **Reference implementation:** <https://github.com/kortukov/future_probes> (uv, Python).

## TL;DR

G1–G4 are blocked on a trained probe + labeled corpus, both of which are `riir-train` / external-data work and explicitly out of scope for the public modelless `katgpt-rs` engine. G5/G6/G7 are proven in pure Rust. Once a trained probe + corpus + engine `ActivationExtractor` wiring land, rerun G1–G4 and make the Phase 5 promote/demote call. Until then, `future_probe` and `fpcg_selector` stay opt-in and Phase 1 (`FeatureClass` vocabulary tag) remains the always-on shippable output.
