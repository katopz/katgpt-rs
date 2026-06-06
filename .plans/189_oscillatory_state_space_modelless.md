# Plan 189: Oscillatory State-Space Modelless Distillation

**Date:** 2026-06-05
**Source:** Research 169 — Oscillatory State-Space Modelless Distillation
**Status:** Active
**Feature Gates:** `freq_bandit` (default), `osc_kv` (opt-in), `modal_spec` (experimental)

---

## Summary

Distill OSSM-PINN's oscillatory state-space principles into katgpt-rs as modelless inference-time optimizations. Three fusions: FreqBandit (GOAT, default on), OscKV (conditional, opt-in), ModalSpec (experimental).

---

## Tasks

### Phase 1: FreqBandit — GOAT, Default On

- [x] Implement `FrequencyBandit` in `src/freq_bandit.rs`
  - Arms: {low_freq, mid_freq, high_freq} — pre-defined temporal frequency bands
  - Reward: acceptance_rate × latency_improvement from speculative decode
  - Uses existing `BanditPruner` infrastructure
  - Bandit state: per-domain frequency profile (sigmoid activation, not softmax per constraints)

- [x] Add FFT spectral analysis of recent token streams
  - `token_stream_spectrum(tokens: &[usize], window_size: usize) -> FrequencyProfile`
  - Pre-computed token embedding FFT for top-K modes
  - Low-cost: only analyze last N tokens (N=64 or 128)

- [x] Wire FreqBandit into speculative decode pipeline
  - `FrequencyBand::spec_config()` maps bands to SpecBandConfig
  - Low freq → larger draft tree, deeper lookahead
  - Mid freq → balanced draft tree
  - High freq → shallow draft tree, more verify iterations

- [x] Integrate FreqBandit with `InferenceRouter` + `TriggerGate`
  - FreqBandit recommendation feeds into tier routing decision
  - High-frequency queries → prefer GPU (faster verify)
  - Low-frequency queries → CPU acceptable (longer draft OK)

- [x] Feature gate: `freq_bandit` (default on)
  - Add to `Cargo.toml` features
  - Zero-cost when disabled: standard speculative decode

- [x] Tests: before/after speculative decode quality
  - Test cyclic input: repeated patterns (code loops, JSON arrays)
  - Test non-cyclic input: natural language prose
  - Expected: FreqBandit learns cyclic patterns → higher acceptance rate on cyclic input
  - Expected: No regression on non-cyclic input (bandit falls back to standard)

### Phase 2: OscKV — Conditional, Opt-In

- [ ] Implement `OscKVCache` struct in `src/osc_kv.rs`
  - `OscKVState { y: Vec<f32>, z: Vec<f32>, omega_sq: Vec<f32>, beta: Vec<f32> }`
  - IMEX discretization (symplectic, energy-preserving)
  - Bandit-learned ω from inference-time feedback

- [ ] Implement `QuantizedKVCache` trait for `OscKVCache`
  - `store_key`, `store_value` → update oscillatory state
  - `dequantize_key_into`, `dequantize_value_into` → reconstruct from oscillatory state

- [ ] Wire into `Config` as `OscKVCache` variant
  - Feature gate: `osc_kv` (opt-in, NOT default)
  - Only active when both `osc_kv` and `bandit` features enabled

- [ ] Benchmark: OscKV vs standard attention vs SpectralQuant
  - On cyclic sequences (code generation with loops)
  - On non-cyclic sequences (prose, dialogue)
  - Metric: per-token latency, quality (perplexity surrogate)

### Phase 3: ModalSpec — Experimental

- [ ] Implement LinOSS cell in `crates/katgpt-core/src/linoss.rs`
  - `LinOSSCell { omega_sq: [f32; H], beta: [f32; H] }`
  - `LinOSSState { y: [f32; H], z: [f32; H] }`
  - `imex_step(state, forcing, dt) -> LinOSSState`
  - `parallel_scan(initial, forcings, dt) -> Vec<LinOSSState>` (reuse HLA Blelloch scan)

- [ ] Pre-compute Fourier basis over vocabulary embedding space
  - `VocabFourierBasis { modes: [Vec<f32>; K] }` — top-K Fourier modes of vocab embeddings
  - Compute once at model load time

- [ ] Implement `ModalSpecDrafter`
  - Encode prompt context → LinOSS initial state
  - Parallel scan → modal coefficients over time
  - Reconstruct draft tokens from modal coefficients × vocab Fourier basis

- [ ] Feature gate: `modal_spec` (experimental, NOT default)
  - Only for research/development, not production

- [ ] Test: ModalSpec vs DDTree drafting quality
  - On structured output (JSON, code)
  - On unstructured output (prose)
  - Metric: draft acceptance rate, tokens/second

---

## GOAT Gate

Before any phase is marked default-on:

- [ ] Benchmark: no performance regression when feature enabled vs disabled on same commit
- [ ] Arena proof: at least one arena showing improvement (e.g., code generation latency)
- [ ] If GOAT: feature becomes default
- [ ] If not GOAT: feature stays opt-in, document why

---

## CPU/GPU Auto-Route

- [ ] FreqBandit feeds into `TriggerGate` for compute tier selection
- [ ] High-frequency decode → prefer GPU (faster verify iterations)
- [ ] Low-frequency decode → CPU acceptable (deeper draft tree)
- [ ] Automatic tier promotion/demotion based on FreqBandit arm selection history

---

## Constraints Checklist

- [x] Modelless first — inference-time only, no LLM training
- [x] LoRA only for training — N/A (no training needed)
- [x] Self-learning adaptive CoT — FreqBandit is self-learning at inference time
- [x] SOLID, DRY — reuses BanditPruner, InferenceRouter, TriggerGate
- [x] Tests/examples with before/after metrics
- [x] CPU/GPU auto-route via TriggerGate integration
