# Issue 031: B-Posit Cross-Platform Deterministic Encoding (Deferred Optimization)

> **Opened:** 2026-06-18
> **Research:** [katgpt-rs/.research/265_b_posit_tapered_precision_format.md](../.research/265_b_posit_tapered_precision_format.md)
> **Source paper:** [arXiv:2603.01615v1](https://arxiv.org/abs/2603.01615) — Jonnalagadda, Thotli, Gustafson (2026)
> **Verdict at open time:** Gain (deferred) — no software-level win today, no hardware target.
> **Status:** ⏸ Parked — awaiting trigger

---

## Summary

The b-posit format (`⟨N, rS=6, eS⟩`) bounds the posit regime field to a fixed 5-layout menu, enabling constant-fan-in multiplexer decode with no subnormals. It would give us a **canonical, FTZ/DAZ-immune byte encoding** for cold-tier committed state and a **tapered-precision alternative** for sigmoid-projected scalars at the raw↔latent bridge. Not actionable today because (a) the paper's headline wins are silicon-side and we don't fab chips, (b) software emulation is slower than native f32 arithmetic on every platform we target, (c) we have no current consumer that needs subnormal-immune encoding beyond the existing FTZ-comments in `riir-gpu`.

## Why parked (not plan-created)

1. **No software speedup.** Software b-posit decode ≈ 50–200 ns/op (XOR + 5-case LUT + shift); native f32 is sub-ns. B-posit is *storage*, not *arithmetic*, in our stack.
2. **No hardware target.** No ANE / GPU / CPU we ship to supports posit instructions as of 2026-06. Adding a `WeightDtype::BPosit32` variant would have no consumer.
3. **Existing quant stack is strictly stronger for our hot path.** SpectralQuant (eigenbasis-adaptive bit allocation) dominates b-posit's *global* tapered precision on KV-cache compression (research 039, benchmark 013).

## Re-evaluation triggers (any one)

- [ ] **T1 — Hardware trigger.** A target backend we ship to (Apple ANE, NVIDIA, AMD, RISC-V vector extension, or a WebGPU shader pipeline) adds posit / b-posit instruction support. → Re-evaluate verdict upward; likely GOAT for that backend's storage format.
- [ ] **T2 — Fusion benchmark trigger.** SpectralQuant × b-posit fovea-allocation fusion shows measurable residual-cosine gain over pure SpectralQuant at matched bit budget in `tests/bench_spectralquant.rs`. → Promote to GOAT, create plan with feature flag.
- [ ] **T3 — Posit Standard trigger.** Posit Working Group ratifies b-posit into a standard revision, or a major framework (PyTorch / JAX / MLX) adds native b-posit dtype support. → Re-evaluate; ecosystem support changes the cost/benefit math.
- [ ] **T4 — Cross-platform drift bug trigger.** A real production bug surfaces from FTZ/DAZ disagreement between CPU (Rust f32) and GPU (riir-gpu cubecl subnormal-as-zero path documented in `gemv_q4k_cubecl.rs` / `attention_q8kv_cubecl.rs`). → B-posit becomes the canonical commitment format at the cold tier; create plan.

## Speculative follow-ups (do NOT start without trigger)

- **SpectralQuant × b-posit fovea-allocation.** Allocate per-eigen-direction precision tier: high-variance "outlier" directions stay on TurboQuant int8; low-variance "foveal" directions store as `⟨16,6,3⟩` b-posit. Hypothesis: at steep post-rotation spectra, beats pure SpectralQuant cosine at matched bits. Needs literature sweep first — is "tapered-precision-per-eigenvalue" already published?
- **Cold-tier canonical encoding.** Replace ad-hoc `f32::to_le_bytes()` commitments with `⟨32,6,5⟩` b-posit bytes for the 5 bridge scalars (`valence, arousal, desperation, calm, fear`) at the sync boundary. Gives bit-identical decode on any future hardware without FTZ/DAZ negotiation.
- **`WeightDtype::BPosit32` enum variant + GGUF extension.** Only if a model loader (llama.cpp, candle, mistral.rs) defines a b-posit tensor type.

## Related

- Research 039 (SpectralQuant), 065 (RotorQuant), 159 (KVarN), 200 (Quantization Outlier Collapse Security)
- Plan 179 (KVarN), 271 (Attention Matching compaction)
- Concrete FTZ/DAZ pain point: `riir-ai/crates/riir-gpu/src/gemv_q4k_cubecl.rs:62`, `riir-ai/crates/riir-gpu/src/attention_q8kv_cubecl.rs:65`
- `katgpt-rs/crates/katgpt-core/src/types.rs::WeightDtype` (where a variant would land if ever added)

## TL;DR

B-posit is a clean subnormal-immune tapered-precision format with no software-level speedup for us today and no hardware target. Park as Issue 031 with 4 concrete re-evaluation triggers (hardware support, fusion benchmark win, standard ratification, or a real FTZ/DAZ drift bug). Do not plan, do not extend `WeightDtype`, do not implement decode/encode — wait for a trigger.
