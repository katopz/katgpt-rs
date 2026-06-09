# Plan 238: MUX-Latent Context Compression

**Status:** 🟢 Phases 1-3 Complete, Phase 4 Partial, Phases 5-6 Remaining
**Date:** 2026-06-10
**Research:** `.research/211_LCLM_Latent_Context_Language_Model_Distillation.md`
**Feature Gate:** `mux_latent_context` (opt-in initially, promote to default after GOAT proof)
**Depends On:** Existing `mux_demux.rs` (MUX superposition), `domain_latent` (mid-layer injection), `MuxDdTree` (speculative decoding)
**GOAT Criteria:** TTFT reduction > 2× at 16k context with < 5% quality loss (perplexity)

---

## Summary

Implement inference-time context compression using MUX superposition as the encoder. No training required — MUX's vocabulary superposition compresses token spans into single latent tokens, injected at mid-layer via existing `domain_latent` infrastructure. This is the GOAT fusion from Research 211: LCLM's compression idea distilled into our existing MUX superposition pipeline.

Key insight: MUX superposition already produces position-weighted token combinations. We repurpose this as a lossless context compressor — each span of `span_size` tokens becomes one latent slot. The `domain_latent` mid-layer injection already exists for consuming latent representations. The result is fixed-size latent context that doesn't grow with input length.

---

## Architecture

```mermaid
graph TD
    Input[Input Tokens] --> Encoder[MUX Superposition Encoder]
    Encoder -->|span_size tokens per slot| Latent[latent_slots]
    Latent -->|mid-layer injection| DomainLatent[domain_latent]
    DomainLatent --> Decode[Standard Causal Attention + Latent Context]
    Decode --> KV[KV Cache standard]
    Decode --> LC[Latent Cache fixed-size no growth]
    Input -->|raw tokens| Raw[System Prompt + Instructions uncompressed]
    Raw --> KV
```

---

## Task

### Phase 1: Core MUX-Latent Encoder ✅

- [x] Create `src/mux_latent/` module directory
- [x] Implement `MuxLatentEncoder` struct that reuses existing `mux_demux.rs` superposition logic
  - Takes a span of tokens (span_size=16 default)
  - Produces one MUX latent token per span (position-weighted superposition)
  - Configurable compression ratio: 4x, 8x, 16x
- [x] Implement `MuxLatentConfig` with compression_ratio, span_size, injection_layer
- [x] Add feature gate `mux_latent_context` to Cargo.toml
- [x] Write unit tests for MUX latent encoding (encode roundtrip, compression ratio)

### Phase 2: Context Compression Pipeline ✅

- [x] Implement `LatentContextBuffer` — manages compressed context segments
  - Stores MUX latent tokens for compressed regions
  - Stores raw tokens for uncompressed regions (instructions, system prompt)
  - Supports segment-level compression with configurable boundaries
- [x] Implement `compress_context(tokens, config) -> CompressedContext`
  - Segments input into windows (configurable window_size, default 1024)
  - Compresses each window via MUX-Latent encoder
  - Returns compressed representation with segment metadata
- [x] Implement `decompress_segment(compressed, segment_id) -> Vec<Token>` (EXPAND analog)
  - MUX lossless guarantee means we can recover original tokens
- [x] Integration test: compress 4k context → 256 latent tokens → verify recall

### Phase 3: Decoder-Side Injection — Partially Done

- [x] Wire `CompressedContext` into existing `domain_latent` injection point
  - `LatentPrefillAdapter` converts `CompressedContext` → `MixedPrefillSequence`
  - `PrefillEntry::Latent` entries for compressed spans, `PrefillEntry::Raw` for raw tokens
  - `latent_indices()` identifies which entries need mid-layer injection
- [x] Modify prefill path to handle mixed raw+latent context
  - Raw tokens: standard prefill
  - Latent tokens: single KV cache entry per span (not per original token)
  - `CompressionSummary` estimates KV savings and TTFT reduction
- [ ] Deep integration: modify `forward_prefill()` in `transformer.rs` to accept `MixedPrefillSequence`
  - Blocked on: needs `domain_latent` feature co-enabled, careful transformer.rs surgery
  - Current approach: adapter + sequence prepared, actual `forward_prefill` wiring deferred to benchmark phase

### Phase 4: Adaptive LOD (Opt-In) — Partially Done

- [ ] Feature gate `lclm_adaptive_lod`
- [x] Implement `SpectralLOD` that computes spectral energy per window
  - Uses token variance proxy (SIMD FFT integration deferred to Phase 4 completion)
  - High energy: low compression (keep more tokens)
  - Low energy: high compression (aggressive MUX superposition)
- [x] Implement `adaptive_compress(tokens, target_ratio) -> CompressedContext`
  - Global budget allocation across windows
  - Per-window compression ratio determined by spectral energy
  - Integrated into `LatentContextBuffer::new_adaptive`
- [ ] Benchmark: fixed vs adaptive LOD on RULER-style NIAH tasks

### Phase 5: GOAT Proof

- [ ] Benchmark: latency comparison with/without `mux_latent_context`
  - Measure TTFT reduction at 4k, 8k, 16k, 64k contexts
  - Measure memory reduction (KV cache size)
  - Measure quality (perplexity on held-out data)
- [ ] Benchmark: compression ratio sweep (4x, 8x, 16x) with quality metrics
- [ ] Benchmark: adaptive LOD vs fixed compression
- [ ] GOAT gate: promote to default if TTFT reduction > 2x at 16k with < 5% quality loss
- [ ] Write benchmark results to `.benchmarks/`

### Phase 6: Integration Tests + Examples

- [ ] Example: `mux_latent_compress` — compress a long prompt and generate from it
- [ ] Example: `mux_latent_expand` — compress then selectively expand segments
- [ ] Integration test: compress → decode → verify output matches uncompressed baseline
- [ ] Doc comments and README update

---

## Dependencies

| Dependency | Module | Usage |
|---|---|---|
| MUX superposition | `mux_demux.rs` | Position-weighted token combination |
| Domain latent | `domain_latent` | Mid-layer latent injection |
| MUX speculative | `MuxDdTree` | MUX-aware speculative decoding |
| Spectral SIMD | `spectralquant` | FFT for adaptive LOD (Phase 4) |

---

## Risks

| Risk | Mitigation |
|---|---|
| MUX superposition quality insufficient for context compression | Fallback to raw tokens per-segment, GOAT gate |
| `domain_latent` injection bottleneck | Benchmark injection overhead, may need multi-point injection |
| EXPAND decompression latency | Lazy decompression, cache expanded segments |
| Feature gate bloat | Keep behind `mux_latent_context` gate, promote only if GOAT passes |

---

## TL;DR

Inference-time context compression via MUX superposition encoder → `domain_latent` mid-layer injection. No training. Feature-gated `mux_latent_context`. GOAT proof required before default promotion. Six phases: encoder → compression pipeline → decoder injection → adaptive LOD → benchmarks → integration.
