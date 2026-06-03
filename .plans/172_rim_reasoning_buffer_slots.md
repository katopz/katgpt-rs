# Plan 172: RiM Reasoning Buffer Slots — Fixed Latent Workspace for DDTree

> **Research:** 192 (Reasoning in Memory — Latent Workspace)
> **Status:** Complete
> **Feature gate:** `rim_slots`
> **Default-on:** YES (after GOAT proof, no perf hurt)
> **Depends on:** Plan 171 (FrozenBaseGuard — already done)

---

## Summary

Add fixed "reasoning buffer" token positions to the transformer prefill sequence. These positions act as latent workspace (from RiM paper) — they add zero decode cost, negligible attention cost, and provide the model with internal computation room. Combined with our existing FrozenBaseGuard (Plan 171), this creates zero-cost latent reasoning at intermediate LT2 loop steps.

---

## Tasks

### T1: Config Extension for Buffer Slots
- [x] Add `rim_block_count: usize` field to `Config` (default: 0, meaning disabled)
- [x] Add `rim_tokens_per_block: usize` field to `Config` (default: 2, matching paper's M=2)
- [x] Add `rim_buffer_token: usize` field to `Config` (default: `bos_token`, reused as buffer)
- [x] Add `rim_slot_positions(&self) -> Vec<usize>` method that returns buffer token indices
- [x] Ensure `rim_block_count == 0` means no buffer slots (backward compatible)

### T2: Prefill Integration — Append Buffer Tokens
- [x] In `forward()` / `forward_looped()`, when `rim_block_count > 0`, append K×M buffer token IDs to the input token sequence
- [x] Positions: sequential after last prompt token
- [x] No new token types needed — reuse existing token ID (BOS or padding)
- [x] Ensure KV cache handles the extended sequence length correctly

### T3: Logit Readout from Buffer End
- [x] In `forward()`, when buffer slots are active, read logits from the LAST buffer position instead of the last prompt token
- [x] This is the "readout" position — where the model has had K blocks of latent workspace
- [x] Ensure the logit index is correct: `n_prompt + K * M - 1`

### T4: FrozenBaseGuard Integration
- [x] When buffer slots are active in LT2 loop, ensure FrozenBaseGuard skips screening at intermediate loop steps
- [x] This is already default behavior — verify integration
- [x] Add test: buffer slots + FrozenBaseGuard + LT2 loop → final-step-only screening

### T5: GOAT Proof — Zero Decode Cost
- [x] Benchmark `forward()` with and without buffer positions (K=8, M=2)
- [x] Metric: TTFT difference < 1%
- [x] Add to `bench_108_lt2_looped.rs` as `proof_rim_slots_zero_decode_cost`

### T6: GOAT Proof — No Throughput Regression
- [x] Benchmark LT2 looped inference with and without buffer slots
- [x] Metric: throughput difference < 2%
- [x] Compare: K=0 (baseline), K=4, K=8

### T7: Feature Gate
- [x] Gate all buffer slot code behind `#[cfg(feature = "rim_slots")]`
- [x] Ensure no binary bloat when disabled
- [x] After GOAT proofs pass: set default-on (remove feature gate or make default)

---

## Key Design Decisions

1. **Reuse existing token ID** (BOS) for buffer positions — no vocabulary changes needed at inference
2. **Buffer count = 0 means disabled** — backward compatible, no behavior change for existing configs
3. **Readout at last buffer position** — matches paper's "final block readout"
4. **No custom attention mask** at inference — the standard causal mask works. Custom mask is only needed during training (model-based, riir-ai Plan for this)

---

## Alignment with optimization.md

- Fixed-size buffer positions: `Config` pre-computes, zero alloc in hot path
- Single forward pass: no loop, no decode steps for buffer positions
- Attention cost: O(K×M × seq_len) — negligible for K=8, M=2
- No allocation: buffer token IDs are a stack array `[usize; 16]`
- Feature gate: isolated from default path when disabled
