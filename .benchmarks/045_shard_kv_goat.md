# GOAT Benchmark: ShardKV — Plan 147

**Date:** 2026-05-26
**Status:** CONDITIONAL — not promoted to default

---

## Unit tests: 14/14 pass

- RoPE roundtrip, identity at pos=0, changes vector, various positions
- Hadamard roundtrip, pack/unpack roundtrip
- KV cache roundtrip, zero vector, compression ratio, sink/window exact roundtrip, multi-position, reset clears, multi-layer independence

---

## GOAT proofs: 8/8 pass (5 strict, 3 conditional)

### Proof 1: RoPE removal eigenvalue concentration (G1)
- d_eff(raw keys) = 5.90
- d_eff(no-RoPE keys) = 2.00
- ratio = 0.339 (target: < 0.7) — **PASS**

### Proof 2: K cosine similarity (G6)
- avg cos_k = 0.9880 at d=128, avg_bits_k=4.0 — **CONDITIONAL** (target 0.995, met minimum 0.985)

### Proof 3: V cosine similarity (G7)
- avg cos_v = 0.9407 at avg_bits_v=2.0 — **CONDITIONAL** (target 0.98, met minimum 0.93)

### Proof 4: Compression ratio (G5)
- 9.7× at d=128 (target ≥ 8×) — **PASS**

### Proof 5: Sink + window protection (G10)
- Sink positions (0..4): EXACT roundtrip
- Window positions (192..256): EXACT roundtrip
- Middle positions: COMPRESSED (lossy) — **PASS**

### Proof 6: Cross-method benchmark (THE KEY TEST)

Parameters: head_dim=64, n_keys=256, 3-bit equivalent

| Method | cos_k | cos_v | MSE_k | MSE_v | Compression |
|--------|-------|-------|-------|-------|-------------|
| ShardKV(K=4,V=2) | 0.9957 | 0.9416 | 0.002247 | 0.010489 | 9.0× |
| SpectralQuant(avg=3bit) | 0.9855 | 0.9847 | 0.007461 | 0.002856 | 9.1× |
| TurboQuant(K=3,V=3) | 0.9646 | 0.9834 | 0.018009 | 0.003066 | 9.1× |
| HybridOCTPQ(K=3,V=3) | 0.9862 | 0.9866 | 0.007123 | 0.002493 | 9.1× |

ShardKV does NOT beat all methods on combined fidelity:
- ⚠ SpectralQuant beats ShardKV (1.9703 vs 1.9373 combined)
- ⚠ TurboQuant beats ShardKV (1.9480 vs 1.9373 combined)
- ⚠ HybridOCTPQ beats ShardKV (1.9728 vs 1.9373 combined)

### Proof 7: Asymmetric vs symmetric allocation
- Asymmetric (K=4,V=2): cos_k=0.9956, cos_v=0.9410, combined=1.9366
- Symmetric (K=3,V=3): cos_k=0.9845, cos_v=0.9832, combined=1.9677
- Symmetric wins by 0.0310 combined fidelity
- NOTE: Asymmetric may still be justified by attention error amplification theory

---

## Verdict

**ShardKV is CONDITIONAL — not promoted to default.**

### Wins
- Best K fidelity of all methods (0.9957) — the RoPE-removal + PCA path works
- Compression meets 8× target (9.7× at d=128)
- Sink + window protection works exactly
- RoPE-removal insight validated (66% d_eff improvement)

### Losses
- V path too lossy (0.94 cosine — Hadamard + 2-bit uniform can't match OCTOPUS's 0.99)
- Combined fidelity worst of all methods (1.937 vs HybridOCTPQ 1.973)
- Not suitable for default — stays opt-in

### Recommendation
1. Feature gate `shard_kv` added, NOT in `default` or `full`
2. The RoPE-removal insight should be evaluated as a standalone enhancement to SpectralQuant (Phase 1 of the plan)
3. The V path needs rework — replace Hadamard+uniform with OCTOPUS triplet encoding to close the quality gap
4. Niche use case: long-context memory-bound workloads where K fidelity matters more than V

---

## Commands to reproduce

```bash
cargo test --features "shard_kv,spectral_quant,turboquant,hybrid_oct_pq" --test test_147_shard_kv_goat -- --nocapture
cargo test --features "shard_kv,spectral_quant,turboquant" --lib shard_kv -- --nocapture
```
