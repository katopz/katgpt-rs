# Benchmark 047: Plan 120 — ToaST Compression vs BPE + Rényi Entropy Metric

**Plan:** 120 — ToaST Tokenizer + Rényi Entropy
**Feature Gate:** `toast_tokenizer = []` (opt-in, NOT default-on)
**Date:** 2026-05-31

---

## Architecture

Token-Overlapping Adaptive Sparse Tokenizer (ToaST) is compared against BPE
using compression ratio and Rényi entropy metrics. Tests validate that ToaST
achieves competitive compression, produces no unknown tokens on ASCII, and
that Rényi entropy behaves correctly under known distributions.

```
ToaST tokenizer
 │
 ├──→ Encode: UTF-8 → token sequence  (no unknowns on ASCII)
 ├──→ Decode: token sequence → UTF-8  (roundtrip = identity)
 └──→ Compare vs BPE on compression ratio
      └──→ Rényi entropy: information-theoretic metric
           ├──→ Positive for both tokenizers
           ├──→ Monotonically non-decreasing with vocab size
           ├──→ Uniform dist → efficiency ≈ 1.0
           └──→ Single token → entropy = 0
```

---

## GOAT Proofs (8/8 ✅)

Test file: `tests/test_120_toast_renyi_goat.rs`

| # | Test | Assertion | Status |
|---|------|-----------|--------|
| T5a | `proof_toast_compression_vs_bpe` | ToaST token count ≤ BPE or ≤ byte count on test strings | ✅ |
| T5b | `proof_zero_unknown_tokens_ascii` | No `<unk>` tokens on plain ASCII text | ✅ |
| T5c | `proof_encode_decode_roundtrip` | encode→decode = identity for all test strings | ✅ |
| T6a | `proof_renyi_entropy_positive` | Rényi entropy > 0 for both tokenizers; ToaST ≤ byte count | ✅ |
| T6b | `proof_renyi_efficiency_monotonic` | Rényi efficiency non-decreasing with vocab size | ✅ |
| T6c | `proof_uniform_distribution_efficiency` | Uniform token distribution → Rényi efficiency ≈ 1.0 | ✅ |
| T6d | `proof_single_token_zero_entropy` | Single-token vocabulary → Rényi entropy = 0 | ✅ |

---

## Run

```bash
cargo test --features toast_tokenizer --test test_120_toast_renyi_goat -- --nocapture
```

---

## Status

✅ **GOAT 8/8 PASS**

---

## Module Structure

```
src/toast_tokenizer.rs               # ToaST tokenizer + Rényi entropy helpers
tests/test_120_toast_renyi_goat.rs   # 8 GOAT proofs
```

---

## Feature Gate

```toml
[features]
toast_tokenizer = []  # Plan 120, opt-in
```

No dependencies. Pure Rust.
