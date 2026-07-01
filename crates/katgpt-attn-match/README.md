# katgpt-attn-match

Attention Matching (AM) KV cache compaction — **modelless** (Plan 271,
arxiv 2602.16284). Extracted from the `katgpt-rs` root crate per Issue 359.

## What this is

Implements "Fast KV Compaction via Attention Matching" (Zweger, Fu, Guo, Kim —
MIT, ICML 2026). When compacting a KV cache `(K, V)` to `(Ck, β, Cv)` with
`t < T` tokens, preserves attention output and attention mass on a set of
reference queries — guaranteeing the compacted block's contribution under
concatenation with arbitrary future `(Kfixed, Vfixed)` is preserved.

Closed-form, no gradient descent:

1. Select compact keys `Ck` (HighestAttnKeys or OMP)
2. Fit `β` via NNLS (projected gradient descent)
3. Fit `Cv` via ordinary least squares (normal equations + Cholesky)

## Features

- `attn_match` — core compaction (rayon + serde). Self-contained, zero root deps.
- `still_kv` — RoPE-preserving chunked compaction via `katgpt-kv`'s `PositionFreeCompactor`.

## Scope note (Issue 359)

The `adaptive_cot` (AdaptiveTraceCompactor) glue stays in the `katgpt-rs` **root**
crate at `src/attn_match_adaptive_cot.rs` because it composes `freq_bandit` +
`trigger_gate` (root-only modules). This leaf owns the freq_bandit-free core.

## GOAT gate

Plan 271 GOAT 9/9 PASS — G1 β-recovery, G2 Cv reconstruction, G3 OMP residual,
G4 HighestAttn coverage, G5 reconstruction quality, G6 router determinism,
G7 no-alloc, G8 SIMD speedup, G9 (router).
