# Issue 926: MLA decode absorbs nothing — per-token W_UK/W_UV up-projection on the hot path

**Status:** OPEN — P1 optimization (kimi lane only — the 0.40B is test-arch per owner rule and league models don't run MLA); externally evidenced by arXiv:2610.07940 (Research 608); fix is modelless algebra, feature-flagged, GOAT-gated.

## Context

`crates/katgpt-attn/src/mla.rs` `mla_forward_token` Step 6 (the kimi_k3 MLA decode hot path) reconstructs keys and values for EVERY cached token at EVERY decode step, for EVERY head:

```rust
// IMPORTANT: in MLA decode, we do NOT cache k_c/v_c — we cache c_kv (the
// latent) and up-project at attention time. So for each cached token j, we
// must compute k_c_j = W_UK · c_kv_j and v_c_j = W_UV · c_kv_j.
//
// For Phase 2 (no weight absorption), we recompute per-token. This is O(seq)
// up-projections per decode step — correct but not optimal. Phase 6 may
// cache the up-projected k_c/v_c or use weight absorption.
```

The deferred "Phase 6" has an external price tag now: arXiv:2610.07940 (Hybrid Latent Attention for Looped Language Models, Chen et al., Oct 2026) measures the reconstruct-vs-absorb gap on latent caches directly — the reconstruct-based LLA runs at **7–18% of full-cache throughput**, the direct latent read at **28–59× LLA**, at EQUAL quality. Absorption beats reconstruction by more than quality ever will.

## The arithmetic (Kimi-K3-0.40B config: d_h=64, d_c=128, v_h=64, n_h=16)

Per decode step at seq tokens:
- **Reconstruct (current):** per head per token `W_UK[h]·c_kv_j` (8,192 FLOPs) + dot (128) + `W_UV[h]·c_kv_j` (8,192) + axpy (128) ≈ 16.6K FLOPs → seq=4096: **~1.09 GFLOPs/step** of up-projection work.
- **Absorbed:** per head per STEP `q_abs_h = W_UK[h]ᵀ·q_c_h` (8,192, once) + per token a d_c dot (128) + d_c axpy (128) + per head per step `o_h = W_UV[h]·s_h` (8,192, once) where `s_h = Σ_j w_j·c_kv_j` → seq=4096: **~0.017 GFLOPs/step (~60× fewer)**.

`o_h = W_UV·(Σ_j w_j c_kv_j)` is the value-side absorption: accumulate the softmax-weighted sum IN LATENT SPACE, up-project once. The k_r rope dot, the Kimi output gate (applies AFTER attention assembly — unchanged), and the `use_nope` structure are all untouched.

## Fix sketch

1. Feature flag `mla_absorbed` (opt-in first; promote on measured G2 gain per Feature Flag Discipline; reconstruct path STAYS as the spec-match reference + kill-switch).
2. Absorbed query: `q_abs_h = W_UK[h]ᵀ·q_c_h` once per step per head (scratch: one d_c buffer per head, or reuse the existing scratch pattern).
3. Scores: content part `q_abs_h·c_kv_j` (simd_dot over d_c) + unchanged rope part `q_r·k_r_j`.
4. Values: latent-space accumulator `s_h += w_j·c_kv_j` per head, then `o_h = W_UV[h]·s_h` once.
5. Same fix for the selected-block path in `dash_attn/flashmemory_sparse.rs::mla_forward_token_flashmemory` (same reconstruct shape; secondary scope).

**Scope law: BOTH halves are required for G2 to mean anything.** Absorbing only W_UK (scores) leaves the O(seq) W_UV up-projection in place — half the FLOPs stay and the bench under-reports. Steps 2–4 ship together or not at all.

**Rejected alternative — caching the up-projected k_c/v_c** (the comment's other Phase-6 option): it grows the cache to n_h·(d_h+v_h) per token ≈ the MHA cache (1,024 elems on 0.40B) that MLA exists to avoid, vs absorption's d_c (128) + k_r (32). Absorption keeps the 6.4× cache win AND removes the per-token work; caching trades one for the other. Recorded so a later session doesn't reopen it.

**Softmax note:** this kernel's attention normalization is SOFTMAX — the `mla_g1_spec_match.rs` contract and the model's own math require it. The house "sigmoid, never softmax" rule governs weighting/projection gates in latent space, NOT attention normalization; do not "fix" this kernel to sigmoid.

## Gates (GOAT)

- **G1:** existing `crates/katgpt-attn/tests/mla_g1_spec_match.rs` passes at its 1e-4 tolerance vs the reconstruct reference (fp reassociation only — no quality debate exists; the bar is the existing spec-match).
- **G2:** decode latency/tok-s vs seq {1K, 4K, 16K} on the reconstruct-vs-absorbed pair, `--release`, `black_box` both arms, `scripts/bench_preflight.sh` provenance line quoted. Promote to default only on measured gain.
- **G3:** no regression on the KDA lane, the trajectory benches (Bench 012/014/015/686 consumers of the kimi path), and every existing gate row.
- **G4:** alloc-free hot path (scratch reuse; no per-token allocation — the accumulator is one d_c vector per head).

## Caveats (named up front)

- The ~60× figure is FLOP arithmetic + the paper's analogous geometry, NOT a measured wall-clock on our box. If the step is bound elsewhere (kernel launch overhead on the 0.40B model; weight streaming), the wall-clock win shrinks — G2 decides, and a flat result is a valid outcome (keep opt-in, record the negative).
- League models (Bonsai-27B, qwen3.8-27B) do NOT run the MLA path — this moves the kimi_k3 lane (0.40B test/serving arch, R447 lane, trajectory benches), not league rows.
- fp reassociation changes low bits — any consumer pinning bit-exact MLA outputs beyond the 1e-4 spec bar must be enumerated before promotion (grep says: spec-match tests only).

## Refs

- Research 608 (distill + Path 0 inventory + panel merge)
- arXiv:2610.07940 §2 (MLA absorption lineage), §5.2 (7–18% / 28–59× measurements), §3.3 (direct latent read)
- DeepSeek-V2 (absorbed MLA origin); TransMLA arXiv:2502.07864
- Research 327/330 (MLA math + actual-model divergence — the served latent-attention prior art in-stack)
