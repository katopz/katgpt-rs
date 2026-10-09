# Issue 926: MLA decode absorbs nothing — per-token W_UK/W_UV up-projection on the hot path

**Status:** LANDED 2026-10-08 (opt-in, primary commit `61d76d94b` + secondary `ea0c60e41`) — T1–T5 of the fix sketch DONE: `mla_absorbed` feature (leaf `mla_absorbed = ["mla_attention"]` + root forward), `mla_forward_token_absorbed` (absorbed q + latent value accumulation via the in-tree `simd_transpose_matvec_into` substrate), `mla_forward_token_dispatched` wired into the kimi decode lane with kill-switch `KATGPT_MLA_RECONSTRUCT=1`. The reference fn's shared steps moved to verbatim helpers (`mla_decode_prologue`/`_rope_key_and_cache`/`_epilogue`) so both paths cannot drift on the prefix. GOAT: G1 PASS (absorbed/recon ≤ 8.9e-8, tol 1e-4/1e-3 — 7 tests), G2 PASS-provisional (**23.9–54× wall-clock** over seq {1K,4K,16K}; ratio stable at 1K ±3%, absolutes load-contaminated by a sibling build — quiet-box re-pin pending before promotion), G3 PASS (217 root lib tests + reference/grad-check/layer_diff/phase6/bench-consumer gates green; default posture byte-identical), G4 PASS (0 allocs, 256 steady-state steps). Bench record: `.benchmarks/926_mla_absorbed_decode_goat.md`. NOT promoted — the quiet-box re-pin is the promotion gate. **Re-pin window CLOSED again 2026-10-08 ~19:27** (the Issue-920 census restarted at nice-19 on this box; its ~24 h run holds the memory-bandwidth precondition) — the re-pin runs in the next quiet window after the census lands (~2026-10-09 late evening), command unchanged (runbook below; rebuild via the same `CARGO_TARGET_DIR=/tmp/katgpt-926-root cargo bench …` line if `/tmp` was cleared). **Census EXITED 2026-10-10 ~05:15; window re-checked 05:20 (load 4.85, decaying) then CLOSED again 05:33** (sibling riir-refine `cargo test --all-features --tests` at 12 cores) — the flashmemory A/B is now a PRE-STAGED BENCH (`bench_926b_flashmemory_absorbed_decode` @ `09affb249`, required-features `mla_absorbed,flashmemory_sparse`), not a to-write: run BOTH benches at the next window (load < 2, no sibling rustc/cargo) with preflight provenance, then judge + promote + record. **Secondary scope LANDED same day (`ea0c60e41`)** — see checkbox 5. Open items: the flashmemory G2 paired A/B (pending quiet window, same discipline as the primary re-pin); the reconstruct paths' dead Step 3 (k_c overwritten before read, v_c never read — BOTH lanes now) noted as a future microcleanup, not done here.

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

- [x] 1. Feature flag `mla_absorbed` (opt-in first; promote on measured G2 gain per Feature Flag Discipline; reconstruct path STAYS as the spec-match reference + kill-switch). — Landed: leaf + root features, kill-switch `KATGPT_MLA_RECONSTRUCT=1` in the dispatched router.
- [x] 2. Absorbed query: `q_abs_h = W_UK[h]ᵀ·q_c_h` once per step per head (scratch: `q_abs` `[d_c·n_h]` pre-allocated in `MlaForwardScratch`, populated via the in-tree `simd_transpose_matvec_into` NEON/AVX2/wasm32 substrate).
- [x] 3. Scores: content part `q_abs_h·c_kv_j` (simd_dot over d_c) + unchanged rope part `q_r·k_r_j`.
- [x] 4. Values: latent-space accumulator `s_h += w_j·c_kv_j` per head (ONE transpose matvec over the `[seq × d_c]` cached latent matrix, weights applied after via the single `inv_sum` scaling), then `o_h = W_UV[h]·s_h` once.
- [x] 5. Same fix for the selected-block path in `dash_attn/flashmemory_sparse.rs::mla_forward_token_flashmemory` (same reconstruct shape; secondary scope).
  - **LANDED 2026-10-08 (`ea0c60e41`, opt-in, SAME `mla_absorbed` feature — no new flag).** Steps 1–6 moved verbatim into `flashmemory_decode_prefix` (shared by both paths → block selection definitionally identical), Step 8 into `flashmemory_decode_epilogue`; the reconstruct reference (`mla_forward_token_flashmemory_reconstruct`, pub, always compiled) keeps Step 7 byte-verbatim. Absorbed path: per head `q_abs = W_UKᵀ·q_c` once via `simd_transpose_matvec_into`; scores = latent dot + unchanged rope dot over selected tokens; values = latent accumulator zeroed once + ONE `simd_transpose_matvec_acc` PER CONTIGUOUS SELECTED BLOCK (blocks ascending+disjoint → each block's packed score slice aligns with its latent row run; a single dense matvec over all seq rows would re-read unselected tokens and defeat the sparse read), single `inv_sum` scaling, one `W_UV[h]·s_h`. Zero-alloc from the existing scratch. Kill-switch: `mla_forward_token_flashmemory` is now the dispatched router (all 6 bench consumers unchanged) — `KATGPT_MLA_RECONSTRUCT=1` forces reconstruct (read once via `OnceLock`), feature-off compiles the cfg block away → transparent forward. **G1**: `tests/mla_flashmemory_absorbed_spec_match.rs` 8/8 — pos=0 bit-identical, worst 8.94e-8 vs 1e-4/1e-3 (same envelope as the dense primary); **G4**: alloc check 0 allocs / 256 steps. `required-features` rows = `["mla_absorbed", "flashmemory_sparse"]` (the fn lives behind the lane feature — the green-zero rule). Gates: clippy green at 4 postures (default / mla_absorbed / +flashmemory_sparse / flashmemory_sparse alone), katgpt-attn lib 176/176 incl. 14 in-module flashmemory tests through the absorbed router, `mla_g1_spec_match` 8/8, root lib `--features mla_absorbed,kimi_k3` 217/217, all 6 flashmemory bench consumers compile under the feature. **Flashmemory G2 paired A/B PENDING** — same quiet-box discipline as the primary re-pin; NOT a promotion blocker for the dense lane.

**Scope law: BOTH halves are required for G2 to mean anything.** Absorbing only W_UK (scores) leaves the O(seq) W_UV up-projection in place — half the FLOPs stay and the bench under-reports. Steps 2–4 ship together or not at all.

**Rejected alternative — caching the up-projected k_c/v_c** (the comment's other Phase-6 option): it grows the cache to n_h·(d_h+v_h) per token ≈ the MHA cache (1,024 elems on 0.40B) that MLA exists to avoid, vs absorption's d_c (128) + k_r (32). Absorption keeps the 6.4× cache win AND removes the per-token work; caching trades one for the other. Recorded so a later session doesn't reopen it.

**Softmax note:** this kernel's attention normalization is SOFTMAX — the `mla_g1_spec_match.rs` contract and the model's own math require it. The house "sigmoid, never softmax" rule governs weighting/projection gates in latent space, NOT attention normalization; do not "fix" this kernel to sigmoid.

## Gates (GOAT)

- [x] **G1:** existing `crates/katgpt-attn/tests/mla_g1_spec_match.rs` passes at its 1e-4 tolerance vs the reconstruct reference (fp reassociation only — no quality debate exists; the bar is the existing spec-match). — Reference re-passes 8/8 (default AND mla_backward postures) + grad-check 4/4; NEW `tests/mla_absorbed_spec_match.rs` 7/7 (absorbed vs reconstruct ≤ 8.9e-8).
- [x] **G2:** decode latency/tok-s vs seq {1K, 4K, 16K} on the reconstruct-vs-absorbed pair, `--release`, `black_box` both arms, `scripts/bench_preflight.sh` provenance line quoted. Promote to default only on measured gain. — **23.9× (1K, stable ±3%) / 37–53× (4K) / 36–54× (16K)** across 4 runs; interleaved pairs, medians; box-state disclosed (sibling build load 14–42 during the run — absolutes provisional, ratio load-robust). `benches/bench_926_mla_absorbed_decode.rs`. Quiet-box re-pin pending before promotion (runbook in the bench record).
- [x] **G3:** no regression on the KDA lane, the trajectory benches (Bench 012/014/015/686 consumers of the kimi path), and every existing gate row. — 217 root lib tests under the feature; layer_diff + g4_alloc_free + phase6 green; bench_012 + bench_889 compile; default posture byte-identical (router = transparent forward); backward/training calls the reference directly, structurally unaffected.
- [x] **G4:** alloc-free hot path (scratch reuse; no per-token allocation — the accumulator is one d_c vector per head). — `tests/mla_absorbed_alloc_check.rs`: 256 steady-state steps, 0 allocations, counter canary green.

## Caveats (named up front)

- The ~60× figure is FLOP arithmetic + the paper's analogous geometry, NOT a measured wall-clock on our box. If the step is bound elsewhere (kernel launch overhead on the 0.40B model; weight streaming), the wall-clock win shrinks — G2 decides, and a flat result is a valid outcome (keep opt-in, record the negative).
- League models (Bonsai-27B, qwen3.8-27B) do NOT run the MLA path — this moves the kimi_k3 lane (0.40B test/serving arch, R447 lane, trajectory benches), not league rows.
- fp reassociation changes low bits — any consumer pinning bit-exact MLA outputs beyond the 1e-4 spec bar must be enumerated before promotion (grep says: spec-match tests only).

## Refs

- Research 608 (distill + Path 0 inventory + panel merge)
- arXiv:2610.07940 §2 (MLA absorption lineage), §5.2 (7–18% / 28–59× measurements), §3.3 (direct latent read)
- DeepSeek-V2 (absorbed MLA origin); TransMLA arXiv:2502.07864
- Research 327/330 (MLA math + actual-model divergence — the served latent-attention prior art in-stack)
