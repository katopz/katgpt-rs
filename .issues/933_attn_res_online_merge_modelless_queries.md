# Issue 933: attn_res Decode — Online-Softmax Merge, View Caching, Fixed-Query Probe

**Status:** Open — weight-free decode improvements + one modelless probe on the shipped AttnRes substrate (`katgpt-transformer/src/attn_res.rs`, Kimi-K3-0.40B consumer). **Consumer honesty (verdict round 1): the ONLY `apply_attn_res` consumer in the workspace is `src/kimi_k3/` — Kimi-K3-0.40B, which the owner classifies test-arch-only. Bonsai / qwen3.8 have no AttnRes blocks (grepped 2026-10-12: `attn_res` in `katgpt-rs/src/**` resolves to `kimi_k3/` only; the `dllm` `after_attn_res` field is a standard-residual activation name, not this mechanism). This issue therefore has NO production consumer today — it is substrate hygiene, LOWEST priority; it is independent of riir-train Plan 458 (whose long-context decode tests MAY consume it, but nothing is blocked on it).**

**Source:** Research 619 ([arXiv:2610.13630](https://arxiv.org/abs/2610.13630) MSAR §3.4 two-phase computation + Appendix C memory-access table).

## Context

`apply_attn_res` re-scores every stored block summary per sublayer call (O(entries·d) per call). Correct at N≈8, but the paper's two-phase strategy shows the batched-then-online-merge form exists and is weight-free. Two extractions + one probe:

## Tasks

- [ ] T1: `AttnResOnlineState` — running (max, sum, accumulator) across completed block summaries; a newly completed block merges in O(d); per-sublayer cost drops from re-softmax-over-all-entries to a constant-size merge plus the current partial. Weight-free; zero-alloc (extend the existing scratch pattern).
- [ ] T2: parity gate — tolerance-based, NOT bit-identity (different reduction order): max-drift ≤ 1e-6 vs `apply_attn_res` on the Kimi-K3-0.40B config fixture (the `attn_res_g1_spec_match.rs` pattern); G4 alloc gate; canary-armed loud-zero defense on the timing bench (timed_region_guard_gate class).
- [ ] T3: per-summary view cache — key-views (RMSNorm'd + folded score weights) change only when a summary lands; cache per summary, invalidate on `push`; stale-view canary (mutate a summary in-test, assert recompute). Ship with T1 or skip if T1's measured win makes recompute immaterial at N=8 — record the measurement either way.
- [ ] T4 (E5 probe, weakest-priority, LOWEST PRIORITY — substrate hygiene, see Status): fixed-query modelless AttnRes — deterministic exponential-depth profile replacing learned `score_weight` for checkpoints *without* AttnRes weights (Gemma-2-class). Frozen-read ppl on a fixed corpus vs the shipped arm; no training. The MSAR conv-removal ablation (source diversity retains most value under query degradation) motivates it; adversarial prior: layers may need task-specific lookback fixed queries cannot express. GREEN = ppl within tolerance AND no benchmark regression → new opt-in posture; RED = record and close (the learned queries ARE the mechanism for real checkpoints).
- [ ] T5: bench — decode p50/p99 at N=8 and N=32 block entries (long-context shape), CARGO_TARGET_DIR isolated, box state recorded per house rule.

## Notes

- Bench-target guard: this issue EDITS no bench file's numbers; new benches only, pinned rows re-measured via `scripts/repin_test_gate.sh` if any count moves.
- If T4 goes GREEN it composes with Plan 458's Stage 0 (a fourth posture: modelless-AttnRes on non-AttnRes checkpoints) — but do not block T4 on the plan.
