# Plan 630: rls_depth_mem — Exact Recursive Least Squares Associative Memory

**Status:** Active — Phases 1+2 landed 2026-10-21 (Phase 3–4 open)
**Claimed:** 4090-idle-r630, epoch=1792517412, expires 2026-10-22T17:30:12+00:00
**Date:** 2026-10-20
**Research:** [katgpt-rs/.research/629_Depth_Memory_RLS_Depth_Mixing.md](../.research/629_Depth_Memory_RLS_Depth_Mixing.md)
**Source paper:** [arXiv:2610.18809](https://arxiv.org/abs/2610.18809) — "Depth Memory" (LDM: exact RLS across depth; Sherman–Morrison write, unwinding, windowed writes)
**Target:** `crates/katgpt-core/src/delta_mem/rls.rs` (new, beside `state.rs`) + Cargo feature `rls_depth_mem` (opt-in)
**Related:** Research 626 / Issue 949 (N/D gated summary — the first-order cousin), `hebbian_kernel_memory.rs` + `linalg/ridge_solve.rs` (batch apex), `set_admission.rs` (Sherman–Morrison kernel precedent), `gdn_tree_verify` (never-materialize precedent)

---

## Goal

Ship the exact-online rung of the associative-memory ladder the workspace already ships in approximation (`delta_mem` delta-rule) and batch (`hebbian_kernel_memory`) forms: a fixed-rank `RlsMemoryState` whose write is the paper's exact RLS update — `κ = Pk/(λ + kᵀPk)`, `u = v − Mk`, `M ← M + uκᵀ`, `P ← (P − κkᵀP)/λ` — plus state unwinding (`M_{l−1} = M_l − u_lκ_lᵀ`) and a windowed reader (`M_j·x = M_b·x + Σu_i(κ_iᵀx)`). The headline property, and the G1 gate that distinguishes this primitive from every heuristic in the family: **after N observations the online state provably equals the batch weighted-ridge solution** (`linalg::ridge_solve` over the same window). Opt-in until its GOAT gates pass; promotion to default requires G1–G4 pass AND a real consumer (the gates alone do not promote — Feature Flag Discipline).

## Phase 1 — Core state (G1-proof first)

### Tasks

- [x] **T1.1** `RlsMemoryState<const RANK>` (rank-8 default): fields `m: [[f32; RANK]; RANK]`, `p: [[f32; RANK]; RANK]`, config `alpha: f32 (1.0)`, `lambda: f32 (0.99)` (fixed schedule per Research 629 E6 — no learned scalars); `new()` → `M=0, P=αI`. **Scope note: square memory (`D_v = D_k = RANK`) is the intended shape — this matches the shipped `DeltaMemoryState` square convention; a rectangular `D_v ≠ D_k` variant is out of scope until a consumer asks.** Landed `crates/katgpt-core/src/delta_mem/rls.rs`.
- [x] **T1.2** `write(&mut self, key, value)` with caller-L2-normed inputs (mirror `DeltaMemoryState`'s contract + `FeatureHasher` convention): weighted key `k̃ = √ω·k` with ω=1 (fold out), gain `κ = P·k̃/(λ + k̃ᵀP·k̃)`, pseudo-value `u = v − M·k̃`, rank-1 updates `M += u·κᵀ`, and the **symmetric-by-construction P update** `P ← (P − κκᵀ·(λ + k̃ᵀPk̃))/λ` (algebraically identical to `(P − κk̃ᵀP)/λ` but the downdate numerator `Pk̃k̃ᵀP` is formed symmetrically — in f32 the naive form loses symmetry and, with forgetting, that is the textbook way RLS loses positive-definiteness). Zero-alloc; f32; sequential fold order pinned (Issue-940 determinism law — this state is local-only, so `simd_dot_f32` is allowed with the documented 1e-6 tolerance, but the fold order is fixed and documented either way). Landed with SEQUENTIAL folds (no SIMD — rank-8 dots make the P update the cost anyway; fixed index order, bit-reproducibility test-pinned). Denominator `λ + k̃ᵀPk̃ > 0` asserted loud (corrupted-P failure mode).
- [x] **T1.3** `read_into(&self, query, out)` = `M·q` (mirror `DeltaMemoryState::read_into` row shape).
- [x] **T1.4** **G1 batch-equivalence test — against the DECAYED-ridge objective RLS actually solves.** LANDED (`g1_batch_equivalence_decayed_ridge`): 4 (α, λ) regimes — (1.0, 0.99, N=1000), (1.0, 0.90, 400), (1.0, 0.999, 1000), (0.5, 1.0, 500) — f64 weighted Gram `Σ λ^{N−i}kᵢkᵢᵀ + (λ^N/α)I` + cov `Σ λ^{N−i}kᵢvᵢᵀ` through `ridge_solve_direct_f64` (Wᵀ layout transposed back). **Measured max|M_online − M_batch| = 9.388e-7** worst (α=1, λ=0.99, N=1000); pin 1e-3 (~10³× margin). SPD: `try_cholesky_f32` succeeds after EVERY step over the full N≥1e3 sweeps; denominator invariant asserted inside every write (impl-level, loud).
- [x] **T1.5** `unwind_step(&mut self, ctx: &RlsCtx)` — **M-only unwind**, `M ←= M − u·κᵀ` in place; `p_restore(&mut self, ctx)` re-installs P exactly from the saved triangle. **Round-trip measured (W=64, LIFO unwind): max|M_unwound − M₀| = 1.043e-7** (pin 1e-4); P bit-exact after `p_restore`; the stale-P hazard is test-PINNED (`assert_ne!(P, P₀)` after M-only unwind). `write_ctx` fills the journal record. **DEVIATION (stable Rust):** `p_prev_tri: [f32; RANK·(RANK+1)/2]` cannot be spelled — const-generic arithmetic in array lengths needs `generic_const_exprs`; `RlsCtx` carries `TRI` as a second const param with defaults `<RANK=8, TRI=36>` — byte-identical at the default rank, literal product at other ranks, mis-sized ctx fails loud at pack/unpack bounds.
- [x] **T1.6** Feature `rls_depth_mem` in `katgpt-core` Cargo.toml; module doc carries the axis law + BOTH honest limits. En-route: the linalg `cfg(any(...))` gate list gained `rls_depth_mem` at birth (the Issue-701-R1b rule — `--no-default-features --features rls_depth_mem` caught the E0432 in-session); count_features README totals 705→706 ×5 sites; feature catalog §155 added.

## Phase 2 — WindowedReader

### Tasks

- [x] **T2.1** `WindowedReader { base: &RlsMemoryState, corrections: Vec<(ctx, )> ≤ E }` with `read_into(&self, x, out)` = `base·x + Σ u_i(κ_iᵀx)`, corrections summed in **write order** (pinned). Landed in `rls.rs` — `WindowedReader<'a, RANK, TRI>::new(base, max)` (pre-reserved capacity), `push` (refuses loud past the ≤ E bound — the caller owns window-roll policy), `clear`/`len`/`is_empty`, `read_into` zero-alloc, sequential folds, write order = append order.
- [x] **T2.2** G1: windowed read == materialized-intermediate read — LANDED (`windowed_read_equals_materialized`): B=200 base writes, W=64 journaled writes, 32 randomized unit queries; **measured max|windowed − materialized| = 1.192e-7** (pin 1e-4); the ≤ E bound refusal is test-pinned (`window_reader_refuses_past_bound`) and the empty-window bit-identity to the base read (`window_reader_empty_equals_base`).
- [x] **T2.3** G2 microbench (`benches/bench_rls_depth_mem.rs`, `[[bench]]` row + `required-features = ["rls_depth_mem"]`, harness=false, xorshift-seeded — no global draw): windowed-traffic vs materializing E intermediate states, E=4, release. **Measured (4090 box, idle, CPU 5% / RAM 17.2/31.8 GiB, 2026-10-21): windowed 100.0 ns vs materialized 200.0 ns per E-stage sweep (median of 20k) — windowed 2.00×.** The per-stage materialization cost (1 write + clone RANK² + read RANK²) is the traffic the window avoids; per-single-stage-read the materialized read can win — the sweep is the window's use case. REPORT-ONLY — the Phase-3 T3.1 GOAT is the decision gate.

## Phase 3 — GOAT gate vs the shipped rung

### Tasks

- [ ] **T3.1** G2: rank-8 `write+read` p50/p99 vs `DeltaMemoryState::write`/`read_into` at matched rank, `--release`, box-state PROVENANCE line (bench_preflight; Issue-741 law). Bar: RLS within a small factor of delta-rule (it does strictly more work — the P update; the paper's model-scale accounting is ~1%, our microbench decides on this box). A decisive loss → record, keep opt-in, close the plan honestly.
- [ ] **T3.2** G3: default-features build byte-identical; all existing `delta_mem` tests untouched and green.
- [ ] **T3.3** G4: zero-alloc write/read/unwind with caller scratch; counting-allocator canary (house pattern).
- [ ] **T3.4** E4 ordering re-test on our workloads: RLS vs delta vs a plain outer-product write at fixed inputs — record which orderings replicate (honest negatives are findings, per Research 629 §2.2 E4).
- [ ] **T3.5** E5: `surprise(&self) -> f32` (‖u‖ after write) + composition sketch with the shipped `delta_mem` surprise gate; test-only until a consumer exists.

## Phase 4 — Consumer decision (promotion gate)

### Tasks

- [ ] **T4.1** Name a real consumer or close the plan: candidates from Research 629 §2.5/§2.6 (pipeline-stage memory — needs a measured retrieval gap; healer selection signal — overlaps shipped surfaces; GDN delta-roll journal lives in riir-infer `.issues/049`, separate surface). **No consumer with a measured gap → the primitive stays opt-in and the plan closes with the verdict recorded.** Solving a problem nobody has is not promotion.
- [ ] **T4.2** If GOAT passes AND a consumer lands: promotion decision + demote-the-loser check on the same slot (does `RlsMemoryState` replace `DeltaMemoryState` anywhere real, or do they serve different accuracy/latency points?).

## Non-goals (auditable discards — Research 629 §2.2)

- Groups G (−0.003 nats at paper scale — complexity without measured need).
- Learned λ/ω/g/γ (substituted with fixed schedules; the tuned values are not claimed).
- Any LLM-layer insertion into frozen GGUF serving models (training track owns that path — riir-train Plan 467).
