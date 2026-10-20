# Plan 630: rls_depth_mem — Exact Recursive Least Squares Associative Memory

**Status:** Active — Phase 1 not started
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

- [ ] **T1.1** `RlsMemoryState<const RANK>` (rank-8 default): fields `m: [[f32; RANK]; RANK]`, `p: [[f32; RANK]; RANK]`, config `alpha: f32 (1.0)`, `lambda: f32 (0.99)` (fixed schedule per Research 629 E6 — no learned scalars); `new()` → `M=0, P=αI`. **Scope note: square memory (`D_v = D_k = RANK`) is the intended shape — this matches the shipped `DeltaMemoryState` square convention; a rectangular `D_v ≠ D_k` variant is out of scope until a consumer asks.**
- [ ] **T1.2** `write(&mut self, key, value)` with caller-L2-normed inputs (mirror `DeltaMemoryState`'s contract + `FeatureHasher` convention): weighted key `k̃ = √ω·k` with ω=1 (fold out), gain `κ = P·k̃/(λ + k̃ᵀP·k̃)`, pseudo-value `u = v − M·k̃`, rank-1 updates `M += u·κᵀ`, and the **symmetric-by-construction P update** `P ← (P − κκᵀ·(λ + k̃ᵀPk̃))/λ` (algebraically identical to `(P − κk̃ᵀP)/λ` but the downdate numerator `Pk̃k̃ᵀP` is formed symmetrically — in f32 the naive form loses symmetry and, with forgetting, that is the textbook way RLS loses positive-definiteness). Zero-alloc; f32; sequential fold order pinned (Issue-940 determinism law — this state is local-only, so `simd_dot_f32` is allowed with the documented 1e-6 tolerance, but the fold order is fixed and documented either way).
- [ ] **T1.3** `read_into(&self, query, out)` = `M·q` (mirror `DeltaMemoryState::read_into` row shape).
- [ ] **T1.4** **G1 batch-equivalence test — against the DECAYED-ridge objective RLS actually solves.** With forgetting λ, the online state minimizes `Σᵢ λ^{N−i}·‖vᵢ − M·kᵢ‖² + (λ^N/α)·‖M‖²` — the ridge term DECAYS as `λ^N` (at λ=0.99, N=1e3: λ^N ≈ 4.3e-5), it is NOT a fixed 1/α. The test builds, in f64, the weighted Gram `gram_reg = Σᵢ λ^{N−i}·kᵢkᵢᵀ + (λ^N/α)·I` and cross-cov `cov = Σᵢ λ^{N−i}·vᵢkᵢᵀ`, then calls `linalg::ridge_solve_direct_f64` (the shipped signature takes precomputed `gram_reg` + `cov` — it has no per-row weighting; the test owns the weighting). Assert `M` matches (pinned f32 tolerance; report the measured max deviation). Plus: SPD assertion (Cholesky of P succeeds after every step, N ≥ 1e3 — meaningful only because T1.2 uses the symmetric update form, so the gate tests the ALGORITHM, not f32 accumulation order) and the `λ + k̃ᵀP·k̃ > 0` denominator invariant.
- [ ] **T1.5** `unwind_step(&mut self, ctx: &RlsCtx)` — **M-only unwind**, `M ←= M − u·κᵀ` in place (the paper's own backward pass does exactly this — and SAVES P rather than unwinding it: restoring P by inverse Sherman–Morrison requires dividing through by λ, is numerically unstable, and is deliberately NOT attempted here). `RlsCtx { u, κ, p_prev_tri: [f32; RANK*(RANK+1)/2] }` carries the previous P's symmetric triangle (36 f32 at rank 8 — cheaper than the state itself, the paper's own context shape); `p_restore(&mut self, ctx)` re-installs P exactly from the triangle when writes must RESUME after an unwind. **G1 round-trip**: W writes → W unwinds → (with `p_restore`) M and P both equal pre-write within a PINNED measured ε (report the number); M-only unwind without restore is documented as read-rollback-only — any write after it without `p_restore` uses a stale gain and is a caller bug the doc names.
- [ ] **T1.6** Feature `rls_depth_mem` in `katgpt-core` Cargo.toml; module doc carries the axis law (sequence needs scans → approximations; depth/loops/pipeline-stages are naturally sequential → exact is free) + BOTH honest limits side by side: the (u,κ) journal supports unwind and window-read but NOT re-derivation under a different λ (κ depends on λ; re-fit needs raw rows), and P restoration requires the saved triangle (`p_prev_tri`) — an M-only unwind is read-rollback-only.

## Phase 2 — WindowedReader

### Tasks

- [ ] **T2.1** `WindowedReader { base: &RlsMemoryState, corrections: Vec<(ctx, )> ≤ E }` with `read_into(&self, x, out)` = `base·x + Σ u_i(κ_iᵀx)`, corrections summed in **write order** (pinned).
- [ ] **T2.2** G1: windowed read == materialized-intermediate read (f32 tolerance pinned, order pinned) across randomized writes.
- [ ] **T2.3** G2 microbench (`benches/bench_rls_depth_mem.rs`): windowed-traffic vs materializing E intermediate states, E=4, `--release`.

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
