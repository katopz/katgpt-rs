# Plan 277: Temporal Derivative Kernel — Dual Fast/Slow Surprise Signal

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md)
**Source paper:** [arXiv:2606.08720](https://arxiv.org/abs/2606.08720) — O'Reilly, "This is how the Neocortex Learns" (Jun 2026)
**Target:** `crates/katgpt-core/src/temporal_deriv.rs` (new module) + fusion hooks into `sense/reconstruction.rs` (HLA), `DeltaMemoryState` (Plan 053), `CollapseDetector` (Plan 212), CGSP curiosity (Plan 274)
**Cargo feature:** `temporal_deriv` (opt-in until GOAT gate passes)
**Status:** Active — Phase 0 (not started)

---

## Goal

Ship a generic, zero-allocation, sigmoid-compatible **dual fast/slow temporal-derivative kernel** distilled from O'Reilly 2026. The kernel turns any streaming latent scalar (or fixed-size vector) into a signed "surprise" signal — the implicit prediction-error channel the neocortex uses for credit assignment, computed locally from a signal's own time series with no external target and no backprop.

**Why this matters (per Research 243):** every EMA currently in the codebase is a *single* integrator (`simd_fused_decay_write`, `AdaptiveTraceCompactor::ema_entropy`, `BreakevenTracker::update_ema`, `evolve_hla` itself). The dual `(I_fast − I_slow)` band-pass derivative is absent from shipped code. It is the smallest missing primitive that upgrades four existing pillars:

- **HLA companion** — `evolve_hla` tracks *what is*; derivative tracks *how fast it is changing*.
- **δ-Mem write gate** — currently writes on every query; derivative-gated writes happen only on surprising events.
- **Collapse detector fusion** — entropy collapse is one signal; prediction-derivative collapse is orthogonal.
- **Intrinsic curiosity** — `sigmoid(β · surprise_norm())` is a zero-cost curiosity signal that needs no Solver (unlike CGSP).

**GOAT gate:** Phases 2–5 each have a fusion gate (G2–G5). If any single fusion wins, that fusion promotes to default-on for its consumer. The kernel primitive itself (Phase 1) ships as opt-in `temporal_deriv` and only promotes to default if ≥2 fusions pass.

**Latent vs raw boundary:** operates on latent state; emits a bounded scalar (`surprise_norm`) that may sync as a raw summary statistic. Full N-dim derivative vector stays local per-entity.

---

## Phase 1 — Primitive Skeleton (CORE)

Target: `crates/katgpt-core/src/temporal_deriv.rs`. Generic, no game semantics, no consumer coupling.

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/temporal_deriv.rs` with `TemporalDerivativeKernel<const N: usize>` struct (fields: `fast: [f32; N]`, `slow: [f32; N]`, `alpha_fast: f32`, `alpha_slow: f32`).
  - `pub fn new(alpha_fast: f32, alpha_slow: f32) -> Self` — zero-init state, validate `0 < alpha_slow < alpha_fast <= 1` (panic in debug, clamp in release).
  - `pub fn with_initial(fast: [f32; N], slow: [f32; N], alpha_fast: f32, alpha_slow: f32) -> Self` — for warm starts / snapshot restore.
- [ ] **T1.2** Implement `pub fn observe(&mut self, signal: &[f32; N]) -> [f32; N]` — two EMA updates + subtract. Return per-dim signed surprise vector.
  - Inline. Branch-free. No allocations.
  - Comment citing O'Reilly 2026 §Implementational (CaMKII/DAPK1 mapping).
- [ ] **T1.3** Implement `pub fn surprise_norm(&self) -> f32` — L2 norm of `(fast − slow)`. Use `simd_dot_f32` on the difference buffer when N ≥ 4.
- [ ] **T1.4** Implement `pub fn reset(&mut self)` — fill both arrays with 0.0 (for entity respawn / session restart).
- [ ] **T1.5** Implement `pub fn derivative_slice(&self, out: &mut [f32; N])` — write `(fast − slow)` into a caller-provided buffer (zero-alloc read path for consumers that already have a scratch buffer).
- [ ] **T1.6** Add SIMD-optimized `observe_simd` reusing `simd_fused_decay_write` for the two EMA passes:
  - `simd_fused_decay_write(&mut self.fast, 1.0 − α_fast, signal, α_fast)`
  - `simd_fused_decay_write(&mut self.slow, 1.0 − α_slow, signal, α_slow)`
  - Then SIMD subtract for the output. Gate behind `simd` feature (same as other SIMD paths).
- [ ] **T1.7** Bridge helper `pub fn sigmoid_surprise_gate(derivative: &[f32], beta: f32) -> f32` — `1.0 / (1.0 + (-beta * L2_norm(derivative)).exp())`. Canonical downstream projection. Document: **never** softmax (per AGENTS.md).
- [ ] **T1.8** Add `lib.rs` feature gate `temporal_deriv` in `crates/katgpt-core/src/lib.rs`. Default-off. Add to `.docs/01_overview.md` feature flag table once Phase 1 is green.
- [ ] **T1.9** Unit tests (`#[cfg(test)]` in `temporal_deriv.rs`):
  - Zero signal → zero derivative, integrators stay at 0.
  - Constant signal → derivative converges to 0 (paper's 25→25 and 50→50 cases, both flat → no change).
  - Step signal (0→1) → positive derivative spike that decays as slow integrator catches up.
  - Reverse step (1→0) → negative derivative spike.
  - `alpha_fast > alpha_slow` enforcement (debug panic on violation).
  - `reset()` zeroes state.
  - `surprise_norm()` matches manual L2 computation.
- [ ] **T1.10** Microbenchmark in `crates/katgpt-core/benches/temporal_deriv_bench.rs`:
  - `observe` scalar vs SIMD for N=1, N=8, N=16.
  - Target: <10ns per `observe` call at N=8 on Apple Silicon arm64 release build.
  - 1000-NPC batch (1000 × N=8 kernels in a Vec) target: <10µs total via rayon chunked iteration.

**Phase 1 exit:** `cargo test --features temporal_deriv` green; bench proves <10ns/N=8 observe and <10µs/1000-NPC batch.

---

## Phase 2 — Fusion F1: HLA Companion (sense_composition)

Target: extend `crates/katgpt-core/src/sense/reconstruction.rs`. Adds a per-NPC 8-dim surprise vector as an output channel of the reconstruction cycle.

### Tasks

- [ ] **T2.1** Add `pub surprise: Option<TemporalDerivativeKernel<8>>` to `ReconstructionState` (behind `temporal_deriv` feature). `None` when feature disabled → zero cost.
- [ ] **T2.2** In `reconstruct_inner`, after `evolve_hla` / `evolve_hla_simd`, call `surprise.observe(&self.hla)` if `Some`. Store result in a new `pub last_surprise: [f32; 8]` field (zero-init, only written when feature on).
- [ ] **T2.3** Add accessor `pub fn surprise_vector(&self) -> Option<&[f32; 8]>` — returns `Some(&self.last_surprise)` if feature on, `None` otherwise. Clean downstream API.
- [ ] **T2.4** Add `pub fn surprise_norm(&self) -> f32` — 0.0 when feature off, otherwise delegates to kernel.
- [ ] **T2.5** Wire `ReconstructionConfig` with `temporal_deriv_alpha_fast: f32` (default 0.3) and `temporal_deriv_alpha_slow: f32` (default 0.03). Documented as the paper's ~10× ratio.
- [ ] **T2.6** Synthetic emotional-event trace benchmark (`benches/reconstruction_bench.rs` extension):
  - Generate a 1000-tick trace with embedded events (combat onset at t=200, loot at t=500, encounter at t=800) — HLA gets a step change at those ticks.
  - **G2 gate:** does `surprise_norm()` peak within ±10 ticks of each embedded event? Target: ≥80% recall of events, ≤10% false positives (peaks outside event windows).
  - Compare against baseline: raw `hla.norm()` magnitude does *not* peak at events (it's monotonic). The derivative should.
- [ ] **T2.7** `evolve_hla_simd` path must also feed the surprise kernel — ensure SIMD HLA variant produces identical surprise output to scalar variant (numerical equivalence test).

**Phase 2 exit:** G2 gate passes on synthetic trace. Surprise vector lags HLA by design but peaks at events.

---

## Phase 3 — Fusion F2: δ-Mem Temporal Write Gate (delta_mem feature)

Target: extend `src/delta_mem/` (Plan 053). Adds a derivative-based write gate so memory consolidates only on surprising events.

### Tasks

- [ ] **T3.1** Add `pub surprise_gate: Option<TemporalDerivativeKernel<K>>` to `DeltaMemoryState` (where K = memory rank). Behind combined feature `delta_mem` + `temporal_deriv`.
- [ ] **T3.2** In `write_segment`, before the existing prediction-error-driven write, compute `surprise = surprise_gate.observe(&self.recent_query_embedding)` if `Some`. Skip the write entirely if `surprise_norm() < θ_surprise` (config, default 0.05).
- [ ] **T3.3** Track two counters: `writes_total` and `writes_gated` (how many writes the derivative suppressed). Expose via `pub fn write_suppression_rate(&self) -> f32`.
- [ ] **T3.4** **G3 gate** benchmark: run δ-Mem on the existing synthetic query stream from Plan 053's T8, with vs without the temporal gate. Target: ≥30% write reduction (`write_suppression_rate ≥ 0.30`) with ≤5% recall loss on the existing recall-quality test.

**Phase 3 exit:** G3 passes — ≥30% fewer writes, ≤5% recall loss.

---

## Phase 4 — Fusion F3: Collapse Detector Fusion (collapse_aware_thinking)

Target: extend `src/collapse_aware/` (Plan 212). Adds prediction-derivative collapse as an orthogonal signal to the existing entropy ring-buffer.

### Tasks

- [ ] **T4.1** Add `pub derivative_collapse: Option<TemporalDerivativeKernel<1>>` to the collapse-aware state (single-dim: the entropy signal itself). Behind `collapse_aware_thinking` + `temporal_deriv`.
- [ ] **T4.2** In the per-token observe path, after computing entropy, call `derivative_collapse.observe(&[entropy])` if `Some`. Store the scalar derivative.
- [ ] **T4.3** New collapse signal: `derivative_collapse_detected = |surprise| < τ_deriv AND entropy_ring_buffer_above_threshold`. Logic: entropy hasn't collapsed yet, but its *derivative* has gone to zero — the system is "coasting" and may be about to collapse. Emit a softer early-warning signal.
- [ ] **T4.4** **G4 gate** benchmark: synthetic collapse suite from Plan 212's G1. Inject both entropy-collapse (one-hot forced) and derivative-only-collapse (gradual convergence to a fixed point with non-zero entropy). Target: false-negative rate on the gradual-convergence traces drops by ≥20% with the derivative signal vs without.

**Phase 4 exit:** G4 passes — ≥20% fewer false negatives on gradual-collapse traces.

---

## Phase 5 — Fusion F4: Intrinsic Curiosity (cgsp)

Target: extend `crates/katgpt-core/src/cgsp/` (Plan 274). Adds a derivative-driven curiosity signal as a cheaper alternative to CGSP's Solver-based reward.

### Tasks

- [ ] **T5.1** Add `CuriosityConjecturer` impl `DerivativeCuriosity` that uses a `TemporalDerivativeKernel<D>` on the bandit/arm-preference vector. No Solver call.
- [ ] **T5.2** The conjecturer's "interestingness" score is `sigmoid(β · surprise_norm())` — zero-cost, computed from the bandit's own preference trajectory.
- [ ] **T5.3** **G5 gate** benchmark (mirrors CGSP's G2 collapse-recovery test): force one-hot priorities, count cycles to recover (entropy ≥ τ_low). Target: match CGSP's 1-cycle recovery (or within 2×) at ≤10% of CGSP's per-cycle cost (CGSP is 831ns/cycle; derivative target ≤100ns/cycle since no Solver).
- [ ] **T5.4** Honest comparison: CGSP's `(1 − solve_rate) · guide_score` carries semantic information about problem difficulty that a pure derivative signal cannot. Document where derivative-curiosity is weaker (target-seeking tasks — same caveat as CGSP G1 informational).

**Phase 5 exit:** G5 passes — derivative curiosity matches CGSP on collapse recovery at ≤10% cost. Documented honest comparison of where each wins.

---

## Phase 6 — GOAT Decision & Promotion

### Tasks

- [ ] **T6.1** Aggregate G2–G5 results into `.benchmarks/277_temporal_deriv_goat.md`. Honest scorecard: which fusions passed, which failed, which were informational.
- [ ] **T6.2** **Promotion rule (AGENTS.md):**
  - If ≥2 of {G2, G3, G4, G5} PASS → promote `temporal_deriv` to default-on. Demote any loser (e.g., if derivative-curiosity beats CGSP on cost AND matches on recovery, CGSP stays for target-seeking but derivative becomes default for exploration-only).
  - If exactly 1 PASS → keep `temporal_deriv` opt-in; document the one winning fusion as the canonical use case.
  - If 0 PASS → demote to Gain. Keep the primitive shipped (it's cheap and useful as a library) but mark all fusions as failed experiments.
- [ ] **T6.3** Update `README.md` with a new "Temporal Derivative Kernel" section under Feature Showcase (only if ≥1 fusion passes). Include the GOAT proof table.
- [ ] **T6.4** Update `.docs/01_overview.md` feature flag table with final status (opt-in or default-on).
- [ ] **T6.5** If all 4 fusions pass AND the "unified surprise bus" pattern (one kernel driving all four consumers) benchmarks cleanly (single-α works for all four, or a small α-schedule per consumer is clean): open a follow-up Super-GOAT escalation issue in `.issues/` referencing Research 243 §2.5. Do NOT claim Super-GOAT in this plan — that requires a separate note after validation.

**Phase 6 exit:** Honest verdict in benchmark doc. Feature flag set per promotion rule. Issue opened if Super-GOAT escalation is warranted.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `α_fast`/`α_slow` tuning is fragile across consumers | Sweep in T1.10 bench; allow per-consumer override via `ReconstructionConfig` / `DeltaMemoryConfig`. Document the paper's ~10× ratio as the default starting point. |
| Derivative signal redundant with existing signals on some consumers | That's why each fusion has its own gate. If G3 fails (δ-Mem doesn't benefit), ship the primitive anyway and just don't wire that consumer. |
| SIMD path numerical divergence from scalar | T2.7 explicitly tests equivalence. Reuse `simd_fused_decay_write` which already has scalar-equivalence tests. |
| Feature-flag combinatorial explosion (`temporal_deriv` × `sense_composition` × `delta_mem` × ...) | Each fusion's combined feature is documented in `.docs/01_overview.md`. `cargo check --all-features` must stay green. |
| Per-NPC storage bloat at 1000 NPCs | 72 bytes/NPC × 1000 = 72 KB. Fits in L1. Documented in Research 243 §2.3. |
| "Unified surprise bus" Super-GOAT claim tempts an early write-up | T6.5 explicitly forbids claiming Super-GOAT in this plan. Issue-only escalation post-validation. |

---

## Expected Performance

| Metric | Target | Reason |
|---|---|---|
| `observe()` N=8 scalar | <10ns | 2 × 8 FMA + 8 subtract, branch-free |
| `observe()` N=8 SIMD | <5ns | Two `simd_fused_decay_write` calls on 8-wide SIMD |
| 1000-NPC HLA companion batch | <10µs/tick | Rayon 8 chunks, 72 KB working set fits L1 |
| δ-Mem gate overhead | <50ns/write | One `observe(K)` + one norm + one compare |
| Collapse-detector overhead | <2ns/token | N=1 kernel, called once per token alongside entropy |
| Derivative curiosity cycle | <100ns | No Solver call (vs CGSP 831ns) |

---

## Commercial Alignment

- **katgpt-rs (public, MIT):** ships the generic `TemporalDerivativeKernel` primitive + the four fusion hooks. No game IP.
- **riir-ai (private):** consumes the primitive for NPC intrinsic motivation (future plan, post-GOAT). The "neocortex-style surprise signal for 1000 concurrent NPCs" story is the marketing hook — but only if the GOAT gate passes.
- **riir-train (private):** the paper's *weight-update* mechanism (kinase-driven LTP/LTD) is explicitly out of scope here. If we ever want biologically-plausible *training*, that's a separate `riir-train/.research/` note.

**Flywheel:** better surprise signal → better curiosity/collapse/memory → better NPC behavior → better game → more players → more compute budget for riir-train to train better base models → better surprise signal.

---

## Cross-References

- **Research:** [243_Temporal_Derivative_Kernel_Neocortical_Learning](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md)
- **Prior art (HLA):** [242_Topological_State_Tracking_Recurrent_Belief](../.research/242_Topological_State_Tracking_Recurrent_Belief.md), [Plan 276](276_micro_recurrent_belief_state.md), [Plan 221](221_kg_latent_octree_sense_composition.md)
- **Prior art (δ-Mem):** [Plan 053](053_delta_mem_modelless.md)
- **Prior art (collapse):** [Plan 212](212_collapse_aware_adaptive_thinking.md), [Benchmark 212](../.benchmarks/212_collapse_aware_goat.md)
- **Prior art (curiosity):** [Plan 274](274_curiosity_guided_self_play.md), [Benchmark 274](../.benchmarks/274_cgsp_goat.md)
- **Source:** [arXiv:2606.08720](https://arxiv.org/abs/2606.08720), [bioRxiv 2026.06.05.730489](https://www.biorxiv.org/content/10.64898/2026.06.05.730489v1) (Jang et al. companion experimental paper)

---

**TL;DR:** Ship a generic dual fast/slow temporal-derivative kernel distilled from O'Reilly 2026's neocortical learning theory. Phase 1: primitive + bench. Phases 2–5: four independent fusions (HLA companion, δ-Mem write gate, collapse detector, intrinsic curiosity), each with its own GOAT gate. Phase 6: promote ≥2 winners to default-on, demote losers, honestly document failures. Opt-in `temporal_deriv` feature until the gate passes. The paper's weight-update mechanism is explicitly training → riir-train and is NOT in scope here.
