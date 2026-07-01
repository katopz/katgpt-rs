# Plan 311: Alien Sampler Primitive — Coherence × Availability Frontier Ranking

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/293_Alien_Science_Coherence_Availability_Frontier.md](../.research/293_Alien_Science_Coherence_Availability_Frontier.md)
**Source paper:** [arXiv:2603.01092](https://arxiv.org/abs/2603.01092) — Artiles et al., "The Alien Space of Science" (May 2026)
**Target:** `katgpt-rs/src/alien_sampler/` (new module) + Cargo feature `alien_sampler`
**Status:** Complete — Phase 1+2+3 done. GOAT gate FAILED (1/4). Module ships opt-in, NOT promoted to default. See `.benchmarks/311_alien_sampler_goat.md`.

**"Report the Floor" UQ comparison (Issue 010 T6): N/A — EXCLUDED (not UQ-bearing).** The Alien Sampler produces a within-pool z-scored ranking (`score = (1−β)·z_coh + β·(−z_avail)`), which is a *relative selection signal* (which candidate is more diverse in this pool), not a calibrated uncertainty estimate. It claims no probability distribution, predictive interval, quantile, coverage guarantee, or confidence score. Its GOAT gate measures motif-collapse reduction (diversity), not coverage/CRPS/Winkler. Same structural exclusion as BoM (planning quality) and Sleep-Time (compute gating). No floor comparison needed.

---

## Goal

Ship the generic, modelless `AlienSampler<V, C, A>` primitive distilled from arXiv:2603.01092: a within-pool z-scored linear fusion `(1−β)·zC(S) + β·zU(S)` of a coherence score and an unavailability score, plus a `MedianTopMAvailability` implementation of the paper's load-bearing community-aggregation rule (median over top-m cosine retrievals against a precomputed community bank).

**Open primitive only.** No game semantics, no chain semantics, no neuron-shard semantics in this repo. The riir-ai consumer (`cgsp_runtime/alien_bridge.rs`) is a separate plan in riir-ai — out of scope here. The open kernel is ~150 LOC of pure ranking math; the rest is tests + bench.

**GOAT gate:** prove the dual-encoder `MedianTopMAvailability` produces materially better population diversity than a scalar-redundancy baseline (OPUS-style local penalty) on a synthetic motif-collapse scenario. The paper's evidence (95.7%→34.3% top-10 concentration, 8/10→0/10 motif collapse) sets the bar. Promote to default-on if G1+G2+G3+G4 all pass; demote to opt-in-only-or-remove if G1 fails (no diversity gain over scalar redundancy).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `src/alien_sampler/` module tree:
  - `mod.rs` — module root, re-exports
  - `traits.rs` — `CoherenceScorer<V>` + `AvailabilityScorer<V>` traits
  - `sampler.rs` — `AlienSampler<V, C, A>` struct + `rank()` method
  - `median_top_m.rs` — `MedianTopMAvailability` implementation
  - `types.rs` — `AlienConfig`, `ScoredCandidate`, error types
  - `tests.rs` — unit tests (inline module)

- [x] **T1.2** Implement traits in `traits.rs`:
  ```rust
  pub trait CoherenceScorer<V> {
      fn coherence(&self, atoms: &[V]) -> f32;
  }
  pub trait AvailabilityScorer<V> {
      /// Higher = MORE available to the reference community. Sampler negates.
      fn availability(&self, atoms: &[V]) -> f32;
  }
  ```

- [x] **T1.3** Implement `AlienSampler::rank()` in `sampler.rs`:
  - Take `&[Vec<V>]` candidates + caller-provided scratch buffers `&mut [f32]` (zero-alloc hot path per AGENTS.md).
  - Compute coherence + availability into scratch.
  - Within-pool z-score both (mean + std, population formula for determinism).
  - Fuse: `score = (1−β)·z_coh + β·(−z_avail)`.
  - Return `Vec<(score, candidate_idx)>` sorted desc.
  - Handle edge cases: pool size ≤ 1 (std=0), empty candidates, NaN guard.

- [x] **T1.4** Implement `MedianTopMAvailability` in `median_top_m.rs`:
  - Precomputed `community_bank: Vec<Vec<f32>>` (embeddings of repertoire items).
  - `embedding_fn: Fn(&[V]) -> Vec<f32>` or pre-embedded candidates (caller choice).
  - For each candidate: compute cosine against all bank items, take top-m via `select_nth_unstable` (O(n) partial sort, zero-alloc), return median of top-m.
  - `m` configurable (paper uses top-10). Top-1 fallback if bank smaller than m.

- [x] **T1.5** Add feature flag `alien_sampler` to `Cargo.toml` (default-OFF):
  ```toml
  alien_sampler = ["dep:katgpt-core"]  # if any katgpt-core types needed; else just []
  ```
  Wire module in `src/lib.rs` under `#[cfg(feature = "alien_sampler")]`.

- [x] **T1.6** Run `cargo check --features alien_sampler` + `cargo check --no-default-features`. Both must compile.

- [x] **T1.7** Write Phase-1 unit tests in `tests.rs`:
  - `rank_empty_returns_empty`
  - `rank_single_returns_one`
  - `beta_zero_is_coherence_only` (Fβ=0 ranking equals coherence-only ranking)
  - `beta_one_is_unavailability_only` (Fβ=1 ranking equals negated-availability ranking)
  - `median_top_m_top1_fallback` (bank size 1 → returns that one cosine)
  - `median_top_m_paper_default_m10` (m=10, bank of 50, verify median of top-10)
  - `z_score_handles_zero_variance` (all-equal scores → z=0, no NaN)
  - `determinism_same_seed_same_order` (run twice, identical output)

---

## Phase 2 — Correctness + Microbench

### Tasks

- [x] **T2.1** Property tests via `proptest` (if already a dev-dep; else skip and use random fuzz):
  - `rank_is_permutation_of_indices` (output indices are a permutation of `0..candidates.len()`)
  - `beta_monotone_in_coherence_when_avail_const` (when availability is constant, increasing β toward 0 → more coherence-driven)
  - `median_top_m_invariant_to_bank_permutation` (shuffle bank → same result)

- [x] **T2.2** Microbench `benches/alien_sampler_bench.rs`:
  - `bench_rank_1k_candidates` (1000 candidates × 4 atoms × 16-dim bank of 100)
  - `bench_rank_10k_candidates` (10k candidates — warm-tier batch case)
  - `bench_median_top_m_m10_bank100` (single MedianTopMAvailability call)
  - `bench_median_top_m_m10_bank10k` (large-bank case)
  - Target: `rank` 1k ≤ 500µs SIMD, 10k ≤ 5ms; `median_top_m` bank100 ≤ 5µs, bank10k ≤ 500µs.
  - Use `std::time::Instant` + batched timing per crate convention (matches salience_tri_gate bench pattern). No Criterion dep.

- [x] **T2.3** Module-level doc in `src/alien_sampler/mod.rs`:
  - Cite paper (arXiv:2603.01092)
  - Explain the two-axis decomposition (coherence vs availability)
  - Note the load-bearing `MedianTopMAvailability` rule (median over top-m, not top-1)
  - Cross-reference Research 293 and the OPUS baseline (Research 089)

---

## Phase 3 — GOAT Gate (Motif-Collapse Benchmark)

This is the make-or-break phase. The whole point is proving dual-encoder availability beats scalar redundancy on *population* diversity.

### Tasks

- [x] **T3.1** Build synthetic motif-collapse scenario in `benches/alien_sampler_goat.rs`:
  - 100 "NPCs", each with a 16-dim Conjecturer-equivalent pool (just `Vec<f32>` directions).
  - A shared "zone bank" initialized empty; populated as NPCs emit selections.
  - Coherence scorer: dot-product against a fixed per-NPC "personality direction" (so coherence rewards staying on-personality).
  - Without availability pressure, all NPCs converge to the top-3 highest-coherence directions (the "motif").

- [x] **T3.2** Implement three arms:
  - **Arm A (no availability):** β=0, coherence-only ranking. Expected: severe motif collapse.
  - **Arm B (OPUS-style scalar local redundancy):** per-NPC CountSketch-equivalent penalty against own previous selections. Reuse `OpusBanditPruner` if composable; else a minimal local-redundancy stub.
  - **Arm C (AlienSampler):** `MedianTopMAvailability` against the zone bank, β=0.7.

- [x] **T3.3** Run 10k cycles per arm, 5 seeds each. Record per-cycle:
  - Selected direction index per NPC
  - Coherence score of selected direction
  - Bank state (for arm C)

- [x] **T3.4** Compute metrics:
  - **G1 (motif collapse):** top-10 direction concentration across the zone (fraction of all selections in last 1000 cycles that hit the top-10 most-selected directions). Arm C must be ≤ 50% of Arm B's concentration. (Paper analog: 95.7%→34.3% is ~36% of baseline.)
  - **G2 (quality preservation):** mean coherence of selected directions in last 1000 cycles. Arm C must be ≥ 90% of Arm A's mean coherence. (Diversity must not destroy quality.)
  - **G3 (perf):** per-cycle wall time for Arm C must be ≤ 5× Arm B's per-cycle wall time. (Dual-encoder is allowed to be slower, but not catastrophically.)
  - **G4 (latent boundary):** no `Vec<f32>` from the bank or per-NPC Fβ score appears in a hypothetical `SyncBlock`-equivalent audit. (This is a static check on the bench — the open primitive has no sync concept, so this gate is "no `Vec<f32>` escapes the `rank()` call boundary in the public API".)

- [x] **T3.5** Write results to `.benchmarks/311_alien_sampler_goat.md` with all four gate verdicts.

### Gate decision tree

- **G1 BORDERLINE-FAIL + G2 FAIL + G3 FAIL + G4 PASS** → DEMOTE. Module stays opt-in for paper reproduction. β sweep (β=0.2, 0.3, 0.5, 0.7) found a sharp phase transition at β≈0.4 with no β satisfying both G1 and G2 on the synthetic single-peak-coherence scenario. The dual-encoder mechanism IS validated (2× concentration reduction at β=0.7), but the scenario's quality/diversity tradeoff is unfavorable. See `.benchmarks/311_alien_sampler_goat.md`.
- **G1 FAIL** → demote. The dual-encoder is not worth the complexity over scalar redundancy. Note honestly in `.benchmarks/311_*.md`, keep module as opt-in for research reproduction, do NOT promote.
- **G1 PASS but G2 FAIL** → diversity at the cost of quality. Investigate β sweep (try β=0.5, 0.6); if no β satisfies both, demote.
- **G1+G2 PASS but G3 FAIL** → perf regression. Profile; if fixable in Phase 4 SIMD pass, proceed; else demote to opt-in.

---

## Phase 4 — Hot-path Optimization (only if Phase 3 passes)

### Tasks

- [x] **T4.1** SIMD-ify `MedianTopMAvailability` cosine computation (4 or 8 lanes). **CLOSED via rayon NPC-parallelization in GOAT bench (Issue 002 follow-up, 2026-06-24).** SIMD inner-loop not needed — rayon alone closes G3 (38.42× → ~4.5× on 16 cores). The 4-accumulator `dot_4acc` from Issue 002 C2 was dead code (slower than sequential without `target-cpu=native`) and has been deleted; `dot_seq` is the only shipped kernel.
- [x] **T4.2** Hoist z-score computation into a single pass (compute mean + std in one loop, fuse in second). **DONE** as part of Phase 1 (`fuse_and_sort` kernel).
- [x] **T4.3** Re-run Phase 3 G3 perf measurement; confirm improvement. **DONE** — G3 = ~4.5× post-rayon (target ≤5×), observed range 4.49×–4.99× on Apple M3 Max (16 cores). See `.benchmarks/311_alien_sampler_goat.md` "Post-Rayon G3 re-measurement" section.
- [x] **T4.4** If `rank()` allocates the return `Vec`, add `rank_into(&mut Vec<(f32, usize)>)` variant for callers that want to reuse the output buffer. **DONE** — shipped `rank_into` + `rank_precomputed` (hot-path batch variant).

---

## Phase 5 — Promotion (only if Phase 4 passes)

### Tasks

- [x] **T5.1** Add `alien_sampler` to `default = [...]` in `Cargo.toml`. **SKIPPED — CLOSED N/A.** GOAT gate failed (G1 0.5010, G2 0.6724); promotion precondition not met. Module stays opt-in.
- [x] **T5.2** Update `katgpt-rs/README.md` Feature Flags table + add to "Always-On Hot Path" section if perf qualifies. **SKIPPED — CLOSED N/A** (depends on T5.1).
- [x] **T5.3** Update `katgpt-rs/.docs/01_overview.md` module structure. **SKIPPED — CLOSED N/A** (depends on T5.1).
- [x] **T5.4** Cross-reference from `katgpt-rs/.research/089_OPUS_*.md` ("if population-level diversity is needed, prefer `alien_sampler` over `opus_selection`'s local-redundancy"). **SKIPPED — CLOSED N/A** — the GOAT gate did not prove `alien_sampler` beats OPUS on this scenario.
- [x] **T5.5** Commit on `develop` with `feat:` prefix per AGENTS.md.

---

## Out of Scope (filed as follow-ups, not in this plan)

- **riir-ai consumer wiring** (`cgsp_runtime/alien_bridge.rs` + `CommunityAvailabilityBank` populated from zone NPC emissions). Belongs in riir-ai Plan 312+ — the private selling-point wiring. This plan ships the open math only.
- **Autoregressive coherence transformer training.** → riir-train if anyone wants the full paper reproduction.
- **AlienSampler over our own `.research/` corpus** (meta-meta-research tool). Separate process tool, not a primitive.
- **LatCal commitment of Fβ scores for chain provenance.** Speculative; not needed until a chain consumer asks for it.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| G1 fails — scalar redundancy is "good enough" on synthetic scenario | Medium | This is the most likely outcome. If it happens, the honest note in `.benchmarks/311_*.md` is the deliverable; the module stays as opt-in for paper reproduction. The paper's evidence is on real research corpora, not synthetic NPC populations — transfer to our domain is unvalidated. |
| Synthetic scenario is too easy (trivially passes G1) | Medium | Make the coherence surface multi-modal (3-5 peaks, not 1) so there are multiple "valid" motifs; the alien sampler should spread across them while scalar redundancy collapses to one. |
| MedianTopM perf is bad on large banks | Low | `select_nth_unstable` is O(n) expected; SIMD cosine in Phase 4 helps. If still bad, add approximate top-m via a heap of size m. |
| Feature interaction with existing bandit features (`opus_selection`, `bandit`) | Low | `alien_sampler` is a pure ranking primitive — no Cargo feature deps on other bandit features. OPUS comparison happens in the bench, not in the module. |

---

## TL;DR

Ship `AlienSampler<V, C, A>` as the open generic primitive from arXiv:2603.01092 — within-pool z-scored fusion `(1−β)·zC + β·zU` of coherence × unavailability, plus `MedianTopMAvailability` implementing the paper's load-bearing community-aggregation rule. ~150 LOC of pure ranking math, no game/chain/shard semantics. **GOAT gate (Phase 3) is the make-or-break**: prove on a synthetic 100-NPC motif-collapse scenario that dual-encoder availability beats OPUS-style scalar local redundancy at reducing top-10 direction concentration (target ≤50% of baseline, paper analog 95.7%→34.3%) without sacrificing >10% coherence quality. Promote to default if G1+G2+G3+G4 pass; demote honestly if G1 fails (the most likely failure mode — paper evidence is on research corpora, transfer to NPC populations is unvalidated). Phase 4 SIMD pass only if Phase 3 passes. **Out of scope: riir-ai consumer wiring (separate plan), autoregressive coherence training (→ riir-train).**
