# Plan 341: TEMP — Perturbed-Loss-Vector Diversity Fingerprint (Open Primitive)

**Date:** 2026-06-29
**Research:** [katgpt-rs/.research/323_TEMP_Perturbed_Loss_Vector_Fingerprint.md](../.research/323_TEMP_Perturbed_Loss_Vector_Fingerprint.md)
**Source paper:** [arxiv 2606.26797](https://arxiv.org/abs/2606.26797) — Jin et al., *Reasoning Quality Emerges Early: Data Curation for Reasoning Models*, ICML 2026.
**Target:** `katgpt-rs/crates/katgpt-core/src/diversity/` (new module) + Cargo feature `temp_loss_fingerprint`
**Status:** Active — Phase 1 skeleton

> **Cross-repo:** private selling-point guide at `riir-neuron-db/.research/010_Perturbed_Loss_Vector_Sleep_Consolidation_Guide.md`; private shard integration plan at `riir-neuron-db/.plans/005_temp_consolidation_diversity_selector.md`.

---

## Goal

Ship a generic, modelless primitive that, given two latent-state checkpoints `S_0, S_1` and a candidate experience set, computes a **perturbed-loss-vector diversity fingerprint** per candidate and selects the K-subset with maximal spread. The selection bound (Theorem 3.1 of the source paper, reframed modellessly in Research 323) upper-bounds the gradient-along-`v` differences that the next weight-mutation cycle (freeze/thaw swap or consolidation tick) would induce — **without computing gradients**.

**Feature flag:** `temp_loss_fingerprint` (opt-in). Promotion to default-on requires the GOAT gate (G1–G5 below) to pass.

**Modelless invariant (AGENTS.md):** no training, no gradients, no backprop. The checkpoints are committed shards; the extrapolated snapshots are deterministic linear combinations; the loss is a per-step NLL on a short prefix; the bound is pure arithmetic.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create module `crates/katgpt-core/src/diversity/mod.rs` with feature gate `temp_loss_fingerprint`. Re-export public API under `katgpt_core::diversity::temp::*`.
- [x] **T1.2** Define the `LossKernel` trait:
  ```rust
  /// Per-step negative-log-probability kernel at a given parameter snapshot.
  /// Implementors: `ac_prefix::ConditionalLogprob` (token-level NLL),
  /// HLA surprise wrapper, functor-coherence wrapper, KARC residual wrapper.
  pub trait LossKernel {
      /// Compute `L_z(theta) = sum_{t<N} -log p(z_prefix[t] | z_prefix[<t], theta)`.
      /// `theta` is the flattened parameter snapshot (e.g. `style_weights[64]`).
      /// `z_prefix` is the first N steps of the candidate experience.
      fn short_prefix_loss(&self, theta: &[f32], z_prefix: &[f32]) -> f32;
  }
  ```
- [x] **T1.3** Implement `extrapolated_snapshot_schedule(s0, s1, k, lambda_schedule, noise_seeds, out)` — deterministic linear extrapolation `theta_j = s0 + lambda_j * v` where `v = s1 - s0`, with optional `xi_j` multiplicative noise from a fixed BLAKE3-seeded RNG (paper Eq. 5). Zero-allocation: writes into caller-provided `&mut [Vec<f32>]`. Unit test: `extrapolated_snapshot_schedule(k=4)` with `xi=0` produces 4 evenly-spaced points on the `[s0, s1]` line; with `xi != 0` produces points within `±lambda * sigma` of the line.
- [x] **T1.4** Implement `perturbed_loss_vector(kernel, theta_schedule, z_prefix, out)` — calls `kernel.short_prefix_loss(theta_j, z_prefix)` for each `j`, writes the loss vector `L_z` into `out: &mut [f32]` (len == k). Zero-allocation.
- [x] **T1.5** Implement `lipschitz_gradient_bound(delta, lambda, g, tau, c_h, epsilon) -> f32` — pure arithmetic, paper Theorem 3.1 bound. Unit test: `delta=0` returns `G*(1/sqrt(2)+tau) + C_H*epsilon` (the irreducible floor); monotone in `delta`.
- [x] **T1.6** Implement `pairwise_bound(loss_vectors, lambda, g, tau, c_h, epsilon, out)` — for each pair `(i, j)`, compute `delta_ij = ||L_i - L_j||_inf` and write `lipschitz_gradient_bound(delta_ij, ...)` into `out[i * n + j]`. Zero-allocation, rayon-parallelizable for `n > 64`.
- [x] **T1.7** Implement `select_diverse_subset(loss_vectors, k, scratch) -> Vec<usize>` — greedy max-min spread: seed with the two candidates with max pairwise `delta`; iteratively add the candidate that maximizes min-distance to the current subset. Returns `k` indices. Paper §3.2 Algorithm 1 modelless analog.
- [x] **T1.8** Feature-gate audit: `cargo check --features temp_loss_fingerprint` compiles; `cargo check` (default) does not include the module. Run `cargo hack check --each-feature` + `--all-features` per the `merkle_root` lesson. **DONE 2026-06-29:** `--no-default-features`, default, `--features temp_loss_fingerprint`, `--all-features` all compile clean. 10/10 unit tests pass.

**Phase 1 exit (DONE 2026-06-29):** `cargo test -p katgpt-core --features temp_loss_fingerprint --lib diversity::temp` passes — **10/10 unit tests** (≥8 required). Module compiles in isolation, no game/chain/shard semantics, no dependencies outside `katgpt-core`. Feature isolation verified: `--no-default-features` + default + `--features temp_loss_fingerprint` + `--all-features` all clean.

---

## Phase 2 — GOAT Gate (REQUIRED for default-on consideration)

The primitive does not claim a UQ distribution, so the **"Report the Floor" rule** (Plan 340 / Research 322) does not directly apply. The GOAT gate is the standard G1–G5 below.

### Tasks

- [ ] **T2.1** **G1 — Bound preservation under diversity selection.** Synthetic test: construct K=64 candidate loss vectors with known pairwise `delta_ij`. Verify `select_diverse_subset(k=8)` picks the 8 with maximal min-pairwise-`delta`. Verify the selected subset's `lipschitz_gradient_bound` is ≥ 2× the bound for a random 8-subset. This proves the diversity selector picks the subset whose members would induce maximally-different gradients.
- [ ] **T2.2** **G2 — Prefix-length sweep.** For a fixed candidate set and snapshot pair, sweep `N ∈ {8, 16, 32, 64, 128, 256}` and verify the diversity ranking (Kendall tau) at `N=32` correlates ≥ 0.85 with the ranking at `N=256`. This is the modelless analog of paper Fig. 6 (first-1k correlates with full-trace). Target: N=32 captures ≥ 85% of the signal — confirms token-efficiency.
- [ ] **T2.3** **G3 — Perf.** Bench `perturbed_loss_vector` for K=8, N=100 on a synthetic `LossKernel` (single matmul). Target: < 5µs per candidate (40 µs for K=8 N=100 = 800 FMA lanes). Bench `select_diverse_subset` for 256 candidates, k=32. Target: < 1ms (greedy max-min is O(k * n * k_vec_ops)). Zero-allocation hot path: `cargo test --features temp_loss_fingerprint` with `#[track_caller]` alloc assertions.
- [ ] **T2.4** **G4 — Determinism / quorum-reproducibility.** Two independent runs with the same `(s0, s1, lambda_schedule, noise_seeds, candidates)` produce bit-identical selected subsets. Test: serialize the run config to bytes, run twice, assert `selected_subset_1 == selected_subset_2`. This is the sync-boundary requirement (Research 323 §5).
- [ ] **T2.5** **G5 — Feature isolation.** `cargo check` (default features) compiles without `temp_loss_fingerprint`; `cargo check --all-features` compiles with it; `--each-feature` produces no combo-only regression.

**Phase 2 exit:** all 5 gates pass. Bench results recorded in `.benchmarks/341_temp_loss_fingerprint_goat.md`. **Promotion decision:** if G1–G5 pass AND the integration plan (riir-neuron-db Plan 005) demonstrates a consolidation-quality gain (G2' there), promote `temp_loss_fingerprint` to default-on. Otherwise keep opt-in with documented reason.

---

## Phase 3 — Composition (post-GOAT only)

- [-] **T3.1** Compose with `ac_prefix::ConditionalLogprob` (Plan 313) — token-level NLL kernel for text traces. **Deferred:** requires a text-trace dataset; deferred until riir-neuron-db Plan 005 G2' demonstrates the gain on shard consolidation (the primary substrate).
- [-] **T3.2** Compose with HLA surprise kernel (`sense/reconstruction.rs`) — per-tick HLA surprise as the loss. **Deferred:** requires riir-ai runtime integration; tracked as cross-ref from riir-neuron-db Plan 005 Phase 3.
- [-] **T3.3** Compose with KARC residual (`Plan 308`) — KARC forecast residual as the loss. **Deferred** to post-GOAT.
- [-] **T3.4** DEC cross-reference (`dec/operators.rs`): document the Stokes-theoretic reframing (`|<d(loss_cochain), v>|` is the directional exterior derivative of loss along `v`). **Documentation only, no code change** — the DEC operators already ship; this is the conceptual bridge. Curse-of-dim caveat noted.

---

## Phase 4 — riir-neuron-db integration (cross-repo, separate plan)

See `riir-neuron-db/.plans/005_temp_consolidation_diversity_selector.md`. The katgpt-rs primitive is the dependency; the neuron-db plan consumes it as a diversity pre-filter on `ConsolidationPipeline::wake_events` before `sleep()` averages them.

---

## Open questions

- **Prefix length per kernel.** Paper uses N=100 (LLM CoT) and N=1k (longer reasoning). Our kernels (HLA tick, functor application, KARC forecast step) may need different N. Phase 2 G2 sweeps this; if N=32 captures ≥85% of signal for HLA, ship that as default.
- **K (number of extrapolated checkpoints).** Paper uses n=8. Our shards are 64-dim; the directional extrapolation is along a single `v`. K=4 may suffice (one perturbed checkpoint gives the bound; more refine it). Phase 2 G3 benches K ∈ {2, 4, 8}.
- **Target snapshot `S_1` provenance.** Research 323 §6.1 flags: `S_1` doesn't exist yet at selection time. Resolution options: (a) use `ArchetypeBlendShard` target as `S_1`; (b) use a personality-direction vector from Plan 336 as `v` directly (no `S_1` needed); (c) use a previous cycle's `S_1` (the last consolidation result). The riir-neuron-db plan picks one; the katgpt-rs primitive is agnostic (caller provides both endpoints).

---

## References

- Source paper: [arxiv 2606.26797](https://arxiv.org/abs/2606.26797) — Jin et al., ICML 2026.
- Research note: [katgpt-rs/.research/323_TEMP_Perturbed_Loss_Vector_Fingerprint.md](../.research/323_TEMP_Perturbed_Loss_Vector_Fingerprint.md)
- Closest cousin: [katgpt-rs/.research/317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md](../.research/317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md) — single-checkpoint Gibbs `1/E²`; this plan is the multi-checkpoint generalization.
- Application target: [katgpt-rs/.plans/334_sleep_time_query_anticipator_primitive.md](334_sleep_time_query_anticipator_primitive.md)
- Cross-repo integration: `riir-neuron-db/.plans/005_temp_consolidation_diversity_selector.md`
