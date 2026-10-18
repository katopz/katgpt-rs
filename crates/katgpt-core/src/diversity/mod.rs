//! Diversity-selection primitives for modelless experience curation.
//!
//! Currently home to the TEMP perturbed-loss-vector fingerprint (Plan 341,
//! Research 323). Future diversity selectors may land here alongside it.
//!
//! TEMP — Perturbed-Loss-Vector Diversity Fingerprint (Plan 341, Research 323,
//! arXiv:2606.26797 Jin et al. ICML 2026). Modelless diversity selector: given
//! two committed snapshots S_0, S_1, extrapolate K checkpoints along v = S_1 − S_0,
//! compute per-candidate short-prefix loss vectors, and select the K-subset with
//! maximal Lipschitz-bound spread — gradient-diversity ranking without gradients.
//! Theorem 3.1 modelless reframe: similar loss vectors across K extrapolated
//! checkpoints ⇒ similar gradients along v during the next weight-mutation cycle.
//! Composes with ac_prefix::ConditionalLogprob, HLA surprise, RavenSlotLossKernel
//! (riir-neuron-db Plan 005). DEFAULT-ON (Plan 341 Phase 2, 2026-06-29): G1–G5 ALL PASS.
//!
//! Plan 367 Fusion C — QMC variant of `extrapolated_snapshot_schedule`.
//! Low-discrepancy noise coverage → more diverse loss vectors per unit K.
//! Requires both TEMP substrate and the QMC source trait.

pub mod temp;
