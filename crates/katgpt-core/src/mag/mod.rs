//! MAG — Mining via Activation Geometry (arXiv:2607.04222).
//!
//! Unsupervised, modelless direction-mining + modelless transfer-prediction
//! primitive. Distilled from LeVi, David, Fomin (ICML 2026 FAGEN Workshop).
//!
//! ## What this is
//!
//! **The missing acquisition step** for the direction-vector ecosystem. Today
//! every direction vector in the codebase is either designer-authored
//! (LatentFieldSteering Plan 309) or supervised-extracted (EmotionDirections
//! Plan 162, KG Latent Octree R196 — mean-difference on labeled data). MAG mines
//! directions **unsupervised** from the host's own runtime verdict `y_M` — no
//! human labels, no gradient descent.
//!
//! Plus the §4 transfer-prediction experiment is a genuinely new capability:
//! modelless "which experience teaches the most". The paper achieves 94.7%
//! Top-1 accuracy predicting dataset transfer; raw centroid cosine achieves
//! only ρ ≈ 0.03 (near random).
//!
//! ## The two halves
//!
//! 1. **Mining** ([`mining`]): `mine_direction` / `mine_contrast_direction`
//!    extract unit-norm feature directions from activation shifts, using the
//!    model's own verdict `y_M` as the label. `reconstruction_error` computes
//!    the linearity diagnostic ϵ_Q (≈0 ⇒ steerable, ≈1 ⇒ entrenchied).
//!    `calibrate_alpha` makes injection strength substrate-invariant.
//!    `apply_operator` computes the 8 MAG readout summaries.
//!
//! 2. **Transfer** ([`transfer`]): `transfer_score` / `rank_candidates` predict
//!    which candidate dataset/experience best improves a target capability —
//!    modellessly, via geometric comparison of activation sets.
//!
//! ## Quick start
//!
//! ### (a) Mine a direction from paired activations
//!
//! [`mine_direction`] extracts a unit-norm feature direction from the mean
//! shift `m(Q‖p) − m(p)` between prefix-conditioned and unconditioned
//! activations.
//!
//! ```
//! use katgpt_core::mag::mine_direction;
//!
//! // Synthetic: with_prefix = without_prefix + v  (a perfectly linear shift).
//! let without_prefix = [vec![0.0_f32, 0.0], vec![0.1, 0.2]];
//! let with_prefix     = [vec![1.0_f32, 0.0], vec![1.1, 0.2]]; // exactly +[1, 0]
//!
//! let dir = mine_direction(&with_prefix, &without_prefix)?;
//! // dir.direction is unit-norm and recovers the shift [1, 0].
//! assert!((dir.as_slice()[0] - 1.0).abs() < 1e-5);
//! assert!(dir.as_slice()[1].abs() < 1e-5);
//! # Ok::<(), katgpt_core::mag::MagError>(())
//! ```
//!
//! ### (b) Compute the ϵ_Q linearity diagnostic
//!
//! [`reconstruction_error`] returns `(ϵ_Q, cosine)`. ϵ_Q ≈ 0 ⇒ the shift is a
//! single linear direction (steerable); ϵ_Q ≈ 1 ⇒ no net shift on average;
//! ϵ_Q > 1 ⇒ overshoot (the candidate direction overshoots the true shift).
//!
//! ```
//! use katgpt_core::mag::{mine_direction, reconstruction_error};
//!
//! let without_prefix = [vec![0.0_f32, 0.0], vec![0.1, 0.2]];
//! let with_prefix     = [vec![1.0_f32, 0.0], vec![1.1, 0.2]];
//!
//! let dir = mine_direction(&with_prefix, &without_prefix)?;
//! let (eps_q, _cos) = reconstruction_error(
//!     &with_prefix, &without_prefix, dir.as_slice(), 1.0,
//! )?;
//! assert!(eps_q < 1e-5); // perfectly linear ⇒ ϵ_Q ≈ 0
//! # Ok::<(), katgpt_core::mag::MagError>(())
//! ```
//!
//! ### (c) Rank candidate datasets by transfer score
//!
//! [`rank_candidates`] scores each candidate against a target using a set of
//! [`TransferMetric`]s and returns them sorted by mean percentile rank (the
//! paper's §4 protocol). Higher percentile = better predicted transfer.
//!
//! ```
//! use katgpt_core::mag::{DataSet, rank_candidates, TransferMetric};
//!
//! let cand_a = vec![vec![1.0_f32, 0.0], vec![0.9, 0.1], vec![0.0, 1.0], vec![0.1, 0.9]];
//! let cand_a_labels = [true, true, false, false];
//! let cand_b = vec![vec![0.3_f32, 0.3], vec![0.4, 0.2], vec![0.2, 0.4], vec![0.1, 0.5]];
//! let cand_b_labels = [true, true, false, false];
//! let target = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
//! let target_labels = [true, false];
//!
//! let candidates = [
//!     DataSet { activations: &cand_a, labels: &cand_a_labels },
//!     DataSet { activations: &cand_b, labels: &cand_b_labels },
//! ];
//! let target_ds = DataSet { activations: &target, labels: &target_labels };
//!
//! let ranking = rank_candidates(
//!     &candidates, &target_ds,
//!     &[TransferMetric::CentroidCosine, TransferMetric::Euclidean],
//! )?;
//! assert_eq!(ranking.len(), 2);
//! assert_eq!(ranking[0].candidate_idx, 0); // cand_a matches target better
//! # Ok::<(), katgpt_core::mag::MagError>(())
//! ```
//!
//! ## Why modelless
//!
//! The "label" is the model/runtime's own verdict `y_M` — a runtime observation
//! (did the NPC succeed? did the claim pass the rubric?), NOT a training target.
//! The math is mean-difference (identical to EmotionDirections) + cosine
//! geometry. No gradients, no backprop, no weight mutation. Mined directions are
//! frozen as `BLAKE3`-committed artifacts (same envelope as
//! `MerkleFrozenEnvelope` in riir-neuron-db).
//!
//! ## §3.5 modelless-unblock relevance
//!
//! MAG direction mining is a **path-3** (latent-space correction) tool per the
//! research skill's modelless-unblock protocol
//! (`.agents/skills/research/SKILL.md` §3.5). A systematically biased verdict
//! (e.g., "signal doubled", "position offset") can potentially be corrected by
//! mining the bias direction and projecting it out — before deferring to
//! riir-train. The `ϵ_Q ≈ 1` diagnostic predicts non-steerability (entrenched
//! bias), flagging when a latent correction won't work. This mirrors the
//! AC-Prefix G1 canonical-failure lesson (Plan 313): a systematic,
//! characterizable bias should be checked for a modelless fix before
//! concluding "needs gradient descent".
//!
//! ## Fusion (the Super-GOAT angle)
//!
//! - **F1**: MAG mines directions → Latent Field Steering (P309) injects them →
//!   NPCs discover reasoning directions from their own experience.
//! - **F2**: MAG transfer prediction = directed curiosity ("what transfers to my
//!   goal?") → CGSP (R126) + AnyRAG escalation.
//! - **F3**: MAG mines archetype directions unsupervised → CommittedFieldBlend
//!   (P321) + PersonalityWeightedComposition (P297) blend them.
//! - **F4**: MAG transfer prediction ranks which experiences to consolidate →
//!   Raven/δ-Mem (riir-neuron-db).
//!
//! ## Status
//!
//! Phase 2 COMPLETE (2026-07-09). GOAT G1–G6 ALL PASS. Promoted to DEFAULT-ON.
//! G2 (the headline kill-it gate) verified: contrast directions mined from
//! self-labeled classes ARE linearly separable (LOO accuracy 0.925 at σ=1.5,
//! 0.810 at σ=3.0). G4: MAG class-conditional transfer Top-1 0.720 vs raw
//! centroid cosine 0.220 (3.3×). Phase 2 added `mine_direction_into` +
//! `transfer_score_into` zero-alloc hot-path variants.
//!
//! See: `katgpt-rs/.research/397_Mining_via_Activation_Geometry.md`
//! See: `katgpt-rs/.plans/418_mag_activation_geometry_primitive.md`
//! See: `katgpt-rs/.benchmarks/418_mag_goat.md`
//!
//! MAG — Mining via Activation Geometry (Plan 418, Research 397, arXiv:2607.04222
//! LeVi/David/Fomin ICML 2026 FAGEN). Unsupervised direction mining + modelless
//! transfer prediction. The missing acquisition step for the direction-vector
//! ecosystem: today every direction is designer-authored (Plan 309) or
//! supervised-extracted (Plan 162). MAG mines them unsupervised from the host's
//! own verdict y_M. mine_direction / mine_contrast_direction extract unit-norm
//! feature directions; reconstruction_error gives the ϵ_Q linearity diagnostic;
//! calibrate_alpha normalizes injection strength; apply_operator computes the 8
//! readout summaries; transfer_score / rank_candidates predict dataset transfer
//! (the §4 94.7% Top-1 result). Mined directions are BLAKE3-committed (same
//! envelope as LatentSteeringVector / MerkleFrozenEnvelope). Pure modelless
//! (mean-difference + cosine geometry). DEFAULT-ON since 2026-07-09 (Phase 2
//! GOAT G1-G6 ALL PASS): G2 (the headline kill-it gate) verified contrast
//! directions from self-labeled classes ARE linearly separable (LOO acc 0.925
//! at σ=1.5, 0.810 at σ=3.0). G4: MAG class-conditional transfer Top-1 0.720
//! vs raw cosine 0.220 (3.3×). Phase 2 added mine_direction_into +
//! transfer_score_into zero-alloc hot-path variants.

pub mod mining;
pub mod transfer;
pub mod types;

// Re-export the public API at the module root for ergonomic access
// (`katgpt_core::mag::mine_direction` instead of `...::mag::mining::mine_direction`).
pub use mining::{
    apply_operator, apply_operator_into, calibrate_alpha, mine_contrast_direction, mine_direction,
    mine_direction_into, reconstruction_error,
};
pub use transfer::{DataSet, RankEntry, rank_candidates, transfer_score, transfer_score_into};
pub use types::{MagDirection, MagError, MagOperator, TransferMetric};
