//! Issue 917 T2 — the D2F decode commit policy under the EB-Sampler rule.
//!
//! The D2F decode loops (`d2f_decode_block_prompt_q_core`,
//! `d2f_decode_block_with_prompt_with_sampler`, `D2fPipeline::decode_all`)
//! commit per position with the fixed threshold
//! `chosen_prob >= tau_conf`. Under this module the per-pass commit set is
//! instead the EB-Sampler's (Ben-Hamu, Gat, Severo, Nolte, Karrer —
//! "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded
//! Unmasking", arXiv:2505.24857, NeurIPS 2025, Algorithm 1 / Eq. 8): the
//! largest prefix of the `(NaN-last, entropy, index)` order whose
//! `Σ H − max H` residual stays `≤ γ`. The T1 policy
//! (`katgpt_core::entropy_bounded_commit`) is consumed, never
//! reimplemented; what lives here is only the D2F join — masked-position
//! collection over the flat logits buffer, the per-pass scratch, and the
//! selection bitmap the decode loops read.
//!
//! **Two-level gate.** The compile-time feature (`entropy_bounded_commit`)
//! gates this code; the runtime signal is `D2fDecodeConfig.eb_gamma ≥ 0`
//! (the default `−1.0` disarms). The incumbent τ branch is byte-identical
//! in every disarmed or feature-off build — the feature alone changes no
//! decode, and the crate's existing `d2f` unit tests (which construct
//! configs via `..Default()`, i.e. disarmed) run unchanged in both
//! postures as the identity check.
//!
//! **No-stall by construction (G1).** A singleton prefix has residual
//! exactly `0 ≤ γ` for every armed `γ ≥ 0`, so any pass with at least one
//! finite-entropy masked candidate commits ≥ 1 — the property the
//! incumbent τ rule lacks (a flat distribution commits nothing and burns
//! the pass). A non-finite-entropy candidate (a NaN logits row) is never
//! committed.
//!
//! **UNCALIBRATED.** `eb_gamma`'s wiring default is inherited from the T1
//! primitive's budget (0.1, the DiffusionGemma report's entropy budget) —
//! it is NOT a calibration of this lane. The Bench-917 oracle dial was
//! flat over [0, 0.3] on the ORACLE families; no D2F number exists: the
//! lane A/B (Issue 917 G2/G3) is deferred on a trained D2F checkpoint
//! (riir-train Plan 437 Phase 4, owner-gated).
//!
//! Zero per-pass allocation (G4): all buffers live in [`D2fEbScratch`],
//! sized once per decode and reused every pass.

use katgpt_core::entropy_bounded_commit::{entropy_bounded_commit, position_stats};

/// Per-pass scratch for [`d2f_commit_set_eb`] (capacity = the decode block
/// size). Reused across denoising passes and blocks.
pub struct D2fEbScratch {
    candidates: Vec<u32>,
    entropy: Vec<f32>,
    keys: Vec<f32>,
    selected: Vec<bool>,
}

impl D2fEbScratch {
    /// Scratch sized for `capacity` masked positions.
    pub fn new(capacity: usize) -> Self {
        Self {
            candidates: vec![0; capacity],
            entropy: vec![0.0; capacity],
            keys: vec![0.0; capacity],
            selected: vec![false; capacity],
        }
    }

    /// Whether relative position `rel` (`p − block_start`) is in the pass's
    /// EB commit set — meaningful only right after [`d2f_commit_set_eb`].
    #[inline]
    pub fn is_selected(&self, rel: usize) -> bool {
        self.selected.get(rel).copied().unwrap_or(false)
    }

    /// The selection bitmap for the pass (`selected()[rel]` mirrors
    /// [`D2fEbScratch::is_selected`]; out-of-range reads `false`).
    pub fn selected(&self) -> &[bool] {
        &self.selected
    }
}

/// The D2F pass commit set under the EB-Sampler policy (Issue 917 T2).
///
/// Unfiltered form — every still-masked position of
/// `tokens[block_start..seq_len]` is a candidate. See
/// [`d2f_commit_set_eb_where`] for the eligibility-filtered generalization
/// (the set-diffusion lane's gen-step window).
///
/// Returns `(committed, masked)` — the commit count and the still-masked
/// count at pass start. Zero allocation: `scratch` must be sized
/// `≥ seq_len − block_start`.
pub fn d2f_commit_set_eb(
    logits_flat: &[f32],
    tokens: &[usize],
    block_start: usize,
    seq_len: usize,
    mask: usize,
    vocab: usize,
    scratch: &mut D2fEbScratch,
    gamma: f32,
    max_commit: usize,
) -> (usize, usize) {
    d2f_commit_set_eb_where(
        logits_flat,
        tokens,
        block_start,
        seq_len,
        mask,
        vocab,
        |_| true,
        scratch,
        gamma,
        max_commit,
    )
}

/// The eligibility-filtered form of [`d2f_commit_set_eb`] (Issue 917 T2,
/// cousin lanes): a masked position is a candidate only when
/// `eligible(p − block_start)` is true, so the γ residual budget is spent
/// over exactly the set the pass may commit (the set-diffusion lane's
/// `gen_step ≤ current_step` window — ineligible positions never consume
/// budget). Everything else — the proxy, the order, the residual, the cap,
/// the scratch layout — is identical to the unfiltered form.
///
/// Returns `(committed, masked)` — the commit count and the MASKED-
/// AND-ELIGIBLE count at pass start (a masked-but-ineligible position
/// counts in neither). The set rides `scratch`: `is_selected(p −
/// block_start)` for each committed position. Returns `(0, masked)` when
/// nothing can commit — no masked eligible positions, a NaN or negative
/// `gamma`, or every candidate non-finite-entropy (mirrors the incumbent's
/// `NaN ≥ τ` = false).
pub fn d2f_commit_set_eb_where(
    logits_flat: &[f32],
    tokens: &[usize],
    block_start: usize,
    seq_len: usize,
    mask: usize,
    vocab: usize,
    mut eligible: impl FnMut(usize) -> bool,
    scratch: &mut D2fEbScratch,
    gamma: f32,
    max_commit: usize,
) -> (usize, usize) {
    let rel_positions = seq_len - block_start;
    let mut masked = 0usize;
    for p in block_start..seq_len {
        if tokens[p] == mask {
            let rel = p - block_start;
            if !eligible(rel) {
                continue;
            }
            // T1 indexes entropy/key BY CANDIDATE VALUE — the candidates
            // here are relative positions, so the per-position arrays are
            // rel-indexed (slot order diverges from rel order as soon as a
            // pass commits a non-suffix position).
            let entropy = position_stats(&logits_flat[p * vocab..(p + 1) * vocab]).entropy;
            scratch.entropy[rel] = entropy;
            scratch.keys[rel] = entropy; // proxy fixed to Entropy: key == entropy
            scratch.candidates[masked] = rel as u32;
            masked += 1;
        }
    }
    for slot in &mut scratch.selected[..rel_positions] {
        *slot = false;
    }
    let committed = entropy_bounded_commit(
        &mut scratch.candidates[..masked],
        &scratch.entropy[..rel_positions],
        &scratch.keys[..rel_positions],
        gamma,
        max_commit,
    );
    for &rel in &scratch.candidates[..committed] {
        scratch.selected[rel as usize] = true;
    }
    (committed, masked)
}
