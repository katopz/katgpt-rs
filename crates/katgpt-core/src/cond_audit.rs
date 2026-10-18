//! Conditioning-consistency audit — per-junction forward-KL + the Pinsker
//! total-variation verdict between conditioning regimes (Issue 719;
//! Research 528, arXiv:2609.00865 "MemoryWalker").
//!
//! At any serving site that conditions on a *semantically compressed* context
//! (sliding window, eviction, summarization, budget packing), a two-forward
//! pair over the SAME decode positions yields a distributional
//! behavioral-gap measurement between the conditioning regimes:
//!
//! - **student** — next-token logits under the compressed conditioning;
//! - **teacher** — next-token logits under the full/uncompressed conditioning
//!   (conceptually no-grad; at inference this is just a second forward).
//!
//! The audit sums per-junction forward-KL `KL(teacher ‖ student)` into
//! [`AuditReport::eps_kl`] and reports the Pinsker verdict
//! `TV <= sqrt(eps_kl / 2)` (unconditional — a pure fact about the two
//! categorical distributions, no train-time assumptions), plus the coarse
//! greedy-stream flip counter (`argmax student != argmax teacher` — the
//! Bench 756 paired-gate shape).
//!
//! # Why forward-KL (teacher first)
//! `KL(teacher ‖ student) = Σᵢ tᵢ·(log tᵢ − log sᵢ)` charges the student for
//! every token the full-context teacher finds plausible: where the student
//! *drops* mass the teacher carries — the compression failure mode ("the
//! model was taught it knows the weather") — the term grows without bound
//! and the deficit cannot hide. The reverse direction is mode-seeking and is
//! blind on exactly that deficit side.
//!
//! # Why softmax normalization is correct here
//! The house "sigmoid not softmax" rule targets latent gating and
//! latent→raw projection (never renormalize a score vector into a fake
//! distribution). Next-token logits over a vocabulary ARE a categorical
//! distribution, and KL between categoricals is defined only after
//! normalization — so both logit vectors are log-softmaxed (max-shift +
//! log-sum-exp, stable) before the divergence. This is the categorical-KL
//! diagnostic, not latent gating. The numeric core DELEGATES to the existing
//! [`crate::stale_residual::kl_logits`] substrate (same contract: stable
//! logits-in KL-out, q-side underflow floored at ln(1e-30) so a prohibited
//! token reads as large-but-finite, never +inf) — no duplicate implementation.
//!
//! # Calibrated-zero arm (compression-off control)
//! Run [`audit_conditioning`] with two forwards that emit bit-identical
//! logits: `eps_kl` is EXACTLY `0.0` (every `t·(log t − log s)` term is
//! `t·0`) and the flip count is 0. Small nonzero floors arise only when the
//! two forward paths diverge in last-bit numerics (batched vs unbatched
//! kernels); that floor is the baseline any real conditioning gap must clear
//! before it is interpreted (the Bench 649/802 drift-control discipline).
//!
//! # Honest bound accounting (multi-junction)
//! Pinsker is pairwise: `TV_j <= sqrt(KL_j / 2)` is exact per junction. A
//! chain of K regimes telescopes through the triangle inequality to
//! `TV(first, last) <= Σⱼ sqrt(KL_j / 2) <= sqrt(K · eps_kl / 2)`
//! (Cauchy–Schwarz). [`AuditReport::tv_bound`] carries the issue-prescribed
//! `sqrt(eps_kl / 2)` form (exact for a single junction; the paper's O()
//! convention) and [`AuditReport::tv_bound_chain`] carries the K-aware safe
//! form for multi-junction gating. Per-junction KLs ship in the report for
//! exact per-junction Pinsker.
//!
//! Status: opt-in POC (`cond_audit`), NO live consumer — every shipped
//! numeric-compression surface is already gated stronger at bit-identity;
//! the semantic surfaces (Gemma-4 sliding ring, TokenBudgetPacker, H2O) are
//! trigger-gated in Issue 719 T2–T4. No default promotion, no GOAT claim.
//!
//! Conditioning-consistency audit (Issue 719 / Research 528, arXiv:2609.00865
//! "MemoryWalker"): per-junction forward-KL between a compressed-conditioned
//! (student) forward and a full-context (teacher) forward over the same decode
//! positions + the unconditional Pinsker TV verdict `TV <= sqrt(eps_KL/2)` +
//! the greedy-stream flip counter + the calibrated-zero (compression-off)
//! control arm. Modelless pure-f32 arithmetic; the per-junction KL DELEGATES
//! to `stale_residual::kl_logits` (substrate composition, no duplicate
//! numeric core). Opt-in — NO live consumer (every shipped numeric-compression
//! surface is gated stronger at bit-identity); T2 Gemma-4 ring / T3 packer /
//! T4 H2O stay trigger-gated. No default promotion, no GOAT claim until a
//! consumer exists.

use crate::stale_residual::kl_logits;

/// Audit policy: the only caller knob is the verdict threshold.
#[derive(Debug, Clone)]
pub struct CondAuditConfig {
    /// Total-variation budget that [`AuditReport::tv_bound`] must not exceed
    /// for [`AuditReport::verdict_pass`]. Policy, not a calibrated constant —
    /// consumers set the behavioral-gap budget their surface tolerates (the
    /// Issue 719 PoC gates use 0.05).
    pub tv_threshold: f32,
}

impl Default for CondAuditConfig {
    fn default() -> Self {
        Self { tv_threshold: 0.05 }
    }
}

/// One conditioning-consistency audit over a junction set.
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Junctions measured (== `positions.len()`).
    pub junctions: usize,
    /// Total forward-KL in nats: `Σⱼ KL(teacher_j ‖ student_j)`.
    pub eps_kl: f32,
    /// Pinsker bound `sqrt(eps_kl / 2)` — exact for a single junction; the
    /// issue-prescribed verdict form.
    pub tv_bound: f32,
    /// Triangle-telescoped chain bound `sqrt(K · eps_kl / 2)` — the safe
    /// total-variation bound across a K-junction conditioning chain.
    pub tv_bound_chain: f32,
    /// `tv_bound <= cfg.tv_threshold`.
    pub verdict_pass: bool,
    /// Positions where `argmax(student) != argmax(teacher)` (first-index
    /// tie-break — the engine's pinned first-index argmax convention).
    pub greedy_flips: usize,
    /// Largest single-junction KL (nats).
    pub max_junction_kl: f32,
    /// Per-junction KL in position order (nats) — for exact per-junction
    /// Pinsker and gap-curve sweeps (Issue 719 T2).
    pub per_junction_kl: Vec<f32>,
}

/// Pinsker's inequality (nats): `TV(P, Q) <= sqrt(KL(P ‖ Q) / 2)`.
/// Unconditional for a categorical pair. Negative inputs (cannot occur from
/// [`audit_conditioning`], but defend direct callers) clamp to 0.
#[inline]
pub fn pinsker_tv_bound(eps_kl: f32) -> f32 {
    (eps_kl.max(0.0) / 2.0).sqrt()
}

/// Triangle-telescoped chain form: `sqrt(K · eps_kl / 2)`.
#[inline]
pub fn pinsker_tv_bound_chain(junctions: usize, eps_kl: f32) -> f32 {
    (junctions as f32 * eps_kl.max(0.0) / 2.0).sqrt()
}

/// Run the audit over `positions` (the junction set).
///
/// Contract: each closure writes next-token logits for one decode position
/// into `out` (`out.len() == vocab`); the two arms must cover the same
/// positions with the same vocabulary. `positions` is opaque — the decode
/// positions where the compressed view forked from the full view.
///
/// Cold-path tooling: allocates two `vocab`-sized scratch buffers per call
/// plus the report's per-junction vector. An empty position set (or zero
/// vocab) measures nothing and reports a vacuous pass — callers wanting
/// "audited" evidence must assert a non-empty junction set themselves.
pub fn audit_conditioning<S, T>(
    positions: &[u32],
    vocab: usize,
    mut student: S,
    mut teacher: T,
    cfg: &CondAuditConfig,
) -> AuditReport
where
    S: FnMut(u32, &mut [f32]),
    T: FnMut(u32, &mut [f32]),
{
    let mut report = AuditReport {
        junctions: positions.len(),
        eps_kl: 0.0,
        tv_bound: 0.0,
        tv_bound_chain: 0.0,
        verdict_pass: true,
        greedy_flips: 0,
        max_junction_kl: 0.0,
        per_junction_kl: Vec::new(),
    };
    if positions.is_empty() || vocab == 0 {
        return report;
    }
    report.per_junction_kl = Vec::with_capacity(positions.len());
    let mut s_buf = vec![0.0f32; vocab];
    let mut t_buf = vec![0.0f32; vocab];
    for &pos in positions {
        student(pos, &mut s_buf);
        teacher(pos, &mut t_buf);
        debug_assert_eq!(s_buf.len(), vocab);
        debug_assert_eq!(t_buf.len(), vocab);
        // Forward-KL direction: teacher first — see the module docs.
        let kl = kl_logits(&t_buf, &s_buf);
        report.eps_kl += kl;
        if kl > report.max_junction_kl {
            report.max_junction_kl = kl;
        }
        if argmax(&s_buf) != argmax(&t_buf) {
            report.greedy_flips += 1;
        }
        report.per_junction_kl.push(kl);
    }
    report.tv_bound = pinsker_tv_bound(report.eps_kl);
    report.tv_bound_chain = pinsker_tv_bound_chain(report.junctions, report.eps_kl);
    report.verdict_pass = report.tv_bound <= cfg.tv_threshold;
    report
}

/// First-max argmax — deterministic first-index tie-break (the engine's
/// pinned first-index argmax convention; Issue 718's oracle lesson).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(_pos: u32, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = i as f32 * 0.25 - 4.0;
        }
    }

    #[test]
    fn pinsker_known_values() {
        assert_eq!(pinsker_tv_bound(0.0), 0.0);
        assert_eq!(pinsker_tv_bound(2.0), 1.0);
        assert_eq!(pinsker_tv_bound(0.5), 0.5);
        assert_eq!(pinsker_tv_bound(-1.0), 0.0, "defensive clamp");
        assert_eq!(pinsker_tv_bound_chain(8, 1.0), 2.0);
        assert_eq!(pinsker_tv_bound_chain(0, 5.0), 0.0);
    }

    #[test]
    fn calibrated_zero_is_exactly_zero() {
        let cfg = CondAuditConfig::default();
        let r = audit_conditioning(&[1, 2, 3], 64, ident, ident, &cfg);
        assert_eq!(r.eps_kl, 0.0, "bit-identical arms sum to exactly 0.0");
        assert_eq!(r.tv_bound, 0.0);
        assert_eq!(r.greedy_flips, 0);
        assert!(r.verdict_pass);
    }

    #[test]
    fn empty_positions_is_vacuous() {
        let cfg = CondAuditConfig::default();
        let r = audit_conditioning(&[], 64, ident, ident, &cfg);
        assert_eq!(r.junctions, 0);
        assert!(
            r.verdict_pass,
            "vacuous audit passes trivially (documented)"
        );
        assert!(r.per_junction_kl.is_empty());
    }
}
