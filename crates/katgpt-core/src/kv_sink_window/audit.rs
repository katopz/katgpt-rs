//! The sink-audit vocabulary (Issue 981): one-time per-checkpoint census →
//! [`SinkAuditVerdict`] → [`SinkWindowPolicy::from_audit`].
//!
//! # What this module owns
//!
//! Only the DECISION layer. The census pass itself (loading the checkpoint,
//! running the forward, capturing post-RoPE Q rows, reading K/V from the
//! cache) lives with the model runtime — `riir-infer`'s `sink_audit` lane.
//! This module defines the plain-data profile, the chosen cut points, and the
//! policy selector those lanes call.
//!
//! # The measured signals (arXiv:2505.06708, "Gated Attention for LLMs")
//!
//! - **F-Attn** — the mean attention mass the first position receives, over
//!   full-attention layers, heads, and query positions. The paper measures
//!   CLUSTERS (sink-free 0.048–0.073 vs sunk 0.297–0.467) but no boundary.
//! - **first-N profile** — per-position mean mass over the first N positions,
//!   so secondary/delimiter sinks at positions 1–3 do not hide behind a low
//!   token-0 share.
//! - **M-Act** — mean per-layer max activation (the paper's secondary signal;
//!   measured from the cached V rows' norms). Disclosed, never verdict-bearing.
//!
//! # The cut points are CHOSEN, and stay chosen
//!
//! The paper gives clusters, not a decision boundary, so the cut points below
//! are stated as chosen (Issue 981, round-1 verdict): F-Attn ≤ 0.10 →
//! [`SinkAuditVerdict::SinkFree`]; ≥ 0.25 → [`SinkAuditVerdict::SinkHeavy`];
//! between → [`SinkAuditVerdict::Undecided`] (keep the incumbent posture,
//! disclose — an Undecided cell never folds into either side).
//!
//! # Scope law
//!
//! The audit acts on KV-EVICTION policy only, ONCE per checkpoint. It does not
//! touch the token-gate reserve (`tokengate`'s sink window + final position is
//! a correctness margin for delimiter-class sinks, not an eviction budget),
//! and it does not gate anything in the forward — Plan 287's per-call
//! classifier remains the in-engine lane; the audit can feed it as a prior
//! (composition, not replacement).

use super::policy::SinkWindowPolicy;

/// F-Attn at or below this reads [`SinkAuditVerdict::SinkFree`]. CHOSEN:
/// under the paper's measured sink-free cluster (0.048–0.073) with headroom,
/// and strictly below the sunk cluster's floor (0.297).
pub const SINKFREE_F_ATTN_MAX: f64 = 0.10;

/// F-Attn at or above this reads [`SinkAuditVerdict::SinkHeavy`]. CHOSEN:
/// at the paper's sunk-cluster floor (0.297) minus safety margin, so a
/// borderline checkpoint lands `Undecided` rather than `SinkHeavy`.
pub const SINKHEAVY_F_ATTN_MIN: f64 = 0.25;

/// A leading position still counts as an audit-selected sink pin while its
/// profile mass stays at or above this. CHOSEN: well under the sunk-cluster
/// floor, well over measurement noise; its only job is counting LEADING pins
/// for a SinkFree checkpoint, where every surviving value is tiny.
pub const AUDIT_PROFILE_SINK_MIN: f64 = 0.05;

/// The audit's floor on a shrunken sink pin. The paper's gated models still
/// carry SOME first-position mass; pinning zero positions is exactly the
/// posture T4's end-to-end cell must prove before it is allowed.
pub const AUDIT_N_SINK_FLOOR: usize = 1;

/// The census verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkAuditVerdict {
    /// F-Attn ≤ [`SINKFREE_F_ATTN_MAX`] — the checkpoint reads sink-free
    /// (the paper's gated-F-Attn signature). The pin can shrink.
    SinkFree,
    /// F-Attn ≥ [`SINKHEAVY_F_ATTN_MIN`] — a live sink; the incumbent pin
    /// stays, and long-context extension carries measured risk.
    SinkHeavy,
    /// Between the cut points — keep the incumbent posture and disclose.
    /// Never folded into either side.
    Undecided,
}

impl SinkAuditVerdict {
    /// Classify one checkpoint by its token-0 mass share.
    ///
    /// The cut points are inclusive on both sides of the free/heavy bands and
    /// the gap between them is `Undecided` — a checkpoint exactly ON a cut is
    /// claimed by the cheaper-to-prove band (free ≤, heavy ≥), never by the
    /// gap.
    #[inline]
    pub fn classify(f_attn_token0: f64) -> Self {
        if f_attn_token0 <= SINKFREE_F_ATTN_MAX {
            Self::SinkFree
        } else if f_attn_token0 >= SINKHEAVY_F_ATTN_MIN {
            Self::SinkHeavy
        } else {
            Self::Undecided
        }
    }

    /// The one long-context-risk line for a SinkHeavy verdict (Issue 981
    /// T2). `None` for every other verdict.
    pub fn risk_line(self) -> Option<&'static str> {
        match self {
            // Paper Table 5: ungated checkpoints lose 41.6 RULER at the
            // 32k→ext step vs 6.9 gated; the pin is what protects them.
            Self::SinkHeavy => Some(
                "sink-heavy checkpoint: keep the KV-eviction sink pin; position extension is measured-risk",
            ),
            Self::SinkFree | Self::Undecided => None,
        }
    }
}

/// The census profile: plain data, filled by the model runtime's census pass.
///
/// Every mass field is f64 — the census accumulates sequentially in a fixed
/// (layer, head, query) order, and the profile is the thing the G1
/// determinism claim is made on. Serialization/formatting stays with the
/// caller (the census bin owns its report bytes).
#[derive(Clone, Debug)]
pub struct SinkAuditProfile {
    /// Mean attention mass on position 0 — the F-Attn signal
    /// [`SinkAuditVerdict::classify`] reads.
    pub f_attn_token0: f64,
    /// Per-position mean mass over the first N positions
    /// (`first_n_profile.len() == N`). Position k's mean is over every
    /// (full-attention layer, head, query ≥ k) observation.
    pub first_n_profile: Vec<f64>,
    /// Mean per-layer max activation (from the cached V rows' norms).
    /// Secondary signal; disclosed, never verdict-bearing.
    pub m_act: f64,
    /// Full-attention layers scanned (the census scope — GDN layers have no
    /// softmax sink and are excluded by construction).
    pub layers_scanned: usize,
    /// Heads per layer scanned.
    pub heads_scanned: usize,
    /// Positions scanned (== `first_n_profile.len()`).
    pub positions_scanned: usize,
}

impl SinkAuditProfile {
    /// The verdict this profile classifies to.
    #[inline]
    pub fn verdict(&self) -> SinkAuditVerdict {
        SinkAuditVerdict::classify(self.f_attn_token0)
    }
}

/// What [`SinkWindowPolicy::from_audit`] decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditedSinkPolicy {
    /// `SinkFree` — use this policy: `n_sink` re-derived from the profile
    /// (leading positions at or above [`AUDIT_PROFILE_SINK_MIN`], clamped to
    /// `[`[`AUDIT_N_SINK_FLOOR`]`..= incumbent.n_sink]`), `window` carried
    /// from the incumbent. The audit may SHRINK the pin, never grow it and
    /// never touch the window — the window is the recall/lossy-axis
    /// decision (`WindowFidelity`), not the audit's.
    Audited(SinkWindowPolicy),
    /// `SinkHeavy` — the incumbent pin stays (its risk line:
    /// [`SinkAuditVerdict::risk_line`]).
    KeepIncumbentSinkHeavy,
    /// `Undecided` — the incumbent pin stays; the cell is disclosed, never
    /// folded.
    KeepIncumbentUndecided,
}

impl SinkWindowPolicy {
    /// The Issue-981 policy selector: census verdict + profile in, posture
    /// out.
    ///
    /// - `SinkFree` → [`AuditedSinkPolicy::Audited`]: `n_sink` is the number
    ///   of LEADING positions (contiguous from 0) whose profile mass stays at
    ///   or above [`AUDIT_PROFILE_SINK_MIN`], clamped to
    ///   `[`[`AUDIT_N_SINK_FLOOR`]`..= incumbent.n_sink.max(1)]` — the floor
    ///   until T4 proves 0 safe, and the audit never grows the incumbent's
    ///   pin. `window` is the incumbent's, untouched.
    /// - `SinkHeavy` / `Undecided` → the incumbent pin stands
    ///   ([`AuditedSinkPolicy::KeepIncumbentSinkHeavy`] /
    ///   [`AuditedSinkPolicy::KeepIncumbentUndecided`]).
    pub fn from_audit(
        verdict: SinkAuditVerdict,
        profile: &SinkAuditProfile,
        incumbent: SinkWindowPolicy,
    ) -> AuditedSinkPolicy {
        match verdict {
            SinkAuditVerdict::SinkHeavy => AuditedSinkPolicy::KeepIncumbentSinkHeavy,
            SinkAuditVerdict::Undecided => AuditedSinkPolicy::KeepIncumbentUndecided,
            SinkAuditVerdict::SinkFree => {
                let mut leading = 0usize;
                for &mass in &profile.first_n_profile {
                    if mass >= AUDIT_PROFILE_SINK_MIN {
                        leading += 1;
                    } else {
                        break;
                    }
                }
                let n_sink = leading
                    .max(AUDIT_N_SINK_FLOOR)
                    .min(incumbent.n_sink.max(AUDIT_N_SINK_FLOOR));
                AuditedSinkPolicy::Audited(SinkWindowPolicy {
                    n_sink,
                    window: incumbent.window,
                })
            }
        }
    }
}

/// One head's softmax-row mass accumulation — the census core.
///
/// Given one query's Q row (post-RoPE, post-QK-norm — exactly what the
/// forward feeds its attention kernel), the cached K rows for this query's
/// KV group, and the engine's scale, computes the standard softmax attention
/// row over `n_keys` keys and adds row entry `k` into `profile_acc[k]` for
/// every `k < min(n_profile, n_keys)`.
///
/// This is MEASUREMENT arithmetic, not model weighting: the model's attention
/// IS a softmax (the qwen35-hybrid kernel), so reproducing it in f32 with a
/// stable max-subtraction is what makes the census faithful. The f64
/// `profile_acc` accumulation is sequential and order-fixed by the caller
/// (layers ascending, heads ascending, queries ascending) — the G1
/// determinism contract.
///
/// `row` is the workspace, reused across calls (zero allocation at capacity —
/// the G4 contract prices the census pass, and the census is once per
/// checkpoint).
///
/// # Panics
///
/// Panics when `keys` cannot back `n_keys` rows of `head_dim`, or
/// `profile_acc` is shorter than the profile width — caller-structure errors,
/// not data-dependent ones.
pub fn accumulate_head_mass_into(
    q_row: &[f32],
    keys: &[f32],
    n_keys: usize,
    head_dim: usize,
    scale: f32,
    row: &mut Vec<f32>,
    profile_acc: &mut [f64],
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert!(
        keys.len() >= n_keys * head_dim,
        "keys len {} cannot back {n_keys} rows of {head_dim}",
        keys.len()
    );
    assert!(
        n_keys > 0,
        "a census query always sees itself: n_keys must be positive"
    );
    if row.len() < n_keys {
        row.resize(n_keys, 0.0);
    }
    row.clear();
    row.resize(n_keys, 0.0);

    // Logits: f32 sequential dot per key — the engine's own arithmetic class.
    let mut max = f32::NEG_INFINITY;
    for ki in 0..n_keys {
        let k_row = &keys[ki * head_dim..ki * head_dim + head_dim];
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q_row[d] * k_row[d];
        }
        let logit = dot * scale;
        row[ki] = logit;
        if logit > max {
            max = logit;
        }
    }

    // Stable softmax in f32, then accumulate into f64.
    let mut sum = 0.0f32;
    for v in row[..n_keys].iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    let width = n_keys.min(profile_acc.len());
    for (acc, &p) in profile_acc[..width].iter_mut().zip(row[..width].iter()) {
        *acc += f64::from(p) * f64::from(inv);
    }
}

/// Finalize the census accumulator: position k's mean mass over every
/// (layer, head, query ≥ k) observation.
///
/// By census construction every query `qi ≥ k` contributes exactly one
/// observation to position k per (layer, head) — so the count is derived,
/// never tracked separately: `n_layers * n_heads * (n_positions - k)`.
/// `acc` must be the census width (>= `n_positions`); the returned profile is
/// exactly `n_positions` long.
pub fn finalize_profile(
    acc: &[f64],
    n_layers: usize,
    n_heads: usize,
    n_positions: usize,
) -> Vec<f64> {
    assert!(
        acc.len() >= n_positions,
        "accumulator width {} < census width {n_positions}",
        acc.len()
    );
    assert!(n_layers > 0 && n_heads > 0 && n_positions > 0);
    (0..n_positions)
        .map(|k| {
            let count = (n_layers * n_heads * (n_positions - k)) as f64;
            acc[k] / count
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_of(masses: &[f64]) -> SinkAuditProfile {
        SinkAuditProfile {
            f_attn_token0: masses[0],
            first_n_profile: masses.to_vec(),
            m_act: 0.0,
            layers_scanned: 1,
            heads_scanned: 1,
            positions_scanned: masses.len(),
        }
    }

    // ── the cut points are the contract ──────────────────────────────────

    #[test]
    fn a01_classify_reproduces_the_paper_cluster_ordering() {
        // The paper's measured clusters: sink-free 0.048–0.073, sunk
        // 0.297–0.467. Both endpoints of each cluster classify with it.
        assert_eq!(
            SinkAuditVerdict::classify(0.048),
            SinkAuditVerdict::SinkFree
        );
        assert_eq!(
            SinkAuditVerdict::classify(0.073),
            SinkAuditVerdict::SinkFree
        );
        assert_eq!(
            SinkAuditVerdict::classify(0.297),
            SinkAuditVerdict::SinkHeavy
        );
        assert_eq!(
            SinkAuditVerdict::classify(0.467),
            SinkAuditVerdict::SinkHeavy
        );
        // The chosen cuts, inclusive as documented.
        assert_eq!(SinkAuditVerdict::classify(0.10), SinkAuditVerdict::SinkFree);
        assert_eq!(
            SinkAuditVerdict::classify(0.250),
            SinkAuditVerdict::SinkHeavy
        );
        // The gap is Undecided, never folded.
        assert_eq!(
            SinkAuditVerdict::classify(0.100_001),
            SinkAuditVerdict::Undecided
        );
        assert_eq!(
            SinkAuditVerdict::classify(0.249_999),
            SinkAuditVerdict::Undecided
        );
    }

    #[test]
    fn a02_risk_line_only_for_sink_heavy() {
        assert!(SinkAuditVerdict::SinkHeavy.risk_line().is_some());
        assert!(SinkAuditVerdict::SinkFree.risk_line().is_none());
        assert!(SinkAuditVerdict::Undecided.risk_line().is_none());
    }

    // ── the accumulator core ─────────────────────────────────────────────

    /// Zero Q against zero K: every logit 0, the row uniform.
    #[test]
    fn a03_zero_qk_rows_accumulate_uniform_mass() {
        let head_dim = 8usize;
        let n_keys = 5usize;
        let q = vec![0.0f32; head_dim];
        let keys = vec![0.0f32; n_keys * head_dim];
        let mut row = Vec::new();
        let mut acc = vec![0.0f64; n_keys];
        accumulate_head_mass_into(&q, &keys, n_keys, head_dim, 0.125, &mut row, &mut acc);
        let want = 1.0 / n_keys as f64;
        // The row is f32 arithmetic landing in f64: precision is f32's
        // (~1e-7), not f64's.
        for (k, &got) in acc.iter().enumerate() {
            assert!((got - want).abs() < 1e-6, "acc[{k}] = {got} want {want}");
        }
    }

    /// A planted sink: K[0] parallel to Q, everything else orthogonal — the
    /// mass collapses onto position 0.
    #[test]
    fn a04_planted_sink_collapses_mass_onto_position_zero() {
        let head_dim = 8usize;
        let n_keys = 6usize;
        let mut q = vec![0.0f32; head_dim];
        q[0] = 1.0;
        let mut keys = vec![0.0f32; n_keys * head_dim];
        keys[0] = 10.0; // position 0: strongly parallel to q — logit 10·scale
        for k in 1..n_keys {
            keys[k * head_dim + 1] = 1.0; // orthogonal to q
        }
        let mut row = Vec::new();
        let mut acc = vec![0.0f64; n_keys];
        accumulate_head_mass_into(&q, &keys, n_keys, head_dim, 1.0, &mut row, &mut acc);
        assert!(acc[0] > 0.99, "sink mass {} not collapsed", acc[0]);
        for (k, &stray) in acc.iter().enumerate().skip(1) {
            assert!(stray < 0.01, "stray mass at {k}: {stray}");
        }
    }

    /// The softmax row is a probability row: sums to 1 over the keys seen.
    #[test]
    fn a05_the_softmax_row_sums_to_one() {
        let head_dim = 4usize;
        let n_keys = 9usize;
        let q: Vec<f32> = (0..head_dim).map(|d| (d as f32 - 1.5).powi(2)).collect();
        let keys: Vec<f32> = (0..n_keys * head_dim)
            .map(|i| ((i % 7) as f32 - 3.0) / 4.0)
            .collect();
        let mut row = Vec::new();
        let mut acc = vec![0.0f64; n_keys];
        accumulate_head_mass_into(&q, &keys, n_keys, head_dim, 0.5, &mut row, &mut acc);
        let total: f64 = acc.iter().sum();
        // f32 softmax renormalization leaves ~1e-7 slack in f64.
        assert!((total - 1.0).abs() < 1e-6, "row sums to {total}");
    }

    /// G1 at the module level: the same inputs twice land bit-identical.
    #[test]
    fn a06_accumulation_is_deterministic() {
        let head_dim = 16usize;
        let n_keys = 33usize;
        let q: Vec<f32> = (0..head_dim)
            .map(|d| ((d * 7 + 3) % 11) as f32 / 11.0 - 0.5)
            .collect();
        let keys: Vec<f32> = (0..n_keys * head_dim)
            .map(|i| (((i * 13 + 5) % 17) as f32) / 17.0 - 0.5)
            .collect();
        let mut a = vec![0.0f64; n_keys];
        let mut b = vec![0.0f64; n_keys];
        let mut row = Vec::new();
        accumulate_head_mass_into(&q, &keys, n_keys, head_dim, 0.25, &mut row, &mut a);
        accumulate_head_mass_into(&q, &keys, n_keys, head_dim, 0.25, &mut row, &mut b);
        assert_eq!(a, b, "bit-identical accumulations");
    }

    /// Over-wide accumulators are untouched past the census width; short
    /// accumulator widths cap the write (the caller's width is the contract,
    /// not the key count).
    #[test]
    fn a07_accumulator_width_caps_the_write() {
        let head_dim = 4usize;
        let n_keys = 6usize;
        let q = vec![0.0f32; head_dim];
        let keys = vec![0.0f32; n_keys * head_dim];
        let mut row = Vec::new();
        let mut acc = vec![0.0f64; 3]; // narrower than n_keys
        accumulate_head_mass_into(&q, &keys, n_keys, head_dim, 0.25, &mut row, &mut acc);
        // The visible width got the first 3 entries of the uniform row.
        for &got in acc.iter().take(3) {
            assert!((got - 1.0 / 6.0).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "cannot back")]
    fn a08_short_keys_panic_loud() {
        let q = vec![0.0f32; 4];
        let keys = vec![0.0f32; 4]; // one row, but n_keys says 3
        let mut row = Vec::new();
        let mut acc = vec![0.0f64; 3];
        accumulate_head_mass_into(&q, &keys, 3, 4, 0.25, &mut row, &mut acc);
    }

    // ── finalize ─────────────────────────────────────────────────────────

    #[test]
    fn a09_finalize_divides_by_the_derived_observation_count() {
        // One layer, one head, 4 positions: query qi contributes 1/(qi+1) to
        // every position ≤ qi (the uniform case). Position 0 sees queries
        // 0..4, position 3 sees only query 3.
        let mut acc = vec![0.0f64; 4];
        for qi in 0..4usize {
            for slot in acc.iter_mut().take(qi + 1) {
                *slot += 1.0 / (qi + 1) as f64;
            }
        }
        let profile = finalize_profile(&acc, 1, 1, 4);
        assert!((profile[0] - (1.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0).abs() < 1e-12);
        assert!((profile[3] - 0.25).abs() < 1e-12);
    }

    // ── the policy selector ──────────────────────────────────────────────

    #[test]
    fn a10_sink_free_shrinks_to_the_leading_mass_with_the_floor() {
        let incumbent = SinkWindowPolicy::new(4, 256);
        // profile[0] above the counting cut, profile[1] below → one pin.
        let p = profile_of(&[0.06, 0.01, 0.005, 0.004]);
        match SinkWindowPolicy::from_audit(SinkAuditVerdict::SinkFree, &p, incumbent) {
            AuditedSinkPolicy::Audited(policy) => {
                assert_eq!(policy.n_sink, 1, "the floor is load-bearing");
                assert_eq!(policy.window, 256, "the window is the incumbent's");
            }
            other => panic!("wanted an audited policy, got {other:?}"),
        }
    }

    #[test]
    fn a11_two_leading_positions_make_two_pins() {
        let incumbent = SinkWindowPolicy::new(4, 256);
        let p = profile_of(&[0.06, 0.055, 0.01, 0.01]);
        match SinkWindowPolicy::from_audit(SinkAuditVerdict::SinkFree, &p, incumbent) {
            AuditedSinkPolicy::Audited(policy) => assert_eq!(policy.n_sink, 2),
            other => panic!("wanted an audited policy, got {other:?}"),
        }
    }

    #[test]
    fn a12_a_tiny_incumbent_cannot_be_grown_by_the_audit() {
        // A profile whose leading mass extends past a 1-pin incumbent: the
        // audit may shrink, never grow — the clamp binds at the incumbent.
        let incumbent = SinkWindowPolicy::new(1, 128);
        let p = profile_of(&[0.06, 0.06, 0.06, 0.01]);
        match SinkWindowPolicy::from_audit(SinkAuditVerdict::SinkFree, &p, incumbent) {
            AuditedSinkPolicy::Audited(policy) => assert_eq!(policy.n_sink, 1),
            other => panic!("wanted an audited policy, got {other:?}"),
        }
    }

    #[test]
    fn a13_heavy_and_undecided_keep_the_incumbent() {
        let incumbent = SinkWindowPolicy::new(4, 256);
        let p = profile_of(&[0.40, 0.02, 0.01, 0.01]);
        assert_eq!(
            SinkWindowPolicy::from_audit(SinkAuditVerdict::SinkHeavy, &p, incumbent),
            AuditedSinkPolicy::KeepIncumbentSinkHeavy
        );
        let mid = profile_of(&[0.18, 0.02, 0.01, 0.01]);
        assert_eq!(
            SinkWindowPolicy::from_audit(SinkAuditVerdict::Undecided, &mid, incumbent),
            AuditedSinkPolicy::KeepIncumbentUndecided
        );
    }

    #[test]
    fn a14_profile_verdict_delegates_to_classify() {
        let p = profile_of(&[0.05]);
        assert_eq!(p.verdict(), SinkAuditVerdict::SinkFree);
    }
}
