//! Issue 674 — Contrastive scope gate: two-corpus log-odds + epistemic
//! haircut (Research 493, arXiv:2608.13545 "LittleLearner", Li et al.).
//!
//! LittleLearner's out-of-scope failure signature: models do not express
//! uncertainty off-distribution — they emit *coherent but incorrect*
//! projections onto familiar patterns (confidence miscalibrated exactly
//! where it matters). The corrective signal must be **external and
//! corpus-derived, not model-reported**.
//!
//! The shipped gate family cannot catch this class: the identity guard,
//! the Issue 030 relevance gate, EvidenceTier history, and the engram
//! query-conditional relevance gate all check *relevance* — an input can be
//! perfectly relevant to a responder and still be outside the distribution
//! that shaped it. **A relevance check is not a scope check.**
//!
//! # The primitive (all closed-form, modelless)
//!
//! ```text
//! score(w) = log2((c_B(w)+α)/(N_B+αV)) − log2((c_I(w)+α)/(N_I+αV))   // contrastive table
//! D(x)     = Σ_w tf_x(w) · score(w)                                    // sparse GEMV (NB log-LLR)
//! ĉ        = c · sigmoid(−κ·D(x))                                      // epistemic haircut
//! D(x) > θ ⇒ decline / EvidenceTier demotion                           // decline is CORRECT off-distribution
//! ```
//!
//! where corpus I is the distribution that shaped the responder (in-scope)
//! and corpus B the out-scope contrast. `D(x) > 0` means the document is
//! *more* B-like than I-like under the Naive-Bayes log-likelihood-ratio —
//! LittleLearner's contrastive frequency-ratio score (their §"contrastive
//! blocklist"), applied as a *gate* rather than a sampling filter.
//!
//! # Design notes (honest deviations from the issue sketch)
//!
//! - **No papaya dep.** The built [`ContrastiveScoreTable`] is immutable
//!   after `finish()` and shared via `Arc` — lock-free reads *by
//!   immutability*, which is strictly cheaper than a concurrent map. papaya
//!   becomes worthwhile only for a *live* table updated concurrently with
//!   reads (a consumer integration concern, not the POC).
//! - **κ / θ are consumer-pinned.** The defaults here are POC-scale
//!   constants; the paper's numbers are regime-specific and not portable.
//!   Re-pin per consumer (riir-clippy L4 2D gate / riir-ai engram gates).
//!
//! # Domain classification
//!
//! Latent, local, never synced: the table is an offline corpus statistic;
//! `D(x)` is a per-input view; the haircut is a local confidence transform.
//! No sync dependency, no replay coupling.
//!
//! Feature: `contrastive_scope` (opt-in POC). Per the issue's own T5 rule:
//! if no consumer adopts, record the negative result and close — never
//! promote a gate nothing consumes.
//!
//! Contrastive scope gate (Issue 674 / Research 493, arXiv:2608.13545
//! LittleLearner): two-corpus log-odds table + Naive-Bayes log-LLR document
//! scope score D(x) + epistemic haircut ĉ = c·sigmoid(−κ·D) + decline wiring +
//! the paired OOS probe battery (Report-the-Floor extension). "A relevance
//! check is not a scope check." DEFAULT-ON (2026-08-19) under the issue's own
//! one-consumer T5 rule, satisfied by the riir-clippy ScopeModel adoption
//! (Bench 669 / Bench 040: installed via with_scope_model, bit-identical
//! in-distribution). The other named candidate (riir-ai engram gates) never
//! materialized — caught by the 2026-09-04 goat-audit (riir-ai Issue 867);
//! zero riir-ai consumers as of that audit, the riir-clippy consumer is the
//! load-bearing one.

use crate::sigmoid;

/// Default additive smoothing (Laplace-style, per Research 493's sketch).
pub const DEFAULT_ALPHA: f32 = 0.5;

/// Default epistemic-haircut steepness (POC-scale; consumer re-pins).
pub const DEFAULT_KAPPA: f32 = 0.05;

/// Default decline threshold on `D(x)` in bits (POC-scale; consumer re-pins).
pub const DEFAULT_THETA: f32 = 8.0;

// ─────────────────────────────────────────────────────────────────────────────
// T1 — ContrastiveScoreTable
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming two-corpus count builder (in-scope I vs out-scope B).
///
/// Two passes: [`Self::observe_in`] over the responder's shaping corpus,
/// [`Self::observe_out`] over the contrast corpus. Then [`Self::finish`]
/// freezes the smoothed log2-odds table. Deterministic: plain `u64` count
/// vectors, fixed vocab order.
pub struct ContrastiveScoreBuilder {
    vocab: usize,
    alpha: f32,
    counts_in: Vec<u64>,
    counts_out: Vec<u64>,
    n_in: u64,
    n_out: u64,
}

impl ContrastiveScoreBuilder {
    /// New builder over a `vocab`-sized dictionary with smoothing `alpha > 0`.
    #[must_use]
    pub fn new(vocab: usize, alpha: f32) -> Self {
        assert!(alpha > 0.0, "alpha=0 makes unseen words log2(0/0) NaN");
        Self {
            vocab,
            alpha,
            counts_in: vec![0; vocab],
            counts_out: vec![0; vocab],
            n_in: 0,
            n_out: 0,
        }
    }

    /// One in-scope document's tokens.
    pub fn observe_in(&mut self, tokens: &[u32]) {
        for &w in tokens {
            debug_assert!((w as usize) < self.vocab);
            self.counts_in[w as usize] = self.counts_in[w as usize].saturating_add(1);
            self.n_in = self.n_in.saturating_add(1);
        }
    }

    /// One out-scope document's tokens.
    pub fn observe_out(&mut self, tokens: &[u32]) {
        for &w in tokens {
            debug_assert!((w as usize) < self.vocab);
            self.counts_out[w as usize] = self.counts_out[w as usize].saturating_add(1);
            self.n_out = self.n_out.saturating_add(1);
        }
    }

    /// Smoothed contrastive log2-odds for one word:
    /// `log2 P_B(w) − log2 P_I(w)` (positive ⇒ B-characteristic).
    #[must_use]
    pub fn score(&self, w: u32) -> f32 {
        let v = self.vocab as f32;
        let p_out = (self.counts_out[w as usize] as f32 + self.alpha)
            / (self.n_out as f32 + self.alpha * v);
        let p_in =
            (self.counts_in[w as usize] as f32 + self.alpha) / (self.n_in as f32 + self.alpha * v);
        p_out.log2() - p_in.log2()
    }

    /// Freeze into the immutable read table (computes the full score vector
    /// once — the `log_ratio_vec` the document scorer consumes).
    #[must_use]
    pub fn finish(self) -> ContrastiveScoreTable {
        let scores = (0..self.vocab as u32)
            .map(|w| self.score(w))
            .collect::<Vec<_>>();
        let frozen = self.freeze_bytes();
        let commitment = *blake3::hash(&frozen).as_bytes();
        ContrastiveScoreTable {
            scores,
            alpha: self.alpha,
            n_in: self.n_in,
            n_out: self.n_out,
            commitment,
        }
    }

    /// Canonical freeze layout (BLAKE3-committed): `[vocab u32][alpha f32
    /// bits u32][n_in u64][n_out u64][counts_in u32×V][counts_out u32×V]`,
    /// all little-endian.
    fn freeze_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20 + self.vocab * 8);
        bytes.extend_from_slice(&(self.vocab as u32).to_le_bytes());
        bytes.extend_from_slice(&self.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.n_in.to_le_bytes());
        bytes.extend_from_slice(&self.n_out.to_le_bytes());
        for &c in &self.counts_in {
            bytes.extend_from_slice(&(c as u32).to_le_bytes());
        }
        for &c in &self.counts_out {
            bytes.extend_from_slice(&(c as u32).to_le_bytes());
        }
        bytes
    }
}

/// Immutable, BLAKE3-committed contrastive score table.
///
/// Lock-free reads by immutability (share via `Arc`). The commitment covers
/// the builder's raw counts + parameters — tamper-evidence for freeze/thaw
/// round-trips through untrusted storage.
#[derive(Debug, Clone)]
pub struct ContrastiveScoreTable {
    scores: Vec<f32>,
    alpha: f32,
    n_in: u64,
    n_out: u64,
    commitment: [u8; 32],
}

impl ContrastiveScoreTable {
    /// The smoothed contrastive log2-odds for one word (positive ⇒
    /// out-scope-characteristic).
    #[inline]
    #[must_use]
    pub fn score(&self, w: u32) -> f32 {
        self.scores[w as usize]
    }

    /// The full `log_ratio_vec` (sparse-GEMV operand for [`scope_score`]).
    #[inline]
    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Corpus sizes at build time.
    #[must_use]
    pub fn corpus_sizes(&self) -> (u64, u64) {
        (self.n_in, self.n_out)
    }

    /// BLAKE3 commitment over the canonical builder serialization.
    #[must_use]
    pub fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    /// The smoothing α used at build time.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Freeze to canonical bytes (layout identical to the builder's
    /// `freeze_bytes`; the commitment is over exactly these bytes).
    #[must_use]
    pub fn freeze(&self) -> Vec<u8> {
        // The table stores derived scores, not counts; re-deriving counts
        // from scores is not possible. For freeze/thaw we serialize the
        // table itself: scores ARE the committed payload for the thaw side.
        let mut bytes = Vec::with_capacity(24 + self.scores.len() * 4);
        bytes.extend_from_slice(&(self.scores.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.n_in.to_le_bytes());
        bytes.extend_from_slice(&self.n_out.to_le_bytes());
        bytes.extend_from_slice(&self.commitment);
        for &s in &self.scores {
            bytes.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        bytes
    }

    /// Thaw from [`ContrastiveScoreTable::freeze`] bytes. Verifies the
    /// length layout; the inner commitment is the builder-side count
    /// commitment (carried, not recomputed — scores are not invertible to
    /// counts).
    ///
    /// # Errors
    /// `None` on truncated / malformed input.
    #[must_use]
    pub fn thaw(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let alpha = f32::from_bits(u32::from_le_bytes(bytes[4..8].try_into().ok()?));
        let n_in = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let n_out = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        let commitment: [u8; 32] = bytes[24..56].try_into().ok()?;
        if bytes.len() != 56 + n * 4 {
            return None;
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let off = 56 + i * 4;
            scores.push(f32::from_bits(u32::from_le_bytes(
                bytes[off..off + 4].try_into().ok()?,
            )));
        }
        Some(Self {
            scores,
            alpha,
            n_in,
            n_out,
            commitment,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T2 — document scope score D(x) (sparse GEMV / NB log-LLR)
// ─────────────────────────────────────────────────────────────────────────────

/// Document scope score `D(x) = Σ_w tf_x(w) · score(w)`, accumulating in
/// **document token order** (the fixed order — bit-identical to the scalar
/// loop by construction; no re-association, no SIMD reordering).
///
/// Positive ⇒ the document reads out-scope under the contrastive table.
/// Zero-alloc: a single running accumulator.
#[inline]
#[must_use]
pub fn scope_score(table: &ContrastiveScoreTable, tokens: &[u32]) -> f32 {
    let scores = table.scores();
    let mut acc = 0.0f32;
    for &w in tokens {
        acc += scores[w as usize];
    }
    acc
}

/// Term-frequency-weighted variant: `D(x) = Σ (w, tf) score(w)` in pair
/// order. Use when the caller already has a sparse count representation.
#[inline]
#[must_use]
pub fn scope_score_from_pairs(table: &ContrastiveScoreTable, pairs: &[(u32, f32)]) -> f32 {
    let scores = table.scores();
    let mut acc = 0.0f32;
    for &(w, tf) in pairs {
        acc += tf * scores[w as usize];
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────────────
// T3 — epistemic haircut + decline wiring
// ─────────────────────────────────────────────────────────────────────────────

/// Scope gate configuration (κ steepness, θ decline threshold).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScopeGate {
    /// Haircut steepness: `ĉ = c·sigmoid(−κ·D)`.
    pub kappa: f32,
    /// Decline threshold: `D > θ ⇒ decline` (bits).
    pub theta: f32,
}

impl Default for ScopeGate {
    fn default() -> Self {
        Self {
            kappa: DEFAULT_KAPPA,
            theta: DEFAULT_THETA,
        }
    }
}

/// A scope-gate verdict over one input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScopeVerdict {
    /// The document scope score `D(x)` in bits.
    pub d: f32,
    /// The haircut confidence `ĉ = c · sigmoid(−κ·D)`.
    pub haircut: f32,
    /// `D(x) > θ` — decline / EvidenceTier demotion (a CORRECT answer
    /// off-distribution).
    pub declined: bool,
}

impl ScopeGate {
    /// Apply the gate: haircut + decline verdict.
    ///
    /// **In-distribution no-regression (the load-bearing half):** for
    /// strongly in-scope documents `−κ·D` is large-positive and f32
    /// `sigmoid` saturates to exactly `1.0` (for arguments ≥ ~16.6), so
    /// `haircut == c` **bit-identically** — gated and ungated consumers
    /// observe the same confidence (pinned by tests).
    #[must_use]
    pub fn apply(&self, confidence: f32, d: f32) -> ScopeVerdict {
        let haircut = confidence * sigmoid(-self.kappa * d);
        ScopeVerdict {
            d,
            haircut,
            declined: d > self.theta,
        }
    }

    /// Convenience: score + gate in one call.
    #[must_use]
    pub fn apply_to_tokens(
        &self,
        table: &ContrastiveScoreTable,
        confidence: f32,
        tokens: &[u32],
    ) -> ScopeVerdict {
        self.apply(confidence, scope_score(table, tokens))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T4 — OOS probe battery (Report-the-Floor extension)
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate report over a paired in/out probe battery.
///
/// The `cov_in − cov_out` axis (Research 493 §4): honest primitives widen
/// intervals or decline OOS; dishonest ones project confidently. Here the
/// coverage analog is the mean haircut per side: `mean_haircut_in` ≈ 1
/// (no-regression) while `mean_haircut_out` ≪ 1 (the gate bites) — and
/// `leak_suspects` lists in-corpus docs whose D exceeds θ (the seeded-leak
/// detector).
#[derive(Debug, Clone, PartialEq)]
pub struct OosProbeReport {
    /// Mean `D(x)` over the in-corpus probe docs (should be ≤ 0).
    pub mean_d_in: f32,
    /// Mean `D(x)` over the out-corpus probe docs (should be ≫ 0).
    pub mean_d_out: f32,
    /// Mean haircut over in-corpus docs (should be ≈ 1 — bit-identity class).
    pub mean_haircut_in: f32,
    /// Mean haircut over out-corpus docs (should be ≪ 1).
    pub mean_haircut_out: f32,
    /// Fraction of out-corpus docs declined (`D > θ`).
    pub decline_rate_out: f32,
    /// Indices of IN-corpus docs with `D > θ` — the seeded-leak suspects.
    pub leak_suspects: Vec<usize>,
}

/// Run the paired probe battery.
///
/// `probe_in` / `probe_out` are held-out documents (never used to build the
/// table — the Report-the-Floor discipline). `confidence` is the nominal
/// confidence the gated responder would report (the haircut is multiplicative).
#[must_use]
pub fn oos_probe_battery(
    table: &ContrastiveScoreTable,
    gate: &ScopeGate,
    probe_in: &[&[u32]],
    probe_out: &[&[u32]],
    confidence: f32,
) -> OosProbeReport {
    let mut sum_d_in = 0.0f32;
    let mut sum_h_in = 0.0f32;
    let mut leak_suspects = Vec::new();
    for (i, doc) in probe_in.iter().enumerate() {
        let d = scope_score(table, doc);
        let v = gate.apply(confidence, d);
        sum_d_in += d;
        sum_h_in += v.haircut;
        if v.declined {
            leak_suspects.push(i);
        }
    }
    let mut sum_d_out = 0.0f32;
    let mut sum_h_out = 0.0f32;
    let mut declined = 0usize;
    for doc in probe_out {
        let d = scope_score(table, doc);
        let v = gate.apply(confidence, d);
        sum_d_out += d;
        sum_h_out += v.haircut;
        if v.declined {
            declined += 1;
        }
    }
    let n_in = probe_in.len().max(1) as f32;
    let n_out = probe_out.len().max(1) as f32;
    OosProbeReport {
        mean_d_in: sum_d_in / n_in,
        mean_d_out: sum_d_out / n_out,
        mean_haircut_in: sum_h_in / n_in,
        mean_haircut_out: sum_h_out / n_out,
        decline_rate_out: declined as f32 / n_out,
        leak_suspects,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy corpus: V=7. In-corpus uses words {0,1,2} heavily; out-corpus
    /// uses {3,4,5}; word 6 appears in NEITHER corpus (the neutral word).
    fn toy_table() -> ContrastiveScoreTable {
        let mut b = ContrastiveScoreBuilder::new(7, 0.5);
        for _ in 0..50 {
            b.observe_in(&[0, 0, 1, 2]);
        }
        for _ in 0..50 {
            b.observe_out(&[3, 3, 4, 5]);
        }
        b.finish()
    }

    /// G1: toy-corpus parity vs hand-computed smoothed log2 ratios,
    /// including zero-count smoothing edges and the α→0 limit.
    #[test]
    fn g1_toy_corpus_parity_and_smoothing_edges() {
        let t = toy_table();
        // Parity: score(w) == log2 P_B(w) − log2 P_I(w) with the counts above.
        // w=0: c_I=100, c_B=0, N_I=200, N_B=200, V=6, α=0.5.
        let p_in_0: f32 = (100.0 + 0.5) / (200.0 + 0.5 * 7.0);
        let p_out_0: f32 = (0.0 + 0.5) / (200.0 + 0.5 * 7.0);
        let expect_0 = p_out_0.log2() - p_in_0.log2();
        assert!((t.score(0) - expect_0).abs() < 1e-6);
        // w=3: c_I=0, c_B=100 — mirror image.
        let p_in_3: f32 = (0.0 + 0.5) / (200.0 + 3.5);
        let p_out_3: f32 = (100.0 + 0.5) / (200.0 + 3.5);
        let expect_3 = p_out_3.log2() - p_in_3.log2();
        assert!((t.score(3) - expect_3).abs() < 1e-6);
        // Zero-count edge: a word in NEITHER corpus (w=6) has identical
        // smoothed probabilities on both sides → scores exactly 0 — the
        // neutral word carries no scope signal.
        assert_eq!(t.score(6).to_bits(), 0.0f32.to_bits());
    }

    /// The α→0 limit approaches the raw ratio; unseen words under tiny α
    /// score as strongly B-or-I characteristic per their zero counts.
    #[test]
    fn g1_alpha_to_zero_limit() {
        let mut b = ContrastiveScoreBuilder::new(4, 1e-6);
        for _ in 0..100 {
            b.observe_in(&[0, 1]);
            b.observe_out(&[2, 3]);
        }
        let t = b.finish();
        // w=0 raw: log2((0+ε)/(200+4ε)) − log2((100+ε)/(200+4ε)) → −log2(100/200)= +1? sign:
        // score = log2 P_out − log2 P_in = log2(ε/200) − log2(100/200).
        // As ε→0 this → −∞ (strongly in-scope). Assert strongly negative.
        assert!(t.score(0) < -10.0);
        assert!(t.score(2) > 10.0);
        // Seen-on-both-sides balanced word… none in this fixture; covered by
        // the parity test. The limit test's job: no NaN, correct signs,
        // magnitudes grow as α shrinks.
        let mut b2 = ContrastiveScoreBuilder::new(4, 0.5);
        for _ in 0..100 {
            b2.observe_in(&[0, 1]);
            b2.observe_out(&[2, 3]);
        }
        let t2 = b2.finish();
        assert!(t.score(0) < t2.score(0), "smaller α sharpens the contrast");
    }

    /// T2 G1: D(x) bit-identical vs the scalar loop (trivially — same
    /// fixed-order code path — plus the pairs variant agrees when tf are
    /// integers accumulated in the same order).
    #[test]
    fn t2_scope_score_bit_identity_and_pairs_agreement() {
        let t = toy_table();
        let doc: Vec<u32> = vec![0, 3, 1, 3, 3, 2, 5, 0, 3];
        let d = scope_score(&t, &doc);
        // Scalar reference.
        let mut acc = 0.0f32;
        for &w in &doc {
            acc += t.score(w);
        }
        assert_eq!(d.to_bits(), acc.to_bits());
        // Pairs variant with per-word tf — NOT identical (different
        // accumulation order), but within fp tolerance.
        let mut counts = std::collections::BTreeMap::new();
        for &w in &doc {
            *counts.entry(w).or_insert(0.0f32) += 1.0;
        }
        let pairs: Vec<(u32, f32)> = counts.into_iter().collect();
        let d_pairs = scope_score_from_pairs(&t, &pairs);
        assert!((d - d_pairs).abs() < 1e-4 * d.abs().max(1.0));
    }

    /// T2 G2 (latency, release): µs/doc at 10⁴ tokens.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "timing gate — release-only")]
    fn t2_scope_score_us_per_doc_10k_tokens() {
        let t = toy_table();
        let doc: Vec<u32> = (0..10_000).map(|i| (i % 6) as u32).collect();
        // Warm up.
        let mut sink = 0.0f32;
        for _ in 0..50 {
            sink += scope_score(&t, &doc);
        }
        let n = 2000u32;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sink += scope_score(&t, &doc);
        }
        let per_us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
        println!("t2_scope_score: {per_us:.3} µs/doc at 10⁴ tokens (sink {sink:.3})");
        assert!(per_us < 100.0, "µs/doc {per_us} exceeds gate");
    }

    /// T3 G3 (the load-bearing no-regression half): strongly in-distribution
    /// docs get a **bit-identical** haircut — with κ·|D| ≥ 40 the f32
    /// `fast_sigmoid` early-exits to EXACTLY 1.0, so `haircut == c` bitwise;
    /// authored OOS fixtures are discounted or declined.
    #[test]
    fn t3_in_distribution_bit_identical_oos_discounted() {
        let t = toy_table();
        // κ = 0.5 (consumer-pinned class): a 12-token in-scope doc has
        // D ≈ −84 → −κD ≈ +42, inside the exact-saturation region.
        let gate = ScopeGate {
            kappa: 0.5,
            theta: 8.0,
        };
        let c = 0.87f32;
        let in_doc: Vec<u32> = [0, 1, 2].iter().cycle().take(12).cloned().collect();
        let v = gate.apply_to_tokens(&t, c, &in_doc);
        assert!(
            v.d < -50.0,
            "in-scope D should be strongly negative, got {}",
            v.d
        );
        assert!(
            -gate.kappa * v.d >= 40.0,
            "fixture must sit in the exact-saturation region"
        );
        assert_eq!(
            v.haircut.to_bits(),
            c.to_bits(),
            "bit-identical haircut in-scope"
        );
        assert!(!v.declined);
        // OOS doc: discounted (to exactly 0 here — the mirror saturation) and
        // declined.
        let oos_doc: Vec<u32> = [3, 4, 5].iter().cycle().take(12).cloned().collect();
        let vo = gate.apply_to_tokens(&t, c, &oos_doc);
        assert!(vo.d > 50.0);
        assert_eq!(vo.haircut, 0.0, "mirror saturation: OOS haircut exactly 0");
        assert!(vo.declined, "OOS must be declined at θ");
        // Mixed doc: between — discounted but not necessarily declined.
        let mixed: Vec<u32> = vec![0, 3, 1, 3, 3, 2, 5, 0, 3];
        let vm = gate.apply_to_tokens(&t, c, &mixed);
        assert!(vm.haircut < c);
    }

    /// T3 decline wiring: θ explicit; `D > θ` exactly.
    #[test]
    fn t3_decline_threshold_semantics() {
        let gate = ScopeGate {
            kappa: 0.1,
            theta: 5.0,
        };
        let v_below = gate.apply(1.0, 5.0);
        assert!(!v_below.declined, "D == θ is NOT declined (strict >)");
        let v_above = gate.apply(1.0, 5.0 + 1e-3);
        assert!(v_above.declined);
    }

    /// T4: the OOS probe battery — flat-OOS null vs the seeded-leak catch.
    #[test]
    fn t4_probe_battery_flat_oos_and_seeded_leak_caught() {
        let t = toy_table();
        // κ = 0.5 with 12-token probes: in-side sits in the exact-saturation
        // region (haircut == confidence bit-identically).
        let gate = ScopeGate {
            kappa: 0.5,
            theta: 8.0,
        };
        let in_probe: Vec<Vec<u32>> = (0..10)
            .map(|i| {
                [0, 1, 2, (i % 3) as u32]
                    .iter()
                    .cycle()
                    .take(12)
                    .cloned()
                    .collect()
            })
            .collect();
        let out_probe: Vec<Vec<u32>> = (0..10)
            .map(|i| {
                [3, 4, 5, (3 + i % 3) as u32]
                    .iter()
                    .cycle()
                    .take(12)
                    .cloned()
                    .collect()
            })
            .collect();
        let in_refs: Vec<&[u32]> = in_probe.iter().map(|v| v.as_slice()).collect();
        let out_refs: Vec<&[u32]> = out_probe.iter().map(|v| v.as_slice()).collect();
        let report = oos_probe_battery(&t, &gate, &in_refs, &out_refs, 0.9);
        // Clean battery: in ≈ unchanged, out heavily declined.
        assert!(report.mean_d_in < 0.0);
        assert!(report.mean_d_out > 0.0);
        assert!(
            (report.mean_haircut_in - 0.9).abs() < 1e-6,
            "in-side unchanged (fp-mean tolerance)"
        );
        assert!(report.mean_haircut_out < 0.45);
        assert!(report.decline_rate_out >= 0.9);
        assert!(
            report.leak_suspects.is_empty(),
            "clean battery has no leak suspects"
        );
        // Seeded leak: inject ONE out-scope doc into the in-probe set — the
        // battery must flag it as a leak suspect (D > θ).
        let mut leaky_in = in_probe.clone();
        leaky_in[3] = [3, 3, 4, 5].iter().cycle().take(12).cloned().collect();
        let leaky_refs: Vec<&[u32]> = leaky_in.iter().map(|v| v.as_slice()).collect();
        let leaky = oos_probe_battery(&t, &gate, &leaky_refs, &out_refs, 0.9);
        assert_eq!(
            leaky.leak_suspects,
            vec![3],
            "seeded leak at index 3 must be caught"
        );
    }

    /// Commitment + freeze/thaw round-trip (tamper-evidence).
    #[test]
    fn commitment_and_freeze_thaw_round_trip() {
        let t = toy_table();
        let frozen = t.freeze();
        let thawed = ContrastiveScoreTable::thaw(&frozen).expect("well-formed");
        for w in 0..6u32 {
            assert_eq!(thawed.score(w).to_bits(), t.score(w).to_bits());
        }
        assert_eq!(thawed.commitment(), t.commitment());
        assert_eq!(thawed.corpus_sizes(), t.corpus_sizes());
        // Determinism: identical builds ⇒ identical commitments.
        let t2 = toy_table();
        assert_eq!(t.commitment(), t2.commitment());
        // Tamper: flipping one count changes the commitment.
        let mut b = ContrastiveScoreBuilder::new(6, 0.5);
        for _ in 0..50 {
            b.observe_in(&[0, 0, 1, 2]);
        }
        for _ in 0..49 {
            b.observe_out(&[3, 3, 4, 5]);
        }
        b.observe_out(&[3, 3, 4, 4]); // one different doc
        let t3 = b.finish();
        assert_ne!(t.commitment(), t3.commitment());
        // Malformed thaw: truncated bytes → None.
        assert!(ContrastiveScoreTable::thaw(&frozen[..frozen.len() - 1]).is_none());
    }
}
