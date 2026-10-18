//! Consumption-weight evidence tripwire (Issue 837 / riir-ai Research 359 /
//! Plan 561 — the D-SCAN transliteration, arXiv:2608.06947, SIGIR'26).
//!
//! Detector family: the cgsp `entropy_nats` collapse detector consumes OUTPUT
//! priorities; this primitive is the same family with a CONSUMPTION-WEIGHT
//! input vector — per-source σ-gate weights (e.g. the engram fusion gates
//! from [`crate::engram::sigmoid_fuse_into`]) paired with the retrieval
//! scores that admitted them.
//!
//! # Measured mechanism (riir-ai Bench 832 — the falsifiable PoC verdict)
//!
//! The σ gate is NON-competitive (sigmoid, not softmax): every uncorrelated
//! source pins at `σ(0) = 0.5`, so normalized gate entropy sits in a narrow
//! ~0.977–1.000 band across every world shape — benign AND adversarial. The
//! entropy axis is additionally COMPOSITION-COUPLED: an H-only threshold
//! fires on benign-shaped worlds at the same rate it fires on hijacks (it
//! responds to how many high-gate sources the world contains, not to the
//! poison). The **rank-inversion** channel carries the discrimination: the
//! D-SCAN N=1 statistic — the retrieval rank of the top-consumed source —
//! detected adversarial injection at 100% with 1.7% split-conformal benign
//! FPR (α = 5%) while benign single-source collapse (the legitimate-collapse
//! control) never fired, across three gate temperatures including the
//! shipped engram default τ = √D.
//!
//! # Scope (honest)
//!
//! - The dual-optimized adversary (poison optimal on BOTH the consumption
//!   and retrieval channels) collapses into the benign single-source
//!   signature — the tripwire sees the construction-cost asymmetry, not a
//!   law of nature. Diffuse equal-cosine utility poison (the Bench-656
//!   regime-A table shape) does not top-gate and is likewise invisible to
//!   the rank channel; that axis belongs to the engram privilege ledger
//!   (`engram_privilege`, the repair half of the detector+repair split).
//! - This module ships METRICS + the split-conformal calibration primitive
//!   only. Threshold policy stays consumer-side: calibrate on YOUR benign
//!   pool (exchangeability is the conformal guarantee's precondition), then
//!   compare [`TripwireMetrics::normalized_top1_rank`] against the
//!   [`conformal_threshold`] of that pool's scores.
//!
//! Zero-allocation by construction: every function operates on
//! caller-owned buffers; nothing in this module allocates.
//!
//! Issue 837 / riir-ai Research 359 — consumption-weight evidence tripwire
//! (D-SCAN transliteration): σ-gate metrics + Kendall τ + split-conformal
//! benign-quantile threshold. Measured verdict (riir-ai Bench 832): the rank-
//! inversion channel discriminates, entropy is composition-coupled telemetry.

/// Default Kendall tie tolerance. Filler σ-gates pin at exactly `σ(0) = 0.5`
/// (and numerically-tied gates differ by ~1e-7), so ties must be excluded
/// from BOTH the numerator and the denominator of τ — an absolute tolerance
/// on the ~0.5-scale axes does that without rank plumbing.
pub const DEFAULT_TIE_EPS: f32 = 1e-4;

/// Internal strict-greater tolerance for retrieval-rank counting (keeps
/// near-equal retrieval scores from inflating a source's rank).
const RANK_TIE_EPS: f32 = 1e-9;

/// Kendall rank correlation (tau-a) with an absolute tie tolerance.
///
/// Pairs where either axis differs by less than `tie_eps` contribute zero to
/// the numerator AND are excluded from the denominator (the house convention
/// from the katgpt-rs diversity/temp harness). Returns 0.0 when fewer than
/// two non-tied pairs exist. O(n²) — intended for the small consumed-set K
/// (the engram seam's K_MAX = 16).
pub fn kendall_tau_a(x: &[f32], y: &[f32], tie_eps: f32) -> f32 {
    let n = x.len().min(y.len());
    let mut c = 0.0f64;
    let mut d = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            if dx.abs() < tie_eps || dy.abs() < tie_eps {
                continue;
            }
            if (dx > 0.0) == (dy > 0.0) {
                c += 1.0;
            } else {
                d += 1.0;
            }
        }
    }
    let den = c + d;
    if den == 0.0 {
        0.0
    } else {
        ((c - d) / den) as f32
    }
}

/// Per-query tripwire metrics over the consumed source set.
///
/// `retrieval` = the scores that admitted the sources (BM25 / `ShardIndex`-
/// style, any monotone scale); `gates` = the per-source consumption weights
/// (the σ-gate outputs, strictly positive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TripwireMetrics {
    /// Consumed-set size K (`retrieval.len() == gates.len()`).
    pub n: usize,
    /// Normalized consumption-weight entropy `H(p)/ln K` over the
    /// softmax-free normalized gates. Uniform ⇒ 1.0. NOTE the measured
    /// σ-floor finding: on sigmoid-gate seams this saturates near 1.0 for
    /// every world shape — use it as telemetry, not as the discriminator.
    pub h_norm: f32,
    /// Top-1 consumption share `max p_k`.
    pub top1_share: f32,
    /// Kendall τ between retrieval scores and consumption gates (ties
    /// excluded per [`DEFAULT_TIE_EPS`]). ≈ +1: consumption follows
    /// retrieval (benign). ≤ 0: mass inversion (adversarial signature).
    pub tau: f32,
    /// Retrieval rank (1..=K) of the TOP-CONSUMED source — the D-SCAN N=1
    /// statistic and the measured discriminator.
    pub top1_consumer_rank: f32,
}

impl TripwireMetrics {
    /// The D-SCAN N=1 statistic normalized to [0, 1]: 0 = the top-consumed
    /// source is also the top-retrieved; 1 = it is retrieval-last.
    pub fn normalized_top1_rank(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        ((self.top1_consumer_rank - 1.0) / (self.n as f32 - 1.0)) as f64
    }

    /// The PoC-verdict detector: fire when the top-consumed source's
    /// normalized retrieval rank exceeds `rank_threshold` — a benign-quantile
    /// threshold from [`conformal_threshold`] over the consumer's own benign
    /// pool of `normalized_top1_rank()` values.
    #[inline]
    pub fn rank_inversion_fires(&self, rank_threshold: f64) -> bool {
        self.normalized_top1_rank() > rank_threshold
    }
}

/// Compute [`TripwireMetrics`] into `out` — zero-allocation, O(K²) for τ.
///
/// Ties on the gate axis are handled exactly like `kendall_tau_a`'s
/// tolerance; the argmax gate source (first maximum on exact ties, a
/// deterministic choice) supplies the N=1 rank statistic.
pub fn tripwire_metrics_into(retrieval: &[f32], gates: &[f32], out: &mut TripwireMetrics) {
    let n = gates.len();
    debug_assert_eq!(
        retrieval.len(),
        n,
        "tripwire: retrieval.len() must equal gates.len()"
    );
    debug_assert!(n > 0, "tripwire: consumed set must be non-empty");

    let total: f32 = gates.iter().copied().sum();
    debug_assert!(total > 0.0, "tripwire: gates must be strictly positive");

    let mut h = 0.0f64;
    let mut top1 = 0.0f32;
    for &g in gates {
        let p = g / total;
        if p > 0.0 {
            h -= p as f64 * (p as f64).ln();
        }
        top1 = top1.max(p);
    }

    let mut am = 0usize;
    for (i, &g) in gates.iter().enumerate() {
        if g > gates[am] {
            am = i;
        }
    }
    let rank = 1.0f32
        + retrieval
            .iter()
            .filter(|&&r| r > retrieval[am] + RANK_TIE_EPS)
            .count() as f32;

    *out = TripwireMetrics {
        n,
        h_norm: (h / (n as f64).ln()) as f32,
        top1_share: top1,
        tau: kendall_tau_a(retrieval, gates, DEFAULT_TIE_EPS),
        top1_consumer_rank: rank,
    };
}

/// Split-conformal benign-quantile threshold: the ⌈(n+1)(1−α)⌉-th order
/// statistic of the benign calibration scores. Under exchangeability, a NEW
/// benign score exceeds the returned threshold with probability ≤ α — the
/// finite-sample benign-FPR guarantee used by the Bench-832 operating points
/// (the one-sided decision-rule sibling of the R322/Plan 340 conformal
/// floor: that floor benchmarks interval UQ, this calibrates a fire rule).
///
/// Sorts `scores` ascending in place (caller-owned buffer, zero alloc).
/// Returns `f64::INFINITY` for an empty slice (never fires).
pub fn conformal_threshold(scores: &mut [f64], alpha: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return f64::INFINITY;
    }
    debug_assert!(
        alpha > 0.0 && alpha < 1.0,
        "conformal_threshold: alpha must be in (0, 1)"
    );
    scores.sort_by(|a, b| a.total_cmp(b));
    let idx = (((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    scores[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(retrieval: &[f32], gates: &[f32]) -> TripwireMetrics {
        let mut out = TripwireMetrics {
            n: 0,
            h_norm: 0.0,
            top1_share: 0.0,
            tau: 0.0,
            top1_consumer_rank: 0.0,
        };
        tripwire_metrics_into(retrieval, gates, &mut out);
        out
    }

    #[test]
    fn tau_identical_reversed_and_tie_excluded() {
        let a = [0.1f32, 0.4, 0.9];
        assert!((kendall_tau_a(&a, &a, DEFAULT_TIE_EPS) - 1.0).abs() < 1e-6);
        let rev = [0.9f32, 0.4, 0.1];
        assert!((kendall_tau_a(&a, &rev, DEFAULT_TIE_EPS) + 1.0).abs() < 1e-6);
        // x ties on pair (0,1) ⇒ that pair drops from numerator AND
        // denominator: the remaining two pairs are concordant ⇒ τ = 1.
        let tied = [1.0f32, 1.0, 2.0];
        let y = [1.0f32, 2.0, 3.0];
        assert!((kendall_tau_a(&tied, &y, DEFAULT_TIE_EPS) - 1.0).abs() < 1e-6);
        // Fully-tied input ⇒ no admissible pairs ⇒ 0.0 (not NaN).
        let flat = [1.0f32, 1.0, 1.0];
        assert_eq!(kendall_tau_a(&flat, &flat, DEFAULT_TIE_EPS), 0.0);
    }

    #[test]
    fn metrics_hand_case_rank_top_tau_one() {
        // gates [0.8, 0.1, 0.1] → p = [0.8, 0.1, 0.1]
        // H = 0.8·ln(1/0.8) + 0.2·ln(10) = 0.63902; /ln 3 = 0.58168
        let mt = m(&[0.9, 0.2, 0.1], &[0.8, 0.1, 0.1]);
        assert_eq!(mt.n, 3);
        assert!(
            (mt.h_norm - 0.581_68).abs() < 2e-4,
            "h_norm = {}",
            mt.h_norm
        );
        assert!((mt.top1_share - 0.8).abs() < 1e-6);
        assert!(
            (mt.tau - 1.0).abs() < 1e-6,
            "top-gate source is top-retrieved"
        );
        assert!((mt.top1_consumer_rank - 1.0).abs() < 1e-6);
        assert!(mt.normalized_top1_rank().abs() < 1e-6);
        assert!(!mt.rank_inversion_fires(0.5));
    }

    #[test]
    fn metrics_rank_inversion_and_boundaries() {
        // Top-gated source is retrieval-LAST: rank 3 of 3 → normalized 1.0.
        let mt = m(&[0.1, 0.5, 0.9], &[0.9, 0.05, 0.05]);
        assert!((mt.top1_consumer_rank - 3.0).abs() < 1e-6);
        assert!((mt.normalized_top1_rank() - 1.0).abs() < 1e-6);
        assert!(mt.rank_inversion_fires(0.5));
        assert!((mt.tau + 1.0).abs() < 1e-6);
        // Single-source set: normalized rank pinned at 0 (never fires).
        let one = m(&[0.3], &[1.0]);
        assert_eq!(one.normalized_top1_rank(), 0.0);
        assert!(!one.rank_inversion_fires(0.0));
        // Uniform gates ⇒ h_norm = 1.0 exactly.
        let uni = m(&[0.3, 0.6, 0.9], &[1.0, 1.0, 1.0]);
        assert!((uni.h_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn conformal_order_statistic_and_edges() {
        let mut s: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // ⌈(101)(0.95)⌉ = 96 → 0-based idx 95 → the 96th smallest of 0..99.
        assert!((conformal_threshold(&mut s, 0.05) - 95.0).abs() < 1e-9);
        let mut s2: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // ⌈(11)(0.5)⌉ = 6 → idx 5.
        assert!((conformal_threshold(&mut s2, 0.5) - 5.0).abs() < 1e-9);
        let mut empty: Vec<f64> = Vec::new();
        assert_eq!(conformal_threshold(&mut empty, 0.05), f64::INFINITY);
        // Unsorted input is handled (sorts in place).
        let mut shuffled = vec![5.0f64, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0];
        assert!((conformal_threshold(&mut shuffled, 0.5) - 5.0).abs() < 1e-9);
    }

    /// The conformal FPR guarantee on the calibration pool itself: the
    /// ⌈(n+1)(1−α)⌉-th order statistic leaves EXACTLY n−1−idx scores above
    /// it — 4 of 100 at α = 0.05 (4% ≤ α) — so a new benign score exceeds
    /// the threshold with probability ≤ α under exchangeability.
    #[test]
    fn conformal_calibration_pool_respects_alpha() {
        let mut s: Vec<f64> = (0..100).map(|i| (i as f64) * 0.37).collect();
        let t = conformal_threshold(&mut s, 0.05);
        let exceed = s.iter().filter(|&&x| x > t).count();
        assert_eq!(
            exceed, 4,
            "n−1−idx = 4 scores above the 96th order statistic"
        );
    }
}
