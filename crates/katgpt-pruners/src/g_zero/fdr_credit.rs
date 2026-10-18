//! FDR family-calibrated partial credit — Terminal-RSI extraction (issue 948 T1).
//!
//! The Terminal-RSI paper's (arXiv:2610.22192) Family Dense Reward is fully
//! offline-computable: score a trajectory by its check passes `P`, normalize by
//! a FROZEN per-family divisor `D_f` (the 90th percentile of `P` on a reference
//! corpus), clamp at zero: `R = max(P / D_f, 0)`.
//!
//! The paper independently converged on the lesson our own harness paid for
//! ("fused-gate birth thresholds do NOT transfer: fit per suite") — normalization
//! is frozen per family, never per batch.
//!
//! # The freeze gate (mandatory — the percentile-index audit lesson)
//!
//! `sorted[(n·0.9) as usize]` lands on the MAX for n ≤ 10 and reports one
//! observation under a percentile's name below the tail-support boundary. The
//! freeze path here uses the nearest-rank p90 index (the katgpt-core
//! `nearest_rank_p99` shape) and REFUSES to freeze a family whose tail support
//! `n − idx` is under [`MIN_TAIL_SUPPORT`] — loud ([`FreezeError`]), never a
//! silent thin-corpus divisor.
//!
//! # Family key
//!
//! A family is any `u64` key the caller owns. In the hint-delta absorb lane
//! (see [`super::delta_absorb`]) the arm index IS the family key — the template
//! that produced the hint.
//!
//! # Determinism
//!
//! Freeze sorts with `total_cmp` (total order over all f32 including NaN) and
//! hashes the frozen table with BLAKE3, so the same reference corpus always
//! yields a byte-identical calibration ([`FdrCalibration::digest`]).

use super::delta_absorb::DeltaGatedAbsorbCompress;
use super::types::HintDelta;
use katgpt_speculative::ScreeningPruner;

/// Minimum observations OVER the frozen percentile index for a family's
/// divisor to freeze: `n − idx ≥ 10` ⇒ a family reference corpus of ≥ 90
/// rows at nearest-rank p90. Below this the divisor is one observation
/// wearing a percentile's name.
pub const MIN_TAIL_SUPPORT: usize = 10;

/// Nearest-rank p90 index for `n` sorted samples — never the max-landing
/// `((n·0.9) as usize)` shape (that returns `n−1` for every `n ≤ 10`).
///
/// Mirrors katgpt-core's `nearest_rank_p99`: rank = ceil(n·0.9), index =
/// rank − 1 clamped into range.
#[inline]
fn nearest_rank_p90(n: usize) -> usize {
    debug_assert!(n > 0, "caller must handle the empty sample");
    let rank = (n as u64 * 90).div_ceil(100) as usize;
    rank.saturating_sub(1).min(n - 1)
}

/// The concrete freeze-refusal reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreezeRefusal {
    /// The family's reference corpus is too thin for a p90 divisor
    /// (empty corpus, or tail support under [`MIN_TAIL_SUPPORT`]).
    ThinTailSupport,
    /// The computed divisor is zero or negative — `max(P/D_f, 0)` is
    /// meaningless without a positive divisor.
    NonPositiveDivisor,
}

/// Why a family's divisor refused to freeze.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreezeError {
    /// The refusing family key.
    pub family: u64,
    /// Rows the family carried (0 for a duplicate-key refusal).
    pub rows: usize,
    /// Tail support the freeze demanded ([`MIN_TAIL_SUPPORT`]).
    pub min_tail_support: usize,
    /// Tail support the family actually had (`rows − idx`).
    pub tail_support: usize,
    /// The concrete reason.
    pub reason: FreezeRefusal,
}

impl std::fmt::Display for FreezeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.reason {
            FreezeRefusal::ThinTailSupport => write!(
                f,
                "family {}: {} rows give tail support {} < {} at p90 — a thin-corpus \
                 divisor is refused, grow the family's reference corpus",
                self.family, self.rows, self.tail_support, self.min_tail_support
            ),
            FreezeRefusal::NonPositiveDivisor => write!(
                f,
                "family {}: frozen p90 divisor is <= 0 — ratio semantics broken, \
                 clean the family's reference corpus",
                self.family
            ),
        }
    }
}

impl std::error::Error for FreezeError {}

/// A frozen family-calibrated partial-credit table.
///
/// Built once offline from a reference corpus ([`FdrCalibration::freeze`]),
/// read-only afterwards: [`credit()`](FdrCalibration::credit) is a binary
/// search + one division, zero-alloc on the hot path.
#[derive(Clone, Debug, PartialEq)]
pub struct FdrCalibration {
    /// `(family, divisor)` rows, sorted ascending by family — binary-searched.
    divisors: Vec<(u64, f32)>,
    /// BLAKE3 over the frozen table (family LE bytes ‖ divisor bits LE, in
    /// table order) — freeze-version provenance for callers that persist the
    /// calibration.
    digest: [u8; 32],
}

impl FdrCalibration {
    /// Freeze per-family p90 divisors from a reference corpus.
    ///
    /// Each row is `(family, samples)` — the family's raw score values.
    /// Samples are sorted internally with a total float order, so the freeze
    /// is deterministic for any input order, and the family table is sorted
    /// so the hot-path lookup is a binary search.
    ///
    /// # Errors
    ///
    /// Refuses — never silently drops — a family whose corpus is empty,
    /// whose tail support is under [`MIN_TAIL_SUPPORT`], or whose p90 divisor
    /// is not strictly positive. A duplicate family key is also refused: the
    /// input is ambiguous about which corpus is authoritative.
    pub fn freeze<'a, I>(families: I) -> Result<Self, FreezeError>
    where
        I: IntoIterator<Item = (u64, &'a [f32])>,
    {
        let mut rows: Vec<(u64, f32)> = Vec::new();

        for (family, samples) in families {
            let refuse = |reason| FreezeError {
                family,
                rows: samples.len(),
                min_tail_support: MIN_TAIL_SUPPORT,
                tail_support: samples
                    .len()
                    .saturating_sub(nearest_rank_p90_or_zero(samples.len())),
                reason,
            };

            let n = samples.len();
            if n == 0 {
                return Err(refuse(FreezeRefusal::ThinTailSupport));
            }
            let idx = nearest_rank_p90(n);
            let tail_support = n - idx;
            if tail_support < MIN_TAIL_SUPPORT {
                return Err(FreezeError {
                    family,
                    rows: n,
                    min_tail_support: MIN_TAIL_SUPPORT,
                    tail_support,
                    reason: FreezeRefusal::ThinTailSupport,
                });
            }

            let mut sorted: Vec<f32> = samples.to_vec();
            sorted.sort_unstable_by(f32::total_cmp);
            // idx < n is guaranteed by nearest_rank_p90's clamp into range.
            let divisor = sorted[idx];
            if !divisor.is_finite() || divisor <= 0.0 {
                return Err(refuse(FreezeRefusal::NonPositiveDivisor));
            }

            if rows.binary_search_by_key(&family, |&(f, _)| f).is_ok() {
                return Err(FreezeError {
                    family,
                    rows: n,
                    min_tail_support: MIN_TAIL_SUPPORT,
                    tail_support,
                    reason: FreezeRefusal::ThinTailSupport,
                });
            }
            rows.push((family, divisor));
        }

        rows.sort_unstable_by_key(|&(f, _)| f);
        let digest = Self::hash_table(&rows);
        Ok(Self {
            divisors: rows,
            digest,
        })
    }

    /// The frozen divisor for `family`, if the family was frozen.
    #[inline]
    pub fn divisor(&self, family: u64) -> Option<f32> {
        self.divisors
            .binary_search_by_key(&family, |&(f, _)| f)
            .ok()
            .map(|idx| self.divisors[idx].1)
    }

    /// Family-calibrated partial credit: `R = max(P / D_f, 0)`.
    ///
    /// Returns `None` for a family that was never frozen — the caller decides
    /// the fallback (the incumbent raw hinge is the documented one); this
    /// primitive never invents a divisor.
    #[inline]
    pub fn credit(&self, family: u64, passes: f32) -> Option<f32> {
        let d = self.divisor(family)?;
        Some((passes / d).max(0.0))
    }

    /// Number of frozen families.
    pub fn len(&self) -> usize {
        self.divisors.len()
    }

    /// True when no family froze (an empty-but-valid calibration).
    pub fn is_empty(&self) -> bool {
        self.divisors.is_empty()
    }

    /// The frozen family keys, ascending.
    pub fn families(&self) -> impl Iterator<Item = u64> + '_ {
        self.divisors.iter().map(|&(f, _)| f)
    }

    /// BLAKE3 over the frozen table — byte-identical freezes agree here.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn hash_table(rows: &[(u64, f32)]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for &(family, divisor) in rows {
            hasher.update(&family.to_le_bytes());
            hasher.update(&divisor.to_bits().to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

/// Tail-support bookkeeping for the empty-corpus error row (`n = 0` has no
/// percentile index — report zero support).
fn nearest_rank_p90_or_zero(n: usize) -> usize {
    if n == 0 { 0 } else { nearest_rank_p90(n) }
}

// ── Absorb-lane adapter ─────────────────────────────────────────────

impl<P: ScreeningPruner> DeltaGatedAbsorbCompress<P> {
    /// FDR family-calibrated hint-delta absorb (issue 948 T1).
    ///
    /// The reward fed to the δ-gate is `max(delta.value / D_f, 0)` with the
    /// per-arm frozen divisor — the family key is the arm index (the template
    /// that produced the hint). Arms absent from the calibration keep the
    /// incumbent raw hinge `delta.value.max(0.0)`.
    #[inline]
    pub fn observe_hint_delta_calibrated(
        &mut self,
        arm: usize,
        delta: &HintDelta,
        credit: &FdrCalibration,
    ) {
        let raw = delta.value.max(0.0);
        let reward = credit.credit(arm as u64, delta.value).unwrap_or(raw);
        self.observe_delta(arm, delta.value, reward);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 100 ascending rows: p90 nearest-rank freezes with tail support 11.
    fn corpus_100(family: u64, base: f32) -> (u64, Vec<f32>) {
        (family, (0..100).map(|i| base + i as f32).collect())
    }

    #[test]
    fn freeze_ok_at_100_rows_and_divisor_is_the_p90() {
        let (family, samples) = corpus_100(1, 0.0);
        let cal = FdrCalibration::freeze([(family, samples.as_slice())]).expect("100 rows freeze");
        assert_eq!(cal.len(), 1);
        // Sorted samples 0..100 → p90 nearest-rank = sample[89] = 89.0.
        assert_eq!(cal.divisor(1), Some(89.0));
        // credit is the ratio, clamped at zero.
        assert_eq!(cal.credit(1, 89.0), Some(1.0));
        assert_eq!(cal.credit(1, 44.5), Some(0.5));
        assert_eq!(cal.credit(1, -5.0), Some(0.0));
        // Unknown family: None — never an invented divisor.
        assert_eq!(cal.credit(2, 50.0), None);
        assert_eq!(cal.divisor(2), None);
    }

    #[test]
    fn freeze_refuses_thin_tail_support_loudly() {
        // n = 89: idx = ceil(80.1) − 1 = 80, tail = 9 < 10 → refused.
        let samples: Vec<f32> = (0..89).map(|i| i as f32).collect();
        let err = FdrCalibration::freeze([(7, samples.as_slice())]).unwrap_err();
        assert_eq!(err.family, 7);
        assert_eq!(err.reason, FreezeRefusal::ThinTailSupport);
        assert_eq!(err.tail_support, 9);
        assert_eq!(err.min_tail_support, MIN_TAIL_SUPPORT);
        // Display names the remedy, not just the refusal.
        assert!(
            err.to_string()
                .contains("grow the family's reference corpus")
        );
        // n = 90 is the floor: idx 80, tail 10 → freezes (samples 1..=90).
        let samples: Vec<f32> = (0..90).map(|i| (i + 1) as f32).collect();
        let cal = FdrCalibration::freeze([(7, samples.as_slice())]).expect("90 rows freeze");
        assert_eq!(cal.divisor(7), Some(81.0));
    }

    #[test]
    fn freeze_refuses_empty_and_nonpositive_divisors() {
        let err = FdrCalibration::freeze([(3, &[] as &[f32])]).unwrap_err();
        assert_eq!(err.reason, FreezeRefusal::ThinTailSupport);
        // All-negative corpus: p90 is negative → ratio semantics broken.
        let samples: Vec<f32> = (0..100).map(|i| -100.0 + i as f32).collect();
        let err = FdrCalibration::freeze([(4, samples.as_slice())]).unwrap_err();
        assert_eq!(err.reason, FreezeRefusal::NonPositiveDivisor);
    }

    #[test]
    fn freeze_refuses_duplicate_family_keys() {
        let (family, samples_a) = corpus_100(9, 0.0);
        let samples_b: Vec<f32> = (0..100).map(|i| 10.0 + i as f32).collect();
        let err =
            FdrCalibration::freeze([(family, samples_a.as_slice()), (9, samples_b.as_slice())])
                .unwrap_err();
        assert_eq!(err.family, 9);
    }

    #[test]
    fn freeze_is_order_deterministic_and_blake3_stamped() {
        let (f1, s1) = corpus_100(1, 0.0);
        let (f2, s2) = corpus_100(2, 5.0);
        let asc = FdrCalibration::freeze([(f1, s1.as_slice()), (f2, s2.as_slice())]).unwrap();
        // Same corpora, reversed input order + reversed samples → same table.
        let mut shuffled: Vec<f32> = (0..100).map(|i| 5.0 + i as f32).collect();
        shuffled.reverse();
        let desc = FdrCalibration::freeze([(2, shuffled.as_slice()), (f1, s1.as_slice())]).unwrap();
        assert_eq!(asc, desc);
        assert_eq!(asc.digest(), desc.digest());
        // A different corpus moves the digest.
        let (g2, t2) = corpus_100(2, 6.0);
        let other = FdrCalibration::freeze([(f1, s1.as_slice()), (g2, t2.as_slice())]).unwrap();
        assert_ne!(asc.digest(), other.digest());
        assert_eq!(asc.families().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn nearest_rank_p90_never_lands_on_the_max_past_the_boundary() {
        // (n, expected idx, expected tail support = n − idx)
        let table = [
            (90usize, 80usize, 10usize),
            (100, 89, 11),
            (101, 90, 11),
            (1000, 899, 101),
        ];
        for (n, want_idx, want_support) in table {
            let got = nearest_rank_p90(n);
            assert_eq!(got, want_idx, "n={n}: rank index");
            assert_eq!(n - got, want_support, "n={n}: tail support");
            assert!(got < n, "n={n}: index {got} out of range");
        }
        // The defect, stated directly: the naive float form returns n−1 at n=10.
        assert_eq!(((10_f64) * 0.9) as usize, 9, "the old shape");
        assert_eq!(nearest_rank_p90(10), 8, "the new one");
    }
}

/// Behavioral wiring test — separate module so the `AllowAll` pruner stub
/// stays beside the only test that needs it.
#[cfg(test)]
mod absorb_tests {
    use super::*;
    use crate::absorb_compress::{AbsorbCompressLayer, CompressConfig};
    use crate::g_zero::delta_absorb::{DeltaGatedAbsorbCompress, DeltaGatedConfig};
    use crate::g_zero::types::HintDelta;

    /// No-op screener that allows everything (delta_absorb's own test stub).
    struct AllowAll;
    impl ScreeningPruner for AllowAll {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn hint(value: f32) -> HintDelta {
        HintDelta {
            value,
            query: "q".into(),
            hint: "h".into(),
            a_hard: "a".into(),
            a_assisted: "b".into(),
            logp_q: -2.0,
            logp_qh: -2.0 - value,
        }
    }

    #[test]
    fn calibrated_absorb_normalizes_per_arm_and_keeps_raw_hinge_fallback() {
        // Arm 0 frozen with divisor 0.075 → a raw δ of 0.15 credits to 2.0.
        // Arm 1 NOT frozen → keeps the incumbent raw hinge (0.15).
        let samples: Vec<f32> = std::iter::repeat_n(0.075, 90).collect();
        let cal = FdrCalibration::freeze([(0, samples.as_slice())]).unwrap();

        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let inner = AbsorbCompressLayer::new(AllowAll, 2, compress_config);
        let mut layer = DeltaGatedAbsorbCompress::new(inner, 2, config);

        layer.observe_hint_delta_calibrated(0, &hint(0.15), &cal);
        layer.observe_hint_delta_calibrated(1, &hint(0.15), &cal);

        // Arm 0: reward was (0.15 / 0.075) = 2.0 → Q = 2.0.
        assert!((layer.inner().arm_q_value(0) - 2.0).abs() < 1e-5);
        // Arm 1: unknown family → raw hinge 0.15 → Q = 0.15.
        assert!((layer.inner().arm_q_value(1) - 0.15).abs() < 1e-5);
        // δ accumulation is unchanged by reward calibration.
        assert!((layer.total_delta(0) - 0.15).abs() < 1e-6);
        // Negative δ still credits zero and does not accumulate:
        // rewards (2.0 + 0.0) over 2 visits → Q drops to 1.0.
        layer.observe_hint_delta_calibrated(0, &hint(-0.5), &cal);
        assert!((layer.inner().arm_q_value(0) - 1.0).abs() < 1e-5);
        assert!(
            (layer.total_delta(0) - 0.15).abs() < 1e-6,
            "negative δ must not accumulate"
        );
    }
}
