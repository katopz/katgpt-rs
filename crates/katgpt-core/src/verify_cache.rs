//! Verify-cache posture + probability-weighted cache error (katgpt-rs
//! Issue 1210 C7 — the Speculative Self-Distillation extraction,
//! arXiv:2611.12016; Provenance: SSD NeurIPS 2026, Apache-2.0, pinned
//! `cccf657b`, `src/ssd/config.py` read + coordinator-verified before
//! filing).
//!
//! # The problem
//!
//! A cached producer distribution (teacher probabilities stored at one
//! posture and read at another) can silently lie: inputs drift, weights
//! refresh, the coarsened tail bucket rounds. The paper's answer is a
//! THREE-posture cache contract — `off` (no cache), `verify` (read the
//! cache AND run the fresh forward, log the reproduction error), `on`
//! (trust the cache) — where promotion `verify → on` is GATED on the
//! measured reproduction error. This module is the closed-form half of
//! that contract: the error accounting, the error taxonomy, and the
//! promotion gate. The cache itself, the fresh forward, and the serving
//! loop are consumer surfaces — this module only prices what a verify
//! pass observed.
//!
//! # The error unit is PROBABILITY, not nats (sentinel honesty)
//!
//! The obvious log-space reading `|cached_logprob − fresh_logprob|` is a
//! trap the fixtures caught before landing: the `-1e9` sentinel is the
//! finite spelling of "no mass here", and against a fresh logprob of −40
//! ("very little mass") it produces a 1e9-nat *arithmetic* error while
//! the *behavioral* difference is |0 − 4e−18| ≈ nothing. Accounting that
//! artifact as error makes every legitimate coarsened cache (C6 rows
//! carry sentinels on their tails) permanently unpromotable. So the
//! per-token error is taken in probability space —
//! `|exp(cached) − exp(fresh)|` — where the sentinel underflows to exact
//! `0.0` and reads as the honest zero it means. The input contract
//! enforces this: logprobs are ≤ 0 by definition (a probability cannot
//! exceed 1), and a positive input is a typed refusal
//! [`VerifyCacheError::PositiveLogprob`] — which also bounds every `exp`
//! to `[0, 1]`, so no report field can overflow.
//!
//! # Probability weighting (the discount law)
//!
//! A raw per-token error overstates the serving cost: a token the
//! producer would never emit (`p ≈ 0`) contributes nothing to behavior no
//! matter how wrong the cache reads there. Each token's error is
//! therefore weighted by the producer's probability **measured on the
//! FRESH side** — the verify pass's own ground truth, not the cache's
//! claim. Weighting by the cached probability would discount exactly the
//! corruption signal a verify posture exists to catch (a cache entry
//! zeroed by staleness would weigh itself to invisibility); weighting by
//! the fresh probability makes "errors on never-emitted tokens discount
//! to zero" literal — never-emitted is a property of the true
//! distribution. A fully-sentinel fresh row (mass exactly 0) defines the
//! weighted mean as exactly 0.0: nothing emits, nothing can be behaviorally
//! wrong — the raw legs still witness whatever the cache claimed.
//!
//! # The error taxonomy (alignment vs truncation)
//!
//! The report separates the two failure shapes the issue names:
//!
//! - **Alignment** — LARGE, SPORADIC: at least one token's probability
//!   error exceeds [`VerifyCacheConfig::alignment_bar_prob`]. This is the
//!   stale/misaligned-cache class (a mass-bearing token materially
//!   misread — wrong position, refreshed weights, corrupted entries), the
//!   class the verify posture exists to catch.
//! - **Truncation** — SMALL, SYSTEMATIC: every token bounded under the
//!   alignment bar (the coarsened-tail class — inherent to the
//!   representation, calibrated by the weighted bar alone).
//! - **Clean** — bitwise-zero error (identical readings).
//!
//! # The promotion gate (deliberately strict, documented)
//!
//! [`promote_posture`] returns [`CachePosture::On`] only when BOTH legs
//! hold: the probability-weighted mean error ≤
//! [`VerifyCacheConfig::promote_bar_prob`], AND the classification is not
//! Alignment. The second leg is this module's one judgment call beyond
//! the issue's single-bar sentence: the weighted bar is an AGGREGATE and
//! can average away one locally-wrong token when the rest of the row is
//! exact — but a cache that is locally wrong at one mass-bearing token
//! cannot be trusted to be locally right elsewhere. A consumer wanting
//! the bare weighted-bar semantics reads [`CacheVerifyReport`] fields
//! directly — the data is never hidden, the strictness lives only in the
//! gate fn.
//!
//! # Modelless discipline
//!
//! Pure closed-form arithmetic. No training, no learned parameters, no
//! RNG. Determinism: sequential f64 folds in ascending index order (the
//! Issue-940 reduction-order law). Zero allocation: two slice walks, the
//! report on the stack. Inputs are f32 logprobs; every accumulator is
//! f64. Sigmoid never softmax — nothing here needs either.

/// The three-posture cache contract (the paper's `teacher_cache ∈ {off,
/// verify, on}`). `Verify` produces [`CacheVerifyReport`]s each pass;
/// [`promote_posture`] adjudicates the `verify → on` step from one report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CachePosture {
    /// No cache reads; every read is a fresh forward.
    Off,
    /// Read the cache AND run the fresh forward; feed the pair to
    /// [`verify_cache_reproduction`] and watch the reports.
    Verify,
    /// Trust the cache; no fresh verification forward. Reached only
    /// through [`promote_posture`] (never constructed blindly by a
    /// consumer that ran verify first).
    On,
}

/// The promotion gate's refusal family: the weighted-error bar, the
/// alignment class, or the posture it was called from.
#[derive(Debug, Clone, PartialEq)]
pub enum PromoteRefusal {
    /// The weighted mean error exceeds the promotion bar.
    WeightedErrorOverBar { weighted_mean: f64, bar: f64 },
    /// The alignment class fired — a mass-bearing token materially
    /// misread. `max_error` names the worst token's probability error,
    /// `n_above_bar` how many sit over the alignment bar.
    AlignmentClass {
        max_error: f64,
        alignment_bar: f64,
        n_above_bar: usize,
    },
    /// The posture was already `On` (nothing to promote) or `Off`
    /// (verify never ran — no report could exist).
    PostureNotVerify { posture: CachePosture },
}

impl std::fmt::Display for PromoteRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightedErrorOverBar { weighted_mean, bar } => {
                write!(
                    f,
                    "weighted mean error {weighted_mean:.6e} over the promotion bar {bar:.3e}"
                )
            }
            Self::AlignmentClass {
                max_error,
                alignment_bar,
                n_above_bar,
            } => write!(
                f,
                "alignment class: {n_above_bar} token(s) over the alignment bar \
                 {alignment_bar:.3e} (max {max_error:.6e}) — a mass-bearing token \
                 materially misread, never promoted on"
            ),
            Self::PostureNotVerify { posture } => match posture {
                CachePosture::Off => {
                    write!(f, "posture Off: verify never ran, no report can exist")
                }
                CachePosture::Verify => write!(f, "posture already Verify"),
                CachePosture::On => write!(f, "posture already On"),
            },
        }
    }
}

impl std::error::Error for PromoteRefusal {}

/// The one failure class of the accounting itself: everything else is a
/// valid reading (a high error is a finding, not a refusal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyCacheError {
    /// The two rows must describe the same token sequence.
    LengthMismatch { cached: usize, fresh: usize },
    /// A logprob was `NaN`/`±inf`. The finite sentinel `-1e9` is the
    /// supported spelling of "no probability here" — it underflows the
    /// probability to exactly `0.0`.
    NonFiniteLogprob { side: &'static str, index: usize },
    /// A logprob was strictly positive — probabilities cannot exceed 1,
    /// so the value is not a logprob. `0.0` (certainty) is valid.
    PositiveLogprob { side: &'static str, index: usize },
    /// Empty rows describe no token sequence.
    Empty,
}

impl std::fmt::Display for VerifyCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { cached, fresh } => {
                write!(f, "length mismatch: cached {cached} vs fresh {fresh}")
            }
            Self::NonFiniteLogprob { side, index } => write!(
                f,
                "non-finite logprob ({side}) at index {index} — use the -1e9 sentinel"
            ),
            Self::PositiveLogprob { side, index } => write!(
                f,
                "positive logprob ({side}) at index {index} — a probability cannot exceed 1"
            ),
            Self::Empty => write!(f, "empty logprob rows"),
        }
    }
}

impl std::error::Error for VerifyCacheError {}

/// The three failure shapes, typed. See the module docs for the taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheErrorClass {
    /// Bitwise-zero error — the cached and fresh readings agree exactly.
    Clean,
    /// Every token error bounded under the alignment bar — the
    /// small-systematic (coarsened-tail) class.
    Truncation,
    /// At least one token over the alignment bar — the large-sporadic
    /// (mass-bearing-token-misread) class.
    Alignment,
}

/// Bars and knobs of the verify contract. Both bars are CONSUMER
/// calibration knobs — this module ships working defaults and pins no
/// universal bar claim (ungoat discipline: a wired consumer pre-registers
/// its own bars in its gate).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerifyCacheConfig {
    /// The promotion bar on the probability-weighted MEAN error (units:
    /// probability per emitted token). Default `1e-2` — one percent of
    /// probability mass, far under any sampling-relevant shift.
    pub promote_bar_prob: f64,
    /// The alignment bar: a single token's probability error above this
    /// is the large-sporadic class regardless of weight. Default `0.1` —
    /// a tenth of probability mass on one token.
    pub alignment_bar_prob: f64,
}

impl Default for VerifyCacheConfig {
    fn default() -> Self {
        Self {
            promote_bar_prob: 1e-2,
            alignment_bar_prob: 0.1,
        }
    }
}

/// One verify pass's reading. Every field is the data a consumer needs to
/// apply its own gate — [`promote_posture`] is a convenience over these.
/// All units are PROBABILITY (see the module docs for why not nats).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheVerifyReport {
    /// Raw probability-weighted error `Σ p_fresh_i · |p_c_i − p_f_i|`.
    /// Scales with sequence length — read the mean for posture decisions.
    pub weighted_error_prob: f64,
    /// `weighted_error_prob / total_producer_mass` — the expected error
    /// per emitted token. THE gate quantity. Exactly `0.0` when the fresh
    /// mass is exactly 0 (nothing emits; documented in the module docs).
    pub weighted_mean_prob: f64,
    /// `Σ p_fresh_i` — the producer's probability mass the pass saw.
    /// `1.0` on a full normalized row; lower on coarsened/kept-only rows
    /// (the honest coverage floor, C6's `kept_mass` sibling).
    pub total_producer_mass: f64,
    /// The worst single-token probability error — the alignment leg's
    /// witness.
    pub max_token_error_prob: f64,
    /// Unweighted mean token probability error — the truncation leg's
    /// witness.
    pub mean_token_error_prob: f64,
    /// Tokens compared.
    pub n_tokens: usize,
    /// Tokens over the alignment bar.
    pub n_above_alignment_bar: usize,
    /// The typed taxonomy reading.
    pub classification: CacheErrorClass,
}

/// The verify half of the contract: one cached row, one fresh row, one
/// report. Weight = the fresh-side producer probability (see the module
/// docs for why the fresh side carries it).
pub fn verify_cache_reproduction(
    cached_logprobs: &[f32],
    fresh_logprobs: &[f32],
    config: &VerifyCacheConfig,
) -> Result<CacheVerifyReport, VerifyCacheError> {
    if cached_logprobs.len() != fresh_logprobs.len() {
        return Err(VerifyCacheError::LengthMismatch {
            cached: cached_logprobs.len(),
            fresh: fresh_logprobs.len(),
        });
    }
    if cached_logprobs.is_empty() {
        return Err(VerifyCacheError::Empty);
    }
    for (i, &x) in cached_logprobs.iter().enumerate() {
        if !x.is_finite() {
            return Err(VerifyCacheError::NonFiniteLogprob {
                side: "cached",
                index: i,
            });
        }
        if x > 0.0 {
            return Err(VerifyCacheError::PositiveLogprob {
                side: "cached",
                index: i,
            });
        }
    }
    for (i, &x) in fresh_logprobs.iter().enumerate() {
        if !x.is_finite() {
            return Err(VerifyCacheError::NonFiniteLogprob {
                side: "fresh",
                index: i,
            });
        }
        if x > 0.0 {
            return Err(VerifyCacheError::PositiveLogprob {
                side: "fresh",
                index: i,
            });
        }
    }

    // Sequential f64 folds, ascending index (Issue-940). Probability
    // space per token: p = exp(logprob) ∈ [0, 1] (the ≤ 0 contract makes
    // the bound structural), err = |p_c − p_f|, weight = p_f.
    let mut weighted_sum = 0.0f64;
    let mut mass_sum = 0.0f64;
    let mut err_sum = 0.0f64;
    let mut max_err = 0.0f64;
    let mut n_above = 0usize;
    let alignment_bar = config.alignment_bar_prob;
    for i in 0..cached_logprobs.len() {
        let p_c = f64::from(cached_logprobs[i]).exp();
        let p_f = f64::from(fresh_logprobs[i]).exp();
        let err = (p_c - p_f).abs();
        weighted_sum += p_f * err;
        mass_sum += p_f;
        err_sum += err;
        if err > max_err {
            max_err = err;
        }
        if err > alignment_bar {
            n_above += 1;
        }
    }

    let classification = if max_err == 0.0 {
        CacheErrorClass::Clean
    } else if n_above > 0 {
        CacheErrorClass::Alignment
    } else {
        CacheErrorClass::Truncation
    };
    let n = cached_logprobs.len();
    let weighted_mean = if mass_sum == 0.0 {
        0.0 // nothing emits — nothing can be behaviorally wrong (documented)
    } else {
        weighted_sum / mass_sum
    };
    Ok(CacheVerifyReport {
        weighted_mean_prob: weighted_mean,
        weighted_error_prob: weighted_sum,
        total_producer_mass: mass_sum,
        max_token_error_prob: max_err,
        mean_token_error_prob: err_sum / n as f64,
        n_tokens: n,
        n_above_alignment_bar: n_above,
        classification,
    })
}

/// The `verify → on` promotion gate over one report: the weighted mean
/// under the bar AND no alignment class (the strictness is deliberate —
/// see the module docs). The posture must be `Verify`; `Off` never ran a
/// verify pass and `On` is already past the gate.
pub fn promote_posture(
    posture: CachePosture,
    report: &CacheVerifyReport,
    config: &VerifyCacheConfig,
) -> Result<CachePosture, PromoteRefusal> {
    if posture != CachePosture::Verify {
        return Err(PromoteRefusal::PostureNotVerify { posture });
    }
    if report.classification == CacheErrorClass::Alignment {
        return Err(PromoteRefusal::AlignmentClass {
            max_error: report.max_token_error_prob,
            alignment_bar: config.alignment_bar_prob,
            n_above_bar: report.n_above_alignment_bar,
        });
    }
    if report.weighted_mean_prob > config.promote_bar_prob {
        return Err(PromoteRefusal::WeightedErrorOverBar {
            weighted_mean: report.weighted_mean_prob,
            bar: config.promote_bar_prob,
        });
    }
    Ok(CachePosture::On)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_config() -> VerifyCacheConfig {
        VerifyCacheConfig::default()
    }

    /// A real normalized logprob row: f64 log-softmax of `logits`, read
    /// back as f32 (the wire dtype). Sums to ≈ 1 within f32 rounding.
    fn log_softmax_row(logits: &[f64]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits.iter().map(|&x| (x - max).exp()).sum::<f64>().ln();
        logits.iter().map(|&x| (x - lse) as f32).collect()
    }

    #[test]
    fn identical_rows_read_clean_and_promote() {
        let fresh = log_softmax_row(&[2.0, 0.5, -1.0, 0.3]);
        let report =
            verify_cache_reproduction(&fresh, &fresh, &fixture_config()).expect("identical rows");
        assert_eq!(report.weighted_error_prob, 0.0);
        assert_eq!(report.weighted_mean_prob, 0.0);
        assert_eq!(report.max_token_error_prob, 0.0);
        assert_eq!(report.classification, CacheErrorClass::Clean);
        // A normalized row's producer mass is 1 up to f32 rounding of the
        // logprobs.
        assert!((report.total_producer_mass - 1.0).abs() < 1e-4);
        assert_eq!(
            promote_posture(CachePosture::Verify, &report, &fixture_config()),
            Ok(CachePosture::On)
        );
    }

    #[test]
    fn never_emitted_tokens_discount_to_zero() {
        // The discount law, literal: the cache claims certainty (p = 1)
        // on ONE token whose true probability is exp(−40) ≈ 6e−18, every
        // other token exact. The raw error at that token is ≈ 1.0 — but
        // the error sits where the producer never emits, so the weighted
        // mean is the truth's own mass (≈ 6e−18), not 1.0.
        let fresh = log_softmax_row(&[2.0, -40.0]);
        let mut cached = fresh.clone();
        cached[1] = 0.0; // certainty where the truth says never
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert!(
            report.weighted_mean_prob < 1e-15,
            "got {}",
            report.weighted_mean_prob
        );
        // The raw leg witnesses the claim: certainty vs never is a full
        // 1.0 probability error at that token.
        assert!(report.max_token_error_prob > 0.9);
        assert_eq!(report.classification, CacheErrorClass::Alignment);
    }

    #[test]
    fn high_mass_misalignment_is_alignment_and_refuses() {
        // The fresh argmax carries ~0.88 mass; the cache flattens the
        // row toward its own two-token reading — the head token's
        // probability error ~0.35, the second token's ~0.23, both over
        // the 0.1 alignment bar. Two mass-bearing tokens materially
        // misread.
        let fresh = log_softmax_row(&[3.0, -1.0, -2.0]);
        let cached = log_softmax_row(&[-0.5, -1.0, -2.0]);
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert_eq!(report.classification, CacheErrorClass::Alignment);
        assert_eq!(report.n_above_alignment_bar, 3);
        assert!(report.weighted_mean_prob > 1e-2);
        assert!(matches!(
            promote_posture(CachePosture::Verify, &report, &fixture_config()),
            Err(PromoteRefusal::AlignmentClass { max_error, n_above_bar: 3, .. })
                if (0.4..0.45).contains(&max_error)
        ));
    }

    #[test]
    fn truncation_under_the_bar_promotes() {
        // Small systematic error on every token — the coarsened-tail
        // class. Under the weighted bar → Truncation does NOT block
        // promotion; only the weighted bar and Alignment do.
        let fresh = log_softmax_row(&[1.0, 0.2, -0.8]);
        let cached = log_softmax_row(&[1.001, 0.199, -0.801]); // ~1e-3 nats off
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert_eq!(report.classification, CacheErrorClass::Truncation);
        assert!(report.weighted_mean_prob <= 1e-2);
        assert_eq!(
            promote_posture(CachePosture::Verify, &report, &fixture_config()),
            Ok(CachePosture::On)
        );
    }

    #[test]
    fn truncation_over_the_bar_refuses_on_weight() {
        // Systematic but large enough in aggregate: every token shifted
        // ~0.05 nats → probability errors ~2e-2 each, weighted mean over
        // the 1e-2 bar, still Truncation (each under the 0.1 alignment
        // bar). The weight leg alone refuses.
        let fresh = log_softmax_row(&[1.0, 0.2, -0.8]);
        let cached = log_softmax_row(&[1.05, 0.15, -0.75]);
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert_eq!(report.classification, CacheErrorClass::Truncation);
        assert!(report.weighted_mean_prob > 1e-2);
        assert_eq!(
            promote_posture(CachePosture::Verify, &report, &fixture_config()),
            Err(PromoteRefusal::WeightedErrorOverBar {
                weighted_mean: report.weighted_mean_prob,
                bar: 1e-2,
            })
        );
    }

    #[test]
    fn alignment_refuses_even_when_the_weighted_bar_passes() {
        // THE strictness arm: ONE mass-bearing token (true p ≈ 0.04)
        // misread to ≈ 0.15 — a 0.11 probability error over the 0.1
        // alignment bar — while every other token is exact. The weighted
        // mean ≈ 0.04·0.11 ≈ 0.0044 rides under a 0.05 consumer bar, and
        // the gate STILL refuses: the aggregate bar can average away one
        // locally-wrong token, and a cache that is locally wrong at a
        // mass-bearing token cannot be trusted to be locally right
        // elsewhere.
        let config = VerifyCacheConfig {
            promote_bar_prob: 5e-2,
            ..VerifyCacheConfig::default()
        };
        let fresh = log_softmax_row(&[2.5, -3.0, -3.2, -3.4, -3.6, -3.8, -4.0, -4.2, -4.4, -4.6]);
        let mut cached = fresh.clone();
        // fresh token 1 carries p ≈ 0.04; move it up to ≈ 0.15.
        let k = 1;
        let fresh_p = f64::from(fresh[k]).exp();
        cached[k] = (fresh_p + 0.11).ln() as f32;
        let report = verify_cache_reproduction(&cached, &fresh, &config).expect("rows");
        assert!(report.weighted_mean_prob < config.promote_bar_prob);
        assert_eq!(report.classification, CacheErrorClass::Alignment);
        assert_eq!(report.n_above_alignment_bar, 1);
        assert!(matches!(
            promote_posture(CachePosture::Verify, &report, &config),
            Err(PromoteRefusal::AlignmentClass { .. })
        ));
    }

    #[test]
    fn cached_sentinel_over_near_zero_truth_reads_truncation_and_promotes() {
        // The sentinel-honesty arm (the one that killed the nat-space
        // draft): the cache stores the -1e9 sentinel ("no mass") where
        // the truth says exp(−40) ≈ 4e−18. In nat space that reads as a
        // 1e9-nat lie; in probability space the error is 4e−18 — the
        // sentinel is honest, the cache is behaviorally exact, and the
        // gate promotes.
        let fresh = log_softmax_row(&[0.0, -40.0, -0.2]);
        let cached = vec![fresh[0], -1.0e9, fresh[2]];
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert!(report.weighted_error_prob.is_finite());
        assert!(report.total_producer_mass.is_finite());
        assert!(report.weighted_mean_prob < 1e-15);
        assert!(report.max_token_error_prob < 1e-15);
        assert_eq!(report.classification, CacheErrorClass::Truncation);
        assert_eq!(
            promote_posture(CachePosture::Verify, &report, &fixture_config()),
            Ok(CachePosture::On)
        );
    }

    #[test]
    fn fresh_sentinel_reads_never_emitted_exactly() {
        // The mirror arm: the FRESH side carries the sentinel — the true
        // producer probability is exactly 0 — so a cache entry claiming
        // certainty there contributes EXACTLY 0.0 weighted (0 · err),
        // while the raw legs still witness the disagreement and the
        // alignment class still refuses. The two legs answer different
        // questions, pinned apart.
        let fresh = vec![fresh_row_head(), -1.0e9, fresh_row_tail()];
        let cached = vec![fresh_row_head(), 0.0, fresh_row_tail()];
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert_eq!(report.weighted_error_prob, 0.0);
        assert_eq!(report.weighted_mean_prob, 0.0);
        assert_eq!(report.classification, CacheErrorClass::Alignment);
        assert!(matches!(
            promote_posture(CachePosture::Verify, &report, &fixture_config()),
            Err(PromoteRefusal::AlignmentClass { .. })
        ));
    }

    fn fresh_row_head() -> f32 {
        -0.5
    }

    fn fresh_row_tail() -> f32 {
        -0.7
    }

    #[test]
    fn all_sentinel_fresh_row_defines_the_mean_as_zero() {
        // Nothing emits (every true probability exactly 0): the weighted
        // mean is DEFINED 0.0 (documented), and whatever the cache
        // claimed still witnesses through the raw legs.
        let fresh = vec![-1.0e9f32, -1.0e9];
        let cached = vec![0.0f32, -0.3];
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert_eq!(report.total_producer_mass, 0.0);
        assert_eq!(report.weighted_mean_prob, 0.0);
        assert_eq!(report.classification, CacheErrorClass::Alignment);
        assert!(report.mean_token_error_prob > 0.0);
    }

    #[test]
    fn refusals_length_positive_nonfinite_and_empty() {
        let fresh = log_softmax_row(&[1.0, 0.5]);
        assert_eq!(
            verify_cache_reproduction(&[-1.0f32], &fresh, &fixture_config()),
            Err(VerifyCacheError::LengthMismatch {
                cached: 1,
                fresh: 2
            })
        );
        let mut positive = fresh.clone();
        positive[1] = 0.5;
        assert_eq!(
            verify_cache_reproduction(&positive, &fresh, &fixture_config()),
            Err(VerifyCacheError::PositiveLogprob {
                side: "cached",
                index: 1
            })
        );
        let mut nan = fresh.clone();
        nan[1] = f32::NAN;
        assert_eq!(
            verify_cache_reproduction(&nan, &fresh, &fixture_config()),
            Err(VerifyCacheError::NonFiniteLogprob {
                side: "cached",
                index: 1
            })
        );
        let mut inf = fresh.clone();
        inf[0] = f32::INFINITY;
        assert_eq!(
            verify_cache_reproduction(&fresh, &inf, &fixture_config()),
            Err(VerifyCacheError::NonFiniteLogprob {
                side: "fresh",
                index: 0
            })
        );
        let mut positive_fresh = fresh.clone();
        positive_fresh[0] = 0.1;
        assert_eq!(
            verify_cache_reproduction(&fresh, &positive_fresh, &fixture_config()),
            Err(VerifyCacheError::PositiveLogprob {
                side: "fresh",
                index: 0
            })
        );
        assert_eq!(
            verify_cache_reproduction(&[], &[], &fixture_config()),
            Err(VerifyCacheError::Empty)
        );
        // Exactly 0.0 (certainty) is valid on both sides.
        let certainty = vec![0.0f32, -2.0];
        assert!(verify_cache_reproduction(&certainty, &certainty, &fixture_config()).is_ok());
    }

    #[test]
    fn folds_are_deterministic_across_runs() {
        // 4096 tokens, deterministic pseudo-inputs, two runs, bitwise.
        let n = 4096;
        let fresh: Vec<f32> = (0..n)
            .map(|i| -0.01 - ((i * 7919) % 97) as f32 * 0.05)
            .collect();
        let cached: Vec<f32> = fresh
            .iter()
            .enumerate()
            .map(|(i, &f)| f + if i % 11 == 0 { -1e-4 } else { 5e-5 })
            .collect();
        let a = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        let b = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        assert_eq!(
            a.weighted_error_prob.to_bits(),
            b.weighted_error_prob.to_bits()
        );
        assert_eq!(
            a.weighted_mean_prob.to_bits(),
            b.weighted_mean_prob.to_bits()
        );
        assert_eq!(
            a.total_producer_mass.to_bits(),
            b.total_producer_mass.to_bits()
        );
        assert_eq!(a.classification, b.classification);
    }

    #[test]
    fn weighted_error_is_monotone_in_each_token_error() {
        // Well-formedness: moving the cache further from the truth on one
        // token never decreases the weighted error (the fold is a sum of
        // non-negative terms; pinned constructively rather than by
        // theorem).
        let fresh = log_softmax_row(&[1.0, 0.0, -1.0, -2.0]);
        let near = fresh.clone();
        let farther = {
            let mut c = fresh.clone();
            c[0] -= 0.2; // moved away from p=1 certainty
            c
        };
        let r0 = verify_cache_reproduction(&near, &fresh, &fixture_config()).expect("rows");
        let r1 = verify_cache_reproduction(&farther, &fresh, &fixture_config()).expect("rows");
        assert!(r1.weighted_error_prob > r0.weighted_error_prob);
        assert!(r1.weighted_mean_prob > r0.weighted_mean_prob);
    }

    #[test]
    fn producer_mass_matches_a_sequential_reference_fold() {
        // The mass leg sums exactly like an independent sequential f64
        // reference on a fixture that includes a sentinel (underflow to
        // exact 0.0) — and a normalized row's mass is ≈ 1.
        let fresh = vec![-0.5f32, -40.0, -0.7, -1.0e9];
        let cached = fresh.clone();
        let report = verify_cache_reproduction(&cached, &fresh, &fixture_config()).expect("rows");
        let mut reference = 0.0f64;
        for &x in &fresh {
            reference += f64::from(x).exp();
        }
        assert_eq!(report.total_producer_mass.to_bits(), reference.to_bits());
        assert!(report.total_producer_mass > 0.0);

        let normalized = log_softmax_row(&[2.0, 0.5, -1.0, 0.3]);
        let report2 =
            verify_cache_reproduction(&normalized, &normalized, &fixture_config()).expect("rows");
        assert!((report2.total_producer_mass - 1.0).abs() < 1e-4);
    }

    #[test]
    fn posture_gate_refuses_off_and_on() {
        let fresh = log_softmax_row(&[1.0, 0.5]);
        let report = verify_cache_reproduction(&fresh, &fresh, &fixture_config()).expect("rows");
        assert_eq!(
            promote_posture(CachePosture::Off, &report, &fixture_config()),
            Err(PromoteRefusal::PostureNotVerify {
                posture: CachePosture::Off
            })
        );
        assert_eq!(
            promote_posture(CachePosture::On, &report, &fixture_config()),
            Err(PromoteRefusal::PostureNotVerify {
                posture: CachePosture::On
            })
        );
    }
}
