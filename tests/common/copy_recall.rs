//! Issue 1241 T1 — the `copy_recall` long-context fixture + offset-averaged
//! scorer (arXiv:2611.13378 §5.3 + App. H protocol, std-only).
//!
//! The loop family's benches are all single-token, random-weights,
//! norm/KL-shaped; none of them can see a recall cliff (the paper's YOCO-U
//! inversion: 97% of the aggregate-BPB gain kept, 30% of the long-range gain
//! forfeited). This fixture isolates the span reachable ONLY through
//! cross-position state:
//!
//! ```text
//! repeat : [ filler × offset ][ block (seeded) ][ the SAME block  ]
//! control: [ filler × offset ][ block (seeded) ][ a FRESH block   ]
//!                                                 ^ copy span (same positions)
//! ```
//!
//! The copy span is predictable only by recalling the first block. **Copy
//! gain** = score(copy span | repeat) − score(copy span | control): the two
//! streams share every position and every prefix token, differing ONLY in
//! whether the span repeats — so a model with no recall reads ≈ 0 whatever
//! its calibration or its drift with absolute position. (Measured, the
//! reason for the matched control: the first draft compared the copy span
//! with the FIRST span, and a random-weights k=4 loop read +0.05 bits/token
//! of "recall" at every offset — pure position drift, the falsifier firing
//! on the fixture itself.) Scores are reported **offset-averaged** (mean +
//! spread over ≥ 3 offsets), never a single-offset number.
//!
//! Logit convention: `logits[p]` is the readout after feeding `tokens[p]`, so
//! it predicts `tokens[p + 1]`. The copy span's first target is excluded: it
//! is predicted from the first block's LAST token, before any recall cue.
//!
//! The bits-per-token readout normalises the model's own logits with a f64
//! log-sum-exp: it MEASURES the readout distribution (the logit-lens
//! exception, `loop_depth_probe`'s "why a softmax appears here") — no
//! decision softmax.

#![allow(dead_code)]

use std::ops::Range;

/// Minimum offsets an averaged score accepts (the issue's "≥ 3 offsets").
pub const MIN_OFFSETS: usize = 3;

/// One copy-recall sequence.
#[derive(Clone, Debug)]
pub struct CopyRecallFixture {
    /// The repeat stream: filler, block, block again.
    pub tokens: Vec<usize>,
    /// The matched control: filler, block, then an independent seeded
    /// block of the same length (same positions, no repetition).
    pub control: Vec<usize>,
    /// Filler length before the first block.
    pub offset: usize,
    /// Block length (the copy distance).
    pub block: usize,
}

impl CopyRecallFixture {
    /// A seeded fixture over `vocab` symbols. The filler and the block draw
    /// from independent seeded streams, so a different `offset` moves the
    /// SAME block to a different absolute position.
    pub fn new(seed: u64, vocab: usize, block: usize, offset: usize) -> Self {
        assert!(
            vocab >= 2 && block >= 2,
            "a fixture needs two symbols and two tokens"
        );
        let mut filler = Lcg::new(seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut body = Lcg::new(seed);
        let mut fresh = Lcg::new(seed ^ 0xc2b2_ae3d_27d4_eb4f);
        let mut tokens = Vec::with_capacity(offset + 2 * block);
        tokens.extend((0..offset).map(|_| filler.below(vocab)));
        let start = tokens.len();
        tokens.extend((0..block).map(|_| body.below(vocab)));
        let mut control = tokens.clone();
        tokens.extend_from_within(start..start + block);
        control.extend((0..block).map(|_| fresh.below(vocab)));
        Self {
            tokens,
            control,
            offset,
            block,
        }
    }

    /// Target positions of the copy (its first token excluded).
    pub fn copy_span(&self) -> Range<usize> {
        self.offset + self.block + 1..self.offset + 2 * self.block
    }
}

/// Exact-match rate and teacher-forced bits/token over one span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpanScore {
    /// Fraction of targets the readout's first-index argmax hits.
    pub exact: f64,
    /// Mean `−log₂ p(target)` (teacher-forced bits per token).
    pub bits: f64,
    /// Standard error of `exact` (binomial, `√(p(1−p)/n)`).
    pub exact_se: f64,
    /// Standard error of `bits` (sample sd / √n, two-pass sequential fold).
    pub bits_se: f64,
}

/// Score the targets `span` (positions into `tokens`) from `logits`
/// (`logits[t − 1]` predicts `tokens[t]`). Sequential ascending fold.
pub fn score_span(logits: &[Vec<f32>], tokens: &[usize], span: Range<usize>) -> SpanScore {
    assert!(span.start >= 1 && span.end <= tokens.len() && span.end <= logits.len() + 1);
    let n = span.len();
    assert!(n > 0, "an empty span has no score");
    let token_bits = |t: usize| {
        let row = &logits[t - 1];
        (log_sum_exp(row) - row[tokens[t]] as f64) / std::f64::consts::LN_2
    };
    let (mut hits, mut sum) = (0usize, 0.0f64);
    for t in span.clone() {
        if first_argmax(&logits[t - 1]) == tokens[t] {
            hits += 1;
        }
        sum += token_bits(t);
    }
    let nf = n as f64;
    let (exact, bits) = (hits as f64 / nf, sum / nf);
    let ss: f64 = span.map(|t| (token_bits(t) - bits).powi(2)).sum();
    let sd = if n > 1 { (ss / (nf - 1.0)).sqrt() } else { 0.0 };
    SpanScore {
        exact,
        bits,
        exact_se: (exact * (1.0 - exact) / nf).sqrt(),
        bits_se: sd / nf.sqrt(),
    }
}

/// One offset's recall reading: the copy span under both streams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecallScore {
    /// The copy span on the repeat stream.
    pub copy: SpanScore,
    /// The same positions on the matched control stream.
    pub control: SpanScore,
}

impl RecallScore {
    /// Score the copy span from the repeat decode (`logits` over
    /// `fixture.tokens`) and the control decode (`control_logits` over
    /// `fixture.control`).
    pub fn of(
        fixture: &CopyRecallFixture,
        logits: &[Vec<f32>],
        control_logits: &[Vec<f32>],
    ) -> Self {
        Self {
            copy: score_span(logits, &fixture.tokens, fixture.copy_span()),
            control: score_span(control_logits, &fixture.control, fixture.copy_span()),
        }
    }

    /// Exact-match gain of the repeat over the matched control.
    pub fn gain_exact(&self) -> f64 {
        self.copy.exact - self.control.exact
    }

    /// Bits saved on the repeat relative to the matched control.
    pub fn gain_bits(&self) -> f64 {
        self.control.bits - self.copy.bits
    }

    /// Standard error of [`Self::gain_exact`] (independent spans).
    pub fn gain_exact_se(&self) -> f64 {
        self.copy.exact_se.hypot(self.control.exact_se)
    }

    /// Standard error of [`Self::gain_bits`] (independent spans).
    pub fn gain_bits_se(&self) -> f64 {
        self.copy.bits_se.hypot(self.control.bits_se)
    }
}

/// The offset-averaged copy gain — mean and spread (max − min) per axis.
#[derive(Clone, Debug, PartialEq)]
pub struct OffsetAveraged {
    pub per_offset: Vec<RecallScore>,
    pub mean_gain_exact: f64,
    pub spread_gain_exact: f64,
    pub mean_gain_bits: f64,
    pub spread_gain_bits: f64,
    /// Mean per-offset standard error of the exact gain — NOT divided by
    /// √offsets: the offsets share one block, so they are correlated.
    pub se_gain_exact: f64,
    /// Mean per-offset standard error of the bits gain (same convention).
    pub se_gain_bits: f64,
}

impl OffsetAveraged {
    /// Average ≥ [`MIN_OFFSETS`] readings; `None` below that (a
    /// single-offset number is refused, never reported).
    pub fn of(per_offset: Vec<RecallScore>) -> Option<Self> {
        if per_offset.len() < MIN_OFFSETS {
            return None;
        }
        let (mean_gain_exact, spread_gain_exact) =
            mean_spread(per_offset.iter().map(RecallScore::gain_exact));
        let (mean_gain_bits, spread_gain_bits) =
            mean_spread(per_offset.iter().map(RecallScore::gain_bits));
        let (se_gain_exact, _) = mean_spread(per_offset.iter().map(RecallScore::gain_exact_se));
        let (se_gain_bits, _) = mean_spread(per_offset.iter().map(RecallScore::gain_bits_se));
        Some(Self {
            se_gain_exact,
            se_gain_bits,
            per_offset,
            mean_gain_exact,
            spread_gain_exact,
            mean_gain_bits,
            spread_gain_bits,
        })
    }
}

fn mean_spread(xs: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut n, mut sum, mut lo, mut hi) = (0usize, 0.0f64, f64::INFINITY, f64::NEG_INFINITY);
    for x in xs {
        n += 1;
        sum += x;
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (sum / n as f64, hi - lo)
}

fn first_argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    best
}

fn log_sum_exp(row: &[f32]) -> f64 {
    let max = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    max + sum.ln()
}

/// Seeded 64-bit LCG (Knuth MMIX constants) — std-only, deterministic.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        )
    }

    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}
