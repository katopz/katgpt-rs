//! Row-relative, sink-exempt logit floor + b-bit logit codec (Issue 882 P2,
//! Research 586).
//!
//! # The primitive
//!
//! For one attention score row with non-sink max `m_r` and a width `w`:
//!
//! ```text
//! l̃ᵢ = m_r − min(w, m_r − lᵢ)  =  max(lᵢ, m_r − w)      (non-sink, unmasked i)
//! ```
//!
//! Because `m_r` is the row max this is a **floor**, not a cap: it raises the
//! far tail to `m_r − w` and bounds every non-sink logit to `[m_r − w, m_r]`.
//! A bounded domain is what makes the row codable in `b` bits with a known
//! step (`w / (2^b − 2)`), and a coded row's `exp` is a `2^b`-entry lookup
//! table instead of `n` transcendental calls.
//!
//! # The two exemptions (both load-bearing)
//!
//! - **Sinks** (positions `< n_sink`, the [`crate::kv_sink_window`]
//!   `SinkWindowPolicy::n_sink` convention) are legitimate outliers (trap 3):
//!   they are excluded from `m_r`, never floored, and never coded — the caller
//!   keeps their `f32` logits. Folding a sink into `m_r` would set the floor
//!   `w` below the SINK and floor most of the real context (pinned negative
//!   in Bench 888).
//! - **Masked** keys (`−∞`, the causal/padding convention) stay `−∞` and code
//!   to [`MASKED`]. A floor that ignored the mask would UN-mask them at mass
//!   `e^{−w}` each.
//!
//! # The closed-form envelope
//!
//! With `A = n_floored · e^{−w}` (each floored entry gains at most `e^{−w}` of
//! unnormalised mass relative to the non-sink max's `1`) and `Z ≥ 1`:
//!
//! - floor: `TV(p̃, p) ≤ A / (1 + A)`;
//! - code: every numerator and the denominator move by a factor in
//!   `[e^{−h}, e^{h}]` (`h` = half step, sinks exact), so per entry
//!   `|p̃ᵢ/pᵢ − 1| ≤ e^{2h} − 1` and `TV ≤ (e^{2h} − 1) / 2`.
//!
//! The width that GUARANTEES a floor budget `TV ≤ ε` for ANY row of `n`
//! non-sink keys is row-independent: [`min_width_for_tv`] `= ln(n/ε)`. The
//! range EMA ([`RangeEma`], the issue's `w_h`) predicts a typical row's
//! range instead; narrower rows code finer, but a row wider than the EMA
//! floors real entries and pays the envelope row by row.
//!
//! Pure arithmetic, no allocation, no dependency. Opt-in (`row_logit_floor`).

/// Code for a masked (`−∞`) key. Real codes are in `[−L, L]`, `L = 2^{b−1} − 1`.
pub const MASKED: i8 = i8::MIN;

/// What [`floor_row_sink_exempt`] measured on one row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowFloor {
    /// Max over non-sink, unmasked logits (`−∞` if there are none).
    pub m_r: f32,
    /// Max over the sink logits (`−∞` if `n_sink == 0`).
    pub sink_max: f32,
    /// Non-sink, unmasked entries raised to the floor.
    pub n_floored: usize,
}

impl RowFloor {
    /// The row's overall max (sinks included) — the softmax shift.
    #[inline]
    pub fn row_max(&self) -> f32 {
        self.m_r.max(self.sink_max)
    }
}

/// Floor `row[n_sink..]` to `m_r − width` in place (sinks and `−∞` untouched)
/// and return the row's measurements.
///
/// Two comparisons per logit (the max reduce + the floor select), branch-free
/// in the loop. `width = +∞` is the kill switch: the floor is `−∞` and the row
/// is returned bit-identical.
#[inline]
pub fn floor_row_sink_exempt(row: &mut [f32], n_sink: usize, width: f32) -> RowFloor {
    let s = n_sink.min(row.len());
    let (sinks, ctx) = row.split_at_mut(s);
    let sink_max = sinks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let m_r = ctx.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = m_r - width;
    let mut n_floored = 0usize;
    for x in ctx.iter_mut() {
        let v = *x;
        let live = v != f32::NEG_INFINITY;
        n_floored += usize::from(live & (v < floor));
        *x = if live { v.max(floor) } else { v };
    }
    RowFloor {
        m_r,
        sink_max,
        n_floored,
    }
}

/// The smallest width whose floor provably costs at most `tv` total
/// variation on ANY row of `n_ctx` non-sink keys: `ln(n_ctx / tv)`.
///
/// (`A ≤ n_ctx·e^{−w}` and `TV ≤ A/(1+A) ≤ A`.) `n_ctx == 0` ⇒ `0`.
#[inline]
pub fn min_width_for_tv(n_ctx: usize, tv: f32) -> f32 {
    match n_ctx {
        0 => 0.0,
        n => ((n as f32) / tv).ln().max(0.0),
    }
}

/// The closed-form softmax error envelope for one floored + coded row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    /// Total-variation bound of the floor alone: `A/(1+A)`.
    pub floor_tv: f32,
    /// Per-entry relative bound of the code alone: `e^{2h} − 1`.
    pub code_rel: f32,
}

impl Envelope {
    /// TV bound of floor + code together (triangle inequality).
    #[inline]
    pub fn total_tv(&self) -> f32 {
        self.floor_tv + 0.5 * self.code_rel
    }
}

/// A symmetric `b`-bit code over `[m_r − w, m_r]`: code `c ∈ [−L, L]` means
/// `l − m_r = c·step − w/2`, `step = w / (2L)`, so `+L` is the max exactly and
/// `−L` the floor exactly. Storage is `i8` for every `b ∈ 2..=8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogitCodec {
    levels: i32,
}

impl LogitCodec {
    /// A codec with `bits ∈ 2..=8` (clamped).
    #[inline]
    pub const fn new(bits: u8) -> Self {
        let b = match bits {
            0..=2 => 2,
            3..=8 => bits,
            _ => 8,
        };
        Self {
            levels: (1 << (b - 1)) - 1,
        }
    }

    /// `L = 2^{b−1} − 1`.
    #[inline]
    pub const fn levels(&self) -> i32 {
        self.levels
    }

    /// Code step for `width`.
    #[inline]
    pub fn step(&self, width: f32) -> f32 {
        width / (2 * self.levels) as f32
    }

    /// The envelope for a row floored at `width` with `n_floored` raised
    /// entries and coded by this codec.
    #[inline]
    pub fn envelope(&self, width: f32, n_floored: usize) -> Envelope {
        let a = n_floored as f32 * (-width).exp();
        Envelope {
            floor_tv: a / (1.0 + a),
            code_rel: self.step(width).exp_m1(),
        }
    }

    /// Code a FLOORED context segment (`row[n_sink..]` after
    /// [`floor_row_sink_exempt`]) into `out` (same length). `−∞` ⇒ [`MASKED`].
    ///
    /// # Panics
    /// If `out.len() != ctx.len()`.
    #[inline]
    pub fn encode_into(&self, ctx: &[f32], m_r: f32, width: f32, out: &mut [i8]) {
        assert_eq!(out.len(), ctx.len(), "encode_into: length mismatch");
        let k = self.kernel(m_r, width);
        for (o, &x) in out.iter_mut().zip(ctx) {
            *o = k.code(x);
        }
    }

    /// The per-row encode constants, shared by [`Self::encode_into`] and
    /// [`floored_coded_exp_inplace`] so the two can never disagree on a code.
    #[inline]
    fn kernel(&self, m_r: f32, width: f32) -> CodeKernel {
        let l = self.levels as f32;
        CodeKernel {
            l,
            inv: (2.0 * l) / width,
            centre: m_r - 0.5 * width,
        }
    }

    /// Decode codes back to logits relative to `m_r` (`MASKED` ⇒ `−∞`).
    ///
    /// # Panics
    /// If `out.len() != codes.len()`.
    #[inline]
    pub fn decode_into(&self, codes: &[i8], m_r: f32, width: f32, out: &mut [f32]) {
        assert_eq!(out.len(), codes.len(), "decode_into: length mismatch");
        let step = self.step(width);
        let base = m_r - 0.5 * width;
        for (o, &c) in out.iter_mut().zip(codes) {
            *o = match c {
                MASKED => f32::NEG_INFINITY,
                c => base + f32::from(c) * step,
            };
        }
    }

    /// Fill the 256-entry exp table indexed by `code as u8`:
    /// `lut[c] = e^{c·step − w/2 + (m_r − shift)}`, `lut[MASKED] = 0`.
    ///
    /// `m_r_minus_shift` is `m_r − row_max` (≤ 0), so sink rows stay finite.
    /// `2L + 1` exps per (head, width) — amortised whenever the row is longer.
    #[inline]
    pub fn exp_lut_into(&self, width: f32, m_r_minus_shift: f32, lut: &mut [f32; 256]) {
        let step = self.step(width);
        let base = m_r_minus_shift - 0.5 * width;
        lut.fill(0.0);
        for c in -self.levels..=self.levels {
            lut[(c as i8) as u8 as usize] = (base + c as f32 * step).exp();
        }
    }
}

/// Per-row encode constants (see [`LogitCodec::kernel`]).
#[derive(Clone, Copy)]
struct CodeKernel {
    l: f32,
    inv: f32,
    centre: f32,
}

impl CodeKernel {
    #[inline(always)]
    fn code(&self, x: f32) -> i8 {
        let c = ((x - self.centre) * self.inv)
            .round()
            .clamp(-self.l, self.l) as i8;
        if x == f32::NEG_INFINITY { MASKED } else { c }
    }
}

/// Floor, code and exp-table one score row IN PLACE — the fused
/// decode-attention form of [`floor_row_sink_exempt`] →
/// [`LogitCodec::encode_into`] → [`softmax_coded_into`], with no code buffer.
///
/// On return `row[i]` holds the UNNORMALISED softmax numerator relative to
/// the row max (sinks exact from `f32`, context from the table, masked `0`),
/// and the second value is their sum `Z`. `row[i] / Z` equals
/// [`softmax_coded_into`]'s output bit for bit (same entries, same summation
/// order). `lut` is scratch (refilled per row).
///
/// `width` must be finite and `> 0`: the kill switch belongs to the caller's
/// plain-softmax path, not to an infinite width here. A row with no live key
/// at all returns `Z = 0`, as a plain softmax would.
#[inline]
pub fn floored_coded_exp_inplace(
    row: &mut [f32],
    n_sink: usize,
    width: f32,
    codec: &LogitCodec,
    lut: &mut [f32; 256],
) -> (RowFloor, f32) {
    let rf = floor_row_sink_exempt(row, n_sink, width);
    let shift = rf.row_max();
    codec.exp_lut_into(width, rf.m_r - shift, lut);
    let s = n_sink.min(row.len());
    let (sinks, ctx) = row.split_at_mut(s);
    let mut z = 0.0f32;
    for x in sinks.iter_mut() {
        let e = (*x - shift).exp();
        *x = e;
        z += e;
    }
    let k = codec.kernel(rf.m_r, width);
    for x in ctx.iter_mut() {
        let e = lut[k.code(*x) as u8 as usize];
        *x = e;
        z += e;
    }
    (rf, z)
}

/// Softmax of a coded row: sinks from their `f32` logits (exact), context
/// from the exp table. Writes `p` for `[sinks.., ctx..]` into `out` and
/// returns the normaliser. `lut` must come from [`LogitCodec::exp_lut_into`]
/// with `m_r_minus_shift = m_r − shift`.
///
/// # Panics
/// If `out.len() != sinks.len() + codes.len()`.
#[inline]
pub fn softmax_coded_into(
    sinks: &[f32],
    codes: &[i8],
    shift: f32,
    lut: &[f32; 256],
    out: &mut [f32],
) -> f32 {
    assert_eq!(
        out.len(),
        sinks.len() + codes.len(),
        "softmax_coded_into: length mismatch"
    );
    let (os, oc) = out.split_at_mut(sinks.len());
    let mut z = 0.0f32;
    for (o, &s) in os.iter_mut().zip(sinks) {
        let e = (s - shift).exp();
        *o = e;
        z += e;
    }
    for (o, &c) in oc.iter_mut().zip(codes) {
        let e = lut[c as u8 as usize];
        *o = e;
        z += e;
    }
    let inv = 1.0 / z;
    for o in out.iter_mut() {
        *o *= inv;
    }
    z
}

/// Per-head EMA of the RAW non-sink row range (`max − min` over unmasked
/// keys) — the issue's `w_h`, latent state. Observe BEFORE flooring: a
/// floored row's range is `min(range, w)`, so an EMA fed floored rows can
/// only shrink.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeEma {
    width: f32,
    alpha: f32,
    primed: bool,
}

impl RangeEma {
    /// New EMA with weight `alpha ∈ (0, 1]` on the newest row.
    #[inline]
    pub const fn new(alpha: f32) -> Self {
        Self {
            width: 0.0,
            alpha,
            primed: false,
        }
    }

    /// Fold in one raw row. Rows with fewer than two unmasked context keys
    /// carry no range and are skipped.
    #[inline]
    pub fn observe(&mut self, row: &[f32], n_sink: usize) {
        let ctx = &row[n_sink.min(row.len())..];
        let (mut lo, mut hi, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0usize);
        for &x in ctx {
            let live = x != f32::NEG_INFINITY;
            lo = if live { lo.min(x) } else { lo };
            hi = if live { hi.max(x) } else { hi };
            n += usize::from(live);
        }
        if n < 2 {
            return;
        }
        let r = hi - lo;
        match self.primed {
            true => self.width += self.alpha * (r - self.width),
            false => {
                self.width = r;
                self.primed = true;
            }
        }
    }

    /// The smoothed width (`None` before the first observed row).
    #[inline]
    pub fn width(&self) -> Option<f32> {
        self.primed.then_some(self.width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax_f64(x: &[f32]) -> Vec<f64> {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = x
            .iter()
            .map(|&v| match v == f32::NEG_INFINITY {
                true => 0.0,
                false => ((v as f64) - m).exp(),
            })
            .collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|v| v / z).collect()
    }

    #[test]
    fn infinite_width_is_bit_identical() {
        let orig = [3.0f32, -1.5, 0.25, f32::NEG_INFINITY, -40.0, 7.0];
        let mut row = orig;
        let rf = floor_row_sink_exempt(&mut row, 1, f32::INFINITY);
        assert_eq!(rf.n_floored, 0);
        for (a, b) in orig.iter().zip(&row) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn floor_skips_sinks_and_masked() {
        let mut row = [50.0f32, -30.0, 2.0, -9.0, f32::NEG_INFINITY, 0.0];
        let rf = floor_row_sink_exempt(&mut row, 2, 5.0);
        assert_eq!(rf.m_r, 2.0);
        assert_eq!(rf.sink_max, 50.0);
        assert_eq!(rf.n_floored, 1); // -9 → -3 only
        assert_eq!(row, [50.0, -30.0, 2.0, -3.0, f32::NEG_INFINITY, 0.0]);
        assert_eq!(rf.row_max(), 50.0);
    }

    #[test]
    fn all_masked_context_is_a_noop() {
        let mut row = [1.0f32, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let rf = floor_row_sink_exempt(&mut row, 1, 4.0);
        assert_eq!(rf.m_r, f32::NEG_INFINITY);
        assert_eq!(rf.n_floored, 0);
        assert!(row[1..].iter().all(|&x| x == f32::NEG_INFINITY));
    }

    #[test]
    fn codec_endpoints_are_exact() {
        for bits in [2u8, 4, 6, 8] {
            let c = LogitCodec::new(bits);
            let (m, w) = (3.5f32, 12.0f32);
            let mut codes = [0i8; 3];
            c.encode_into(&[m, m - w, f32::NEG_INFINITY], m, w, &mut codes);
            assert_eq!(codes, [c.levels() as i8, -(c.levels() as i8), MASKED]);
            let mut back = [0.0f32; 3];
            c.decode_into(&codes, m, w, &mut back);
            assert!((back[0] - m).abs() < 1e-5 && (back[1] - (m - w)).abs() < 1e-5);
            assert_eq!(back[2], f32::NEG_INFINITY);
        }
    }

    #[test]
    fn codec_roundtrip_within_half_step() {
        let c = LogitCodec::new(6);
        let (m, w) = (0.0f32, 10.0f32);
        let xs: Vec<f32> = (0..=200).map(|i| m - w * i as f32 / 200.0).collect();
        let mut codes = vec![0i8; xs.len()];
        let mut back = vec![0.0f32; xs.len()];
        c.encode_into(&xs, m, w, &mut codes);
        c.decode_into(&codes, m, w, &mut back);
        let h = 0.5 * c.step(w);
        for (a, b) in xs.iter().zip(&back) {
            assert!((a - b).abs() <= h + 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn lut_softmax_matches_decoded_softmax_and_envelope() {
        let c = LogitCodec::new(8);
        let sinks = [9.0f32, 4.0];
        let mut ctx: Vec<f32> = (0..300)
            .map(|i| ((i * 37 % 101) as f32) * 0.2 - 18.0)
            .collect();
        ctx[7] = f32::NEG_INFINITY;
        let raw: Vec<f32> = sinks.iter().chain(&ctx).copied().collect();
        let mut row = raw.clone();
        let w = 12.0;
        let rf = floor_row_sink_exempt(&mut row, 2, w);
        let mut codes = vec![0i8; ctx.len()];
        c.encode_into(&row[2..], rf.m_r, w, &mut codes);
        let mut lut = [0.0f32; 256];
        let shift = rf.row_max();
        c.exp_lut_into(w, rf.m_r - shift, &mut lut);
        let mut p = vec![0.0f32; raw.len()];
        softmax_coded_into(&row[..2], &codes, shift, &lut, &mut p);
        assert_eq!(p[2 + 7], 0.0, "masked key must stay at zero mass");
        let exact = softmax_f64(&raw);
        let tv: f64 = 0.5
            * p.iter()
                .zip(&exact)
                .map(|(a, b)| (*a as f64 - b).abs())
                .sum::<f64>();
        let env = c.envelope(w, rf.n_floored);
        assert!(rf.n_floored > 0, "fixture must exercise the floor");
        assert!(
            tv <= env.total_tv() as f64 + 1e-6,
            "tv {tv} > {}",
            env.total_tv()
        );
    }

    #[test]
    fn fused_inplace_is_bit_identical_to_the_composition() {
        for bits in [6u8, 8] {
            let c = LogitCodec::new(bits);
            let mut raw: Vec<f32> = (0..517)
                .map(|i| ((i * 53 % 211) as f32) * 0.13 - 20.0)
                .collect();
            raw[0] = 31.0; // sink outlier
            raw[40] = f32::NEG_INFINITY;
            let (n_sink, w) = (4usize, min_width_for_tv(raw.len() - 4, 1e-2));
            // composition
            let mut row = raw.clone();
            let rf = floor_row_sink_exempt(&mut row, n_sink, w);
            let mut codes = vec![0i8; raw.len() - n_sink];
            c.encode_into(&row[n_sink..], rf.m_r, w, &mut codes);
            let mut lut = [0.0f32; 256];
            let shift = rf.row_max();
            c.exp_lut_into(w, rf.m_r - shift, &mut lut);
            let mut p = vec![0.0f32; raw.len()];
            let z_ref = softmax_coded_into(&row[..n_sink], &codes, shift, &lut, &mut p);
            // fused
            let mut fused = raw.clone();
            let mut lut2 = [0.0f32; 256];
            let (rf2, z) = floored_coded_exp_inplace(&mut fused, n_sink, w, &c, &mut lut2);
            assert_eq!(rf, rf2);
            assert_eq!(z.to_bits(), z_ref.to_bits());
            let inv = 1.0 / z;
            for (a, b) in fused.iter().zip(&p) {
                assert_eq!((a * inv).to_bits(), b.to_bits());
            }
            assert_eq!(fused[40], 0.0, "masked key stays at zero mass");
        }
    }

    #[test]
    fn min_width_guarantees_the_floor_budget() {
        let w = min_width_for_tv(65_536, 1e-3);
        assert!((w - (65_536.0f32 / 1e-3).ln()).abs() < 1e-4);
        let env = LogitCodec::new(8).envelope(w, 65_536);
        assert!(env.floor_tv <= 1e-3 * 1.0001);
        assert_eq!(min_width_for_tv(0, 1e-3), 0.0);
    }

    #[test]
    fn range_ema_reads_raw_rows_and_skips_empty() {
        let mut e = RangeEma::new(0.5);
        assert_eq!(e.width(), None);
        e.observe(&[100.0, f32::NEG_INFINITY, 1.0], 1); // one live key: skipped
        assert_eq!(e.width(), None);
        e.observe(&[100.0, 4.0, -6.0, f32::NEG_INFINITY], 1);
        assert_eq!(e.width(), Some(10.0));
        e.observe(&[100.0, 0.0, -2.0], 1);
        assert_eq!(e.width(), Some(6.0));
    }
}
