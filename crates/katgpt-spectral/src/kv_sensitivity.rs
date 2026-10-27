//! Per-channel KV quantization sensitivity probe — Issue 960 T0
//! (Research 633: arXiv:2606.09864, the alignment-collapse law).
//!
//! The paper's Per-Channel Reduction (PCR) diagnostic: degrade ONE KV
//! channel at a time against its f16 reference, rank channels by
//! BEHAVIORAL damage (how far the attention pattern moves), never MSE
//! alone — the features carrying trained behavior occupy a subspace
//! 10²–10³× more quantization-fragile than the space a PPL/MSE average
//! covers, which is exactly why aggregate-flat / behavior-flipped reads
//! dominate at low bits (the lossy-surface gate's external validation
//! #4, Issue-750-T3).
//!
//! Two modes over one metric core:
//!
//! - **Reduction** (the paper's ablation): each channel's column is
//!   degraded by a deterministic uniform min–max scalar quantizer at
//!   [`ProbeConfig::bits`] and the damage attributed to that channel.
//!   Backend-agnostic — it probes the per-channel GRAIN the whole
//!   bit-width family shares, before any backend's joint codebook enters.
//! - **Attribution** (the "over `QuantizedKVCache`" reading): the caller
//!   drives any real backend (store the reference rows, dequantize them
//!   back) and hands both tensors in via the `*_dequant` field; the probe
//!   decomposes the joint error EXACTLY per channel — for scores,
//!   `s'_ij − s_ij = Σ_c q_i[c]·e_j[c]` splits into the rank-1
//!   per-channel terms `q_i[c]·e_j[c]`, so there is no approximation and
//!   no re-quantization. The backend trait itself lives with the caller's
//!   loop; this module stays dep-free tensor math.
//!
//! Softmax-free by design: K-side damage is measured on PRE-softmax
//! score geometry (top-k attention-target overlap, relative distortion,
//! k=1 pick flips); the V side reads the caller's attention weights
//! (pass `None` for uniform) — the probe never normalizes a
//! distribution itself, per the workspace sigmoid-not-softmax rule.
//!
//! V-side damage is the mean relative context distortion
//! `|Δy_c| / (‖y‖₂ + ε)`: value channels move the attention OUTPUT's
//! content, not the selection pattern, so there is no top-k to overlap.
//! [`ChannelDamage::pick_flip_rate`] is `0.0` on the V side (not
//! meaningful there); downstream pick-flip reads are T3's eval axis.
//!
//! Determinism (Issue 940 law): every reduction is a sequential
//! ascending-index f32 fold over fixed layouts (no parallel lanes, no
//! HashMap iteration), cross-query sums accumulate in f64 in a fixed
//! order, and top-k / argmax ties resolve FIRST-INDEX-WINS by
//! construction (strictly-greater scans, and the streaming top-k
//! displaces the largest index among tied minima so the surviving set
//! matches an index-ascending stable sort exactly). Same inputs →
//! byte-identical output on every platform. This instrument is
//! LOCAL-ONLY (nothing it returns crosses a sync/quorum boundary), so
//! plain sequential folds suffice — reclassify at any call site that
//! ever commits a probe result.
//!
//! Offline instrument: the caller allocates one [`ProbeScratch`] and
//! reuses it across probes; the probe allocates only its returned
//! `Vec<ChannelDamage>`. T0's real-model run half (~20 calibration
//! prompts, cached per (model, backend, bits)) lives with the consumer
//! capture harness — this module takes captured tensors in, damages
//! out. All inputs must be finite; non-finite entries are a caller
//! contract violation (scores could otherwise produce NaN, which sorts
//! as a tie rather than panicking — treated as equal damage order).

use std::cmp::Ordering;

/// Probe configuration.
#[derive(Debug, Clone, Copy)]
pub struct ProbeConfig {
    /// Bits for the reduction mode's per-channel uniform min–max scalar
    /// quantizer. Clamped to `[1, 24]` (24 already exhausts f32 mantissa
    /// resolution for a min–max grid). Attribution mode ignores this — it
    /// reads the caller's dequantized tensors.
    pub bits: u32,
    /// Attention-target overlap depth `k` for the damage metric. Clamped
    /// to `>= 1` and `<= seq` at probe time. With `k = 1`, `damage` is
    /// exactly the pick-flip rate.
    pub top_k: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self { bits: 2, top_k: 1 }
    }
}

/// Key-side probe input. `q` is `[n_q × d]`, `k_ref` is `[seq × d]`,
/// both row-major and finite. The mode is selected by [`KeyTensors::k_dequant`].
#[derive(Debug, Clone, Copy)]
pub struct KeyTensors<'a> {
    /// Query rows against which the attention scores are measured.
    pub q: &'a [f32],
    /// Number of query rows.
    pub n_q: usize,
    /// Reference (f16-host) key rows.
    pub k_ref: &'a [f32],
    /// `Some(dq)` = **attribution mode**: `dq` is the caller's
    /// backend-dequantized key cache ([`seq × d`]), and the joint error
    /// `dq − k_ref` is decomposed exactly per channel. `None` =
    /// **reduction mode**: the internal uniform quantizer at
    /// [`ProbeConfig::bits`] degrades one column at a time.
    pub k_dequant: Option<&'a [f32]>,
    /// Sequence length (key rows).
    pub seq: usize,
    /// Head dimension (channels per row).
    pub d: usize,
}

/// Value-side probe input. `v_ref` is `[seq × d]`, row-major and finite.
/// The mode is selected by [`ValueTensors::v_dequant`].
#[derive(Debug, Clone, Copy)]
pub struct ValueTensors<'a> {
    /// Reference (f16-host) value rows.
    pub v_ref: &'a [f32],
    /// `Some(dq)` = **attribution mode** (caller's backend-dequantized
    /// value cache); `None` = **reduction mode** (internal quantizer at
    /// [`ValueTensors::bits`]).
    pub v_dequant: Option<&'a [f32]>,
    /// Number of query rows the weights carry.
    pub n_q: usize,
    /// Sequence length (value rows).
    pub seq: usize,
    /// Head dimension (channels per row).
    pub d: usize,
    /// The caller's attention weights `[n_q × seq]` row-major, or `None`
    /// for uniform `1/seq`. Never a softmax computed inside this module.
    pub weights: Option<&'a [f32]>,
    /// Reduction-mode quantizer bits (clamped `[1, 24]`; ignored in
    /// attribution mode).
    pub bits: u32,
}

/// Per-channel behavioral damage, sorted damage-desc / channel-asc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelDamage {
    /// The channel index (its column in the `[seq × d]` row-major layout).
    pub channel: usize,
    /// K side: `1 − mean top-k attention-target overlap` — the primary
    /// rank key. `0.0` = the pattern is untouched, `1.0` = every query's
    /// targets moved. V side: the mean relative context distortion (see
    /// the module doc) — same field, different metric, documented.
    pub damage: f32,
    /// K side: mean over queries of
    /// `‖Δrow‖₂ / (‖row − mean(row)‖₂ + ε)` — the scale-free score-space
    /// distortion companion. V side: identical to `damage`.
    pub mean_rel_distortion: f32,
    /// Fraction of queries whose argmax attention target flipped (always
    /// the k=1 read, independent of `top_k`). `0.0` on the V side.
    pub pick_flip_rate: f32,
}

/// Reusable scratch for the probe family. One per thread, sized once via
/// [`ProbeScratch::ensure`], reused across probes and channels.
#[derive(Debug, Default, Clone)]
pub struct ProbeScratch {
    ref_scores: Vec<f32>,
    ref_topk: Vec<usize>,
    ref_argmax: Vec<usize>,
    ref_devnorm: Vec<f32>,
    sort_idx: Vec<usize>,
    kbuf_val: Vec<f32>,
    kbuf_idx: Vec<usize>,
    err_col: Vec<f32>,
    col_buf: Vec<f32>,
    vals: Vec<f32>,
    // joint_key_read-only buffers (sized locally — `ensure` does not know
    // the channel count):
    /// All channels' quantization error columns, `[d × seq]`, filled
    /// channel-ascending before the query loop.
    err_matrix: Vec<f32>,
    /// One query's score-space delta `Σ_c q[i][c]·err_c[j]`, accumulated
    /// in f64 over ascending channels (Issue-940 fixed-order, LOCAL-ONLY).
    delta_row: Vec<f64>,
    /// The perturbed score row `ref + delta` for the query in flight.
    pert_scores: Vec<f32>,
}

impl ProbeScratch {
    /// Size every scratch buffer for the given probe shape. Cheap to call
    /// repeatedly with the same shape (resize to the same length is a
    /// no-op); shrinks stale oversized buffers so a small probe after a
    /// big one does not pin the old footprint.
    pub fn ensure(&mut self, n_q: usize, seq: usize, k: usize) {
        self.ref_scores.resize(n_q * seq, 0.0);
        self.ref_topk.resize(n_q * k, 0);
        self.ref_argmax.resize(n_q, 0);
        self.ref_devnorm.resize(n_q, 0.0);
        self.sort_idx.resize(seq, 0);
        self.kbuf_val.resize(k, 0.0);
        self.kbuf_idx.resize(k, 0);
        self.err_col.resize(seq, 0.0);
        self.col_buf.resize(seq, 0.0);
        self.vals.resize(n_q, 0.0);
    }
}

/// Per-channel uniform min–max scalar quantization error at `bits` bits:
/// `out[j] = q(col[j]) − col[j]`, where `q` snaps to the `2^bits`-level
/// grid spanning `[min(col), max(col)]` (the column minimum always lands
/// exactly on-grid; the maximum within an ulp; interior points carry the
/// error). Constant columns are exact (zero
/// error at any bit width). `bits` is clamped to `[1, 24]`. Deterministic:
/// fixed-order f32 ops, round-half-away-from-zero grid snapping, grid
/// index clamped into `[0, 2^bits − 1]`. Public because the capture
/// harness builds probe fixtures and backend comparisons from it.
pub fn uniform_quant_error_into(col: &[f32], bits: u32, out: &mut [f32]) {
    assert_eq!(
        col.len(),
        out.len(),
        "uniform_quant_error_into: col and out must have the same length"
    );
    let n = col.len();
    if n == 0 {
        return;
    }
    let bits = bits.clamp(1, 24);
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in col {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if mn == mx {
        for e in out[..n].iter_mut() {
            *e = 0.0;
        }
        return;
    }
    let levels = 1u64 << bits;
    let scale = (mx - mn) / ((levels - 1) as f32);
    let inv = 1.0f32 / scale;
    let lmax = (levels - 1) as i64;
    for (e, &x) in out[..n].iter_mut().zip(col.iter()) {
        let idx = (((x - mn) * inv).round() as i64).clamp(0, lmax);
        let qv = idx as f32 * scale + mn;
        *e = qv - x;
    }
}

/// K-side probe. Reduction or attribution mode per
/// [`KeyTensors::k_dequant`]; returns the per-channel damages sorted
/// damage-desc / channel-asc.
///
/// # Panics
/// Panics when any tensor's length does not match its declared shape.
pub fn probe_key_channels(
    t: KeyTensors,
    cfg: &ProbeConfig,
    scratch: &mut ProbeScratch,
) -> Vec<ChannelDamage> {
    assert_eq!(t.q.len(), t.n_q * t.d, "q must be [n_q × d] row-major");
    assert_eq!(
        t.k_ref.len(),
        t.seq * t.d,
        "k_ref must be [seq × d] row-major"
    );
    if let Some(dq) = t.k_dequant {
        assert_eq!(
            dq.len(),
            t.seq * t.d,
            "k_dequant must be [seq × d] row-major"
        );
    }
    key_damage(&t, cfg, scratch)
}

/// V-side probe. Reduction or attribution mode per
/// [`ValueTensors::v_dequant`]; returns the per-channel context
/// distortions sorted damage-desc / channel-asc.
///
/// # Panics
/// Panics when any tensor's length does not match its declared shape.
pub fn probe_value_channels(t: ValueTensors, scratch: &mut ProbeScratch) -> Vec<ChannelDamage> {
    assert_eq!(
        t.v_ref.len(),
        t.seq * t.d,
        "v_ref must be [seq × d] row-major"
    );
    if let Some(dq) = t.v_dequant {
        assert_eq!(
            dq.len(),
            t.seq * t.d,
            "v_dequant must be [seq × d] row-major"
        );
    }
    if let Some(ws) = t.weights {
        assert_eq!(
            ws.len(),
            t.n_q * t.seq,
            "weights must be [n_q × seq] row-major"
        );
    }
    value_damage(&t, scratch)
}

/// Shared K-side reference pass: reference scores (sequential ascending f32
/// dot folds), per-query deviation norms, argmax (first-index ties) and the
/// (value-desc, index-asc) top-k selection — written into `scratch`. Both
/// [`key_damage`] and [`joint_key_read`] consume exactly this state, so their
/// damage verdicts share one reference computation.
fn key_reference_pass(t: &KeyTensors, k: usize, scratch: &mut ProbeScratch) {
    let (n_q, seq, d) = (t.n_q, t.seq, t.d);
    for i in 0..n_q {
        let base = i * seq;
        for j in 0..seq {
            let qrow = &t.q[i * d..i * d + d];
            let krow = &t.k_ref[j * d..j * d + d];
            let mut acc: f32 = 0.0;
            for c in 0..d {
                acc += qrow[c] * krow[c];
            }
            scratch.ref_scores[base + j] = acc;
        }
        let mut sum: f64 = 0.0;
        for j in 0..seq {
            sum += scratch.ref_scores[base + j] as f64;
        }
        let mean = (sum / seq as f64) as f32;
        let mut dev2: f64 = 0.0;
        for j in 0..seq {
            let dev = scratch.ref_scores[base + j] - mean;
            dev2 += (dev * dev) as f64;
        }
        scratch.ref_devnorm[i] = (dev2 as f32).sqrt();
        let mut am = 0usize;
        let mut bv = scratch.ref_scores[base];
        for j in 1..seq {
            if scratch.ref_scores[base + j] > bv {
                bv = scratch.ref_scores[base + j];
                am = j;
            }
        }
        scratch.ref_argmax[i] = am;
        for (pos, s) in scratch.sort_idx.iter_mut().enumerate().take(seq) {
            *s = pos;
        }
        let vals = &scratch.ref_scores[base..base + seq];
        scratch.sort_idx[..seq].sort_by(|&a, &b| {
            vals[b]
                .partial_cmp(&vals[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        scratch.ref_topk[i * k..i * k + k].copy_from_slice(&scratch.sort_idx[..k]);
    }
}

/// Aggregate K-side read of [`joint_key_read`] — the complement of the
/// per-channel [`ChannelDamage`]: one damage verdict for the WHOLE
/// degradation pattern, not one per channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointKeyRead {
    /// `1 − mean top-k attention-target overlap` under the joint
    /// degradation. `0.0` = the pattern is untouched.
    pub damage: f32,
    /// Mean over queries of `‖Δrow‖₂ / (‖row − mean(row)‖₂ + ε)` for the
    /// joint delta — the scale-free distortion companion.
    pub mean_rel_distortion: f32,
    /// Fraction of queries whose argmax attention target flipped under the
    /// joint degradation (always the k=1 read).
    pub pick_flip_rate: f32,
}

/// The GOAT read for sensitivity-guided bit allocation (Issue 960 T2): the
/// AGGREGATE behavioral damage when ALL key channels are degraded AT ONCE,
/// each at its own allocated bit width — the quantity an allocation arm
/// actually buys retention on. [`probe_key_channels`] measures per-channel
/// MARGINAL damage (one channel degraded at a time); the joint read is its
/// complement, not its sum (score-space deltas from different channels
/// interfere — `s' − s = Σ_c q_i[c]·e_c[j]` with cross terms intact).
///
/// Inputs mirror [`probe_key_channels`]'s reduction posture: `t.k_dequant`
/// must be `None` (pass the backend's own dequantized cache via attribution
/// mode instead when a real backend is under test). `bits_per_channel[c]`
/// is the min–max grid width for channel `c` — one entry per channel, each
/// in `[1, 24]` (typically an [`crate::kv_sensitivity_alloc`] allocation
/// table). `cfg.bits` is IGNORED here (the table overrides it); only
/// `cfg.top_k` is read.
///
/// Determinism: the per-query delta accumulates in f64 over ASCENDING
/// channels (fixed order, Issue-940 LOCAL-ONLY reclassification recorded at
/// this site), the perturbed score row is `ref + delta` in f32, and the
/// argmax / streaming top-k scans are the probe's own first-index-tie laws
/// against the shared reference pass. Same inputs → byte-identical output.
/// Calibration-class like the probe (one-shot per allocation decision), not
/// a per-token serving path.
///
/// # Panics
/// Panics when any tensor's length does not match its declared shape, when
/// `bits_per_channel` length ≠ `d`, or when any entry of
/// `bits_per_channel` is 0 (a zero-width grid is a caller bug — the
/// allocator never emits one).
pub fn joint_key_read(
    t: KeyTensors,
    bits_per_channel: &[u8],
    cfg: &ProbeConfig,
    scratch: &mut ProbeScratch,
) -> JointKeyRead {
    assert!(
        t.k_dequant.is_none(),
        "joint_key_read is the reduction posture: k_dequant must be None (use attribution mode for a real backend)"
    );
    assert_eq!(t.q.len(), t.n_q * t.d, "q must be [n_q × d] row-major");
    assert_eq!(
        t.k_ref.len(),
        t.seq * t.d,
        "k_ref must be [seq × d] row-major"
    );
    assert_eq!(
        bits_per_channel.len(),
        t.d,
        "bits_per_channel must have one entry per channel"
    );
    let (n_q, seq, d) = (t.n_q, t.seq, t.d);
    if seq == 0 || d == 0 || n_q == 0 {
        return JointKeyRead {
            damage: 0.0,
            mean_rel_distortion: 0.0,
            pick_flip_rate: 0.0,
        };
    }
    assert!(
        bits_per_channel.iter().all(|&b| (1..=24).contains(&b)),
        "bits_per_channel entries must be in [1, 24]"
    );
    let k = cfg.top_k.max(1).min(seq);
    scratch.ensure(n_q, seq, k);
    scratch.err_matrix.resize(d * seq, 0.0);
    scratch.delta_row.resize(seq, 0.0);
    scratch.pert_scores.resize(seq, 0.0);

    key_reference_pass(&t, k, scratch);

    // Per-channel quantization errors, channels ascending (fixed order).
    for (c, &b) in bits_per_channel.iter().enumerate() {
        for j in 0..seq {
            scratch.col_buf[j] = t.k_ref[j * d + c];
        }
        uniform_quant_error_into(
            &scratch.col_buf[..seq],
            b as u32,
            &mut scratch.err_col[..seq],
        );
        scratch.err_matrix[c * seq..(c + 1) * seq].copy_from_slice(&scratch.err_col[..seq]);
    }

    let mut overlap_sum: f64 = 0.0;
    let mut distort_sum: f64 = 0.0;
    let mut flip_sum: f64 = 0.0;
    for i in 0..n_q {
        // Score-space delta for this query: Σ_c q[i][c]·e_c[j], f64 over
        // ascending channels. Always accumulated (no zero-skip): skipping a
        // +0.0 term can flip the sign bit of an all-zero accumulator and
        // that is a tie-class difference under first-index ties.
        for v in scratch.delta_row[..seq].iter_mut() {
            *v = 0.0;
        }
        for c in 0..d {
            let qc = t.q[i * d + c] as f64;
            let err = &scratch.err_matrix[c * seq..(c + 1) * seq];
            for (dv, &e) in scratch.delta_row[..seq].iter_mut().zip(err.iter()) {
                *dv += qc * e as f64;
            }
        }
        for j in 0..seq {
            scratch.pert_scores[j] = scratch.ref_scores[i * seq + j] + scratch.delta_row[j] as f32;
        }
        // Relative distortion of the perturbed row vs the reference's
        // deviation norm (the probe's scale-free companion, unchanged).
        let mut num2: f64 = 0.0;
        for &dl in scratch.delta_row[..seq].iter() {
            let delta = dl as f32;
            num2 += (delta * delta) as f64;
        }
        let dist = (num2 as f32).sqrt() / (scratch.ref_devnorm[i] + 1e-12);
        distort_sum += dist as f64;
        // k=1 pick flip: argmax of the perturbed row, first-index ties.
        let mut bv = scratch.pert_scores[0];
        let mut am = 0usize;
        for j in 1..seq {
            if scratch.pert_scores[j] > bv {
                bv = scratch.pert_scores[j];
                am = j;
            }
        }
        if am != scratch.ref_argmax[i] {
            flip_sum += 1.0;
        }
        // Streaming top-k — the probe's own tie law (strictly-greater
        // replacement; among tied minima the LARGEST index is displaced).
        for pos in 0..k {
            scratch.kbuf_val[pos] = scratch.pert_scores[pos];
            scratch.kbuf_idx[pos] = pos;
        }
        for j in k..seq {
            let v = scratch.pert_scores[j];
            let mut min_pos = 0usize;
            let mut min_val = scratch.kbuf_val[0];
            for pos in 1..k {
                let tv = scratch.kbuf_val[pos];
                if tv < min_val
                    || (tv == min_val && scratch.kbuf_idx[pos] > scratch.kbuf_idx[min_pos])
                {
                    min_val = tv;
                    min_pos = pos;
                }
            }
            if v > min_val {
                scratch.kbuf_val[min_pos] = v;
                scratch.kbuf_idx[min_pos] = j;
            }
        }
        for pos in 0..k {
            let ridx = scratch.ref_topk[i * k + pos];
            if scratch.kbuf_idx[..k].contains(&ridx) {
                overlap_sum += 1.0;
            }
        }
    }
    let nqf = n_q as f64;
    JointKeyRead {
        damage: 1.0 - (overlap_sum / (nqf * k as f64)) as f32,
        mean_rel_distortion: (distort_sum / nqf) as f32,
        pick_flip_rate: (flip_sum / nqf) as f32,
    }
}

/// Shared K-side core. Mode per `KeyTensors::k_dequant`.
fn key_damage(t: &KeyTensors, cfg: &ProbeConfig, scratch: &mut ProbeScratch) -> Vec<ChannelDamage> {
    let (n_q, seq, d) = (t.n_q, t.seq, t.d);
    if seq == 0 || d == 0 {
        return Vec::new();
    }
    let k = cfg.top_k.max(1).min(seq);
    let bits = cfg.bits.clamp(1, 24);
    scratch.ensure(n_q, seq, k);
    if n_q == 0 {
        // No queries — no evidence: every channel reads zero damage.
        return (0..d)
            .map(|c| ChannelDamage {
                channel: c,
                damage: 0.0,
                mean_rel_distortion: 0.0,
                pick_flip_rate: 0.0,
            })
            .collect();
    }

    // Reference pass: scores, per-query mean/deviation norm, argmax and
    // top-k indices — all channel-independent, computed once.
    key_reference_pass(t, k, scratch);

    let mut out: Vec<ChannelDamage> = Vec::with_capacity(d);
    for c in 0..d {
        match t.k_dequant {
            None => {
                for j in 0..seq {
                    scratch.col_buf[j] = t.k_ref[j * d + c];
                }
                uniform_quant_error_into(
                    &scratch.col_buf[..seq],
                    bits,
                    &mut scratch.err_col[..seq],
                );
            }
            Some(dq) => {
                for j in 0..seq {
                    scratch.err_col[j] = dq[j * d + c] - t.k_ref[j * d + c];
                }
            }
        }
        let mut overlap_sum: f64 = 0.0;
        let mut distort_sum: f64 = 0.0;
        let mut flip_sum: f64 = 0.0;
        for i in 0..n_q {
            let a = t.q[i * d + c];
            let base = i * seq;
            // Relative distortion of the perturbed score row.
            let mut num2: f64 = 0.0;
            for j in 0..seq {
                let delta = a * scratch.err_col[j];
                num2 += (delta * delta) as f64;
            }
            let dist = (num2 as f32).sqrt() / (scratch.ref_devnorm[i] + 1e-12);
            distort_sum += dist as f64;
            // k=1 pick flip: argmax of the perturbed row, first-index ties.
            let mut bv = scratch.ref_scores[base] + a * scratch.err_col[0];
            let mut am = 0usize;
            for j in 1..seq {
                let v = scratch.ref_scores[base + j] + a * scratch.err_col[j];
                if v > bv {
                    bv = v;
                    am = j;
                }
            }
            if am != scratch.ref_argmax[i] {
                flip_sum += 1.0;
            }
            // Streaming top-k of the perturbed row. Strictly-greater
            // replacement keeps first-index winners; among tied minima the
            // LARGEST index is displaced, which reproduces the reference
            // (value desc, index asc) selection set exactly.
            for pos in 0..k {
                scratch.kbuf_val[pos] = scratch.ref_scores[base + pos] + a * scratch.err_col[pos];
                scratch.kbuf_idx[pos] = pos;
            }
            for j in k..seq {
                let v = scratch.ref_scores[base + j] + a * scratch.err_col[j];
                let mut min_pos = 0usize;
                let mut min_val = scratch.kbuf_val[0];
                for pos in 1..k {
                    let tv = scratch.kbuf_val[pos];
                    if tv < min_val
                        || (tv == min_val && scratch.kbuf_idx[pos] > scratch.kbuf_idx[min_pos])
                    {
                        min_val = tv;
                        min_pos = pos;
                    }
                }
                if v > min_val {
                    scratch.kbuf_val[min_pos] = v;
                    scratch.kbuf_idx[min_pos] = j;
                }
            }
            for pos in 0..k {
                let ridx = scratch.ref_topk[i * k + pos];
                if scratch.kbuf_idx[..k].contains(&ridx) {
                    overlap_sum += 1.0;
                }
            }
        }
        let nqf = n_q as f64;
        let overlap_denom = nqf * k as f64;
        out.push(ChannelDamage {
            channel: c,
            damage: 1.0 - (overlap_sum / overlap_denom) as f32,
            mean_rel_distortion: (distort_sum / nqf) as f32,
            pick_flip_rate: (flip_sum / nqf) as f32,
        });
    }
    out.sort_by(|a, b| {
        b.damage
            .partial_cmp(&a.damage)
            .unwrap_or(Ordering::Equal)
            .then(a.channel.cmp(&b.channel))
    });
    out
}

/// Shared V-side core. Mode per `ValueTensors::v_dequant`.
fn value_damage(t: &ValueTensors, scratch: &mut ProbeScratch) -> Vec<ChannelDamage> {
    let (n_q, seq, d) = (t.n_q, t.seq, t.d);
    if seq == 0 || d == 0 {
        return Vec::new();
    }
    scratch.ensure(n_q, seq, 1);
    if n_q == 0 {
        return (0..d)
            .map(|c| ChannelDamage {
                channel: c,
                damage: 0.0,
                mean_rel_distortion: 0.0,
                pick_flip_rate: 0.0,
            })
            .collect();
    }
    let inv_seq = 1.0f32 / seq as f32;
    let w_at = |i: usize, j: usize| -> f32 {
        match t.weights {
            Some(ws) => ws[i * seq + j],
            None => inv_seq,
        }
    };

    // Reference context norms ‖y_i‖₂, y_i[c] = Σ_j w_ij·v_j[c] — fixed
    // i → c → j ascending fold order.
    for i in 0..n_q {
        let mut ysum2: f64 = 0.0;
        for c in 0..d {
            let mut acc: f32 = 0.0;
            for j in 0..seq {
                acc += w_at(i, j) * t.v_ref[j * d + c];
            }
            ysum2 += (acc * acc) as f64;
        }
        scratch.vals[i] = (ysum2 as f32).sqrt();
    }

    let mut out: Vec<ChannelDamage> = Vec::with_capacity(d);
    for c in 0..d {
        match t.v_dequant {
            None => {
                for j in 0..seq {
                    scratch.col_buf[j] = t.v_ref[j * d + c];
                }
                uniform_quant_error_into(
                    &scratch.col_buf[..seq],
                    t.bits.clamp(1, 24),
                    &mut scratch.err_col[..seq],
                );
            }
            Some(dq) => {
                for j in 0..seq {
                    scratch.err_col[j] = dq[j * d + c] - t.v_ref[j * d + c];
                }
            }
        }
        let mut rel_sum: f64 = 0.0;
        for i in 0..n_q {
            let mut g: f32 = 0.0;
            for j in 0..seq {
                g += w_at(i, j) * scratch.err_col[j];
            }
            let rel = g.abs() / (scratch.vals[i] + 1e-12);
            rel_sum += rel as f64;
        }
        let damage = (rel_sum / n_q as f64) as f32;
        out.push(ChannelDamage {
            channel: c,
            damage,
            mean_rel_distortion: damage,
            pick_flip_rate: 0.0,
        });
    }
    out.sort_by(|a, b| {
        b.damage
            .partial_cmp(&a.damage)
            .unwrap_or(Ordering::Equal)
            .then(a.channel.cmp(&b.channel))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// splitmix64-backed deterministic f32 stream in [−1, 1). No external
    /// RNG dep; same seed → same stream on every platform.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Bimodal K fixture whose damage is provable at the f32 level:
    /// d=4, seq=6, n_q=4. Channel 0 cycles {0, 1, −1}; channels 1..4 are
    /// the constant 0.5 (quantization-exact). Queries alternate q₀ ∈
    /// {+2, −2}. At 1 bit the 0-column snaps UP to +1 (round(0.5) = 1,
    /// all arithmetic Sterbenz-exact), so every q=+2 query's argmax flips
    /// to the first position and every q=−2 query keeps its argmax —
    /// damage = 0.5 exactly, channels 1..4 = 0.0 exactly.
    fn bimodal_fixture() -> (Vec<f32>, Vec<f32>, usize, usize, usize) {
        let (d, seq, n_q) = (4usize, 6usize, 4usize);
        let mut k = vec![0.5f32; seq * d];
        for j in 0..seq {
            k[j * d] = match j % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => -1.0,
            };
        }
        let mut q = vec![0.0f32; n_q * d];
        for i in 0..n_q {
            q[i * d] = if i % 2 == 0 { 2.0 } else { -2.0 };
        }
        (q, k, n_q, seq, d)
    }

    fn key_probe<'a>(
        q: &'a [f32],
        k_ref: &'a [f32],
        n_q: usize,
        seq: usize,
        d: usize,
        k_dequant: Option<&'a [f32]>,
    ) -> KeyTensors<'a> {
        KeyTensors {
            q,
            n_q,
            k_ref,
            k_dequant,
            seq,
            d,
        }
    }

    #[test]
    fn quantizer_basics() {
        let mut err = vec![0.0f32; 3];
        // Endpoints always land on-grid.
        uniform_quant_error_into(&[0.0, 1.0], 1, &mut err[..2]);
        assert_eq!(&err[..2], &[0.0, 0.0]);
        // Constant columns are exact at any bit width.
        uniform_quant_error_into(&[3.3, 3.3, 3.3], 1, &mut err);
        assert_eq!(err, vec![0.0; 3]);
        // Interior points carry the error: 0.4 snaps DOWN to 0 at 1 bit.
        uniform_quant_error_into(&[0.0, 0.4, 1.0], 1, &mut err);
        assert!(err[1] < -0.39 && err[1] > -0.41, "err[1] = {}", err[1]);
        assert_eq!(err[0], 0.0);
        assert_eq!(err[2], 0.0);
        // bits = 0 clamps to 1: same result as bits = 1.
        let mut e1 = vec![0.0f32; 3];
        uniform_quant_error_into(&[0.0, 0.6, 1.0], 0, &mut err);
        uniform_quant_error_into(&[0.0, 0.6, 1.0], 1, &mut e1);
        assert_eq!(err, e1);
        // bits = 100 clamps to 24: near-exact on a tiny interior point.
        uniform_quant_error_into(&[0.0, 1e-9, 1.0], 100, &mut err);
        assert!(err.iter().all(|e| e.abs() < 1e-6), "err = {err:?}");
    }

    #[test]
    fn determinism_repeat_identical() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let a = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        let b = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        assert_eq!(a, b);
        // The provable fixture verdict: channel 0 at exactly 0.5 damage.
        assert_eq!(a[0].channel, 0);
        assert_eq!(a[0].damage, 0.5);
        assert_eq!(a[0].pick_flip_rate, 0.5);
    }

    #[test]
    fn signal_channel_dominates_damage() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let res = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        assert_eq!(res.len(), d);
        assert_eq!(res[0].channel, 0, "signal channel must rank first");
        assert!(res[0].damage > 0.0);
        for r in &res[1..] {
            assert_eq!(r.damage, 0.0, "constant channels are quantization-exact");
        }
    }

    #[test]
    fn damage_monotone_in_bits() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let mut sc = ProbeScratch::default();
        let mut prev = f32::INFINITY;
        for bits in [1u32, 2, 4, 8] {
            let cfg = ProbeConfig { bits, top_k: 1 };
            let res = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
            let dmg = res[0].damage;
            assert!(
                dmg <= prev + 1e-6,
                "damage must be non-increasing in bits: {prev} -> {dmg}"
            );
            prev = dmg;
        }
        // And the coarse end provably fires on this fixture.
        assert_eq!(prev, 0.0);
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let res = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        assert_eq!(res[0].damage, 0.5);
    }

    // ── joint_key_read (Issue 960 T2 GOAT read) ─────────────────────────

    #[test]
    fn joint_read_zero_when_every_channel_is_quantization_exact() {
        // All-constant K is exact at ANY per-channel width → untouched
        // pattern → exactly zero damage.
        let (q, mut k, n_q, seq, d) = bimodal_fixture();
        for v in k.iter_mut() {
            *v = 0.5;
        }
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let read = joint_key_read(
            key_probe(&q, &k, n_q, seq, d, None),
            &[1, 1, 1, 1],
            &cfg,
            &mut sc,
        );
        assert_eq!(read.damage, 0.0);
        assert_eq!(read.mean_rel_distortion, 0.0);
        assert_eq!(read.pick_flip_rate, 0.0);
    }

    #[test]
    fn joint_read_bimodal_at_uniform_min_bits_matches_the_provable_flips() {
        // On the bimodal fixture with EVERY channel at 1 bit, only channel
        // 0 carries error (channels 1..4 are constant → exact), so the
        // joint read equals the single-channel marginal read: q=+2 queries
        // flip, q=−2 keep → damage exactly 0.5.
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let read = joint_key_read(
            key_probe(&q, &k, n_q, seq, d, None),
            &[1, 1, 1, 1],
            &cfg,
            &mut sc,
        );
        assert_eq!(read.damage, 0.5);
        assert_eq!(read.pick_flip_rate, 0.5);
    }

    #[test]
    fn joint_read_repeat_identical_and_shape_agnostic_to_bits_order() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let a = joint_key_read(
            key_probe(&q, &k, n_q, seq, d, None),
            &[2, 1, 3, 1],
            &cfg,
            &mut sc,
        );
        let b = joint_key_read(
            key_probe(&q, &k, n_q, seq, d, None),
            &[2, 1, 3, 1],
            &cfg,
            &mut sc,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn joint_read_monotone_in_bits() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let mut prev = f32::INFINITY;
        for bits in [1u8, 2, 4, 8] {
            let read = joint_key_read(
                key_probe(&q, &k, n_q, seq, d, None),
                &[bits, bits, bits, bits],
                &cfg,
                &mut sc,
            );
            assert!(
                read.damage <= prev + 1e-6,
                "joint damage must be non-increasing in bits: {prev} -> {}",
                read.damage
            );
            prev = read.damage;
        }
        assert_eq!(prev, 0.0, "fine grids on this fixture provably heal it");
    }

    #[test]
    fn joint_read_top_k_two_counts_set_overlap() {
        // k=2: the overlap denominator doubles; on the bimodal fixture the
        // flipped q=+2 queries lose ONE of their two targets, so damage
        // reads 0.25 (one lost slot of two, half the queries).
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 2 };
        let mut sc = ProbeScratch::default();
        let read = joint_key_read(
            key_probe(&q, &k, n_q, seq, d, None),
            &[1, 1, 1, 1],
            &cfg,
            &mut sc,
        );
        assert_eq!(read.damage, 0.25);
    }

    #[test]
    fn joint_read_refuses_zero_bits_and_attribution_mode() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let zero = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = joint_key_read(
                key_probe(&q, &k, n_q, seq, d, None),
                &[1, 0, 1, 1],
                &cfg,
                &mut sc,
            );
        }));
        assert!(zero.is_err(), "zero-width grid must refuse");
        let dq = vec![0.0f32; seq * d];
        let attr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = joint_key_read(
                key_probe(&q, &k, n_q, seq, d, Some(&dq)),
                &[1, 1, 1, 1],
                &cfg,
                &mut sc,
            );
        }));
        assert!(
            attr.is_err(),
            "attribution mode must refuse (reduction posture only)"
        );
    }

    #[test]
    fn joint_read_degenerate_shapes_read_zero() {
        // seq = 0, d = 0, n_q = 0 all read the zero verdict, never panic.
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let zero = joint_key_read(
            key_probe(&[], &[], 0, 0, 4, None),
            &[1, 1, 1, 1],
            &cfg,
            &mut sc,
        );
        assert_eq!(zero.damage, 0.0);
        let zero = joint_key_read(key_probe(&[], &[], 2, 0, 0, None), &[], &cfg, &mut sc);
        assert_eq!(zero.damage, 0.0);
        let k = vec![0.5f32; 6 * 4];
        let zero = joint_key_read(
            key_probe(&[], &k, 0, 6, 4, None),
            &[1, 1, 1, 1],
            &cfg,
            &mut sc,
        );
        assert_eq!(zero.damage, 0.0);
    }

    #[test]
    fn attribution_zero_error_reads_zero_damage() {
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let cfg = ProbeConfig::default();
        let mut sc = ProbeScratch::default();
        let res = probe_key_channels(key_probe(&q, &k, n_q, seq, d, Some(&k)), &cfg, &mut sc);
        assert!(res.iter().all(|r| r.damage == 0.0));
    }

    #[test]
    fn attribution_matches_reduction_on_known_grid() {
        // Hand-build the 1-bit quantized channel-0 column for the bimodal
        // fixture ({0 → 1, 1 → 1, −1 → −1}; channels 1..4 exact) and check
        // the attribution path reads the same 0.5 damage — both error
        // columns are Sterbenz-exact, so the match is byte-identical.
        let (q, k, n_q, seq, d) = bimodal_fixture();
        let mut dq = k.clone();
        for j in 0..seq {
            if j % 3 == 0 {
                dq[j * d] = 1.0;
            }
        }
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let attr = probe_key_channels(key_probe(&q, &k, n_q, seq, d, Some(&dq)), &cfg, &mut sc);
        let red = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        assert_eq!(attr, red);
        assert_eq!(attr[0].damage, 0.5);
    }

    #[test]
    fn pick_flip_detected_on_tie_push() {
        // d=1, seq=3: reference argmax is j=2 (the column max, which never
        // moves under min–max quantization); the interior 9.6 snaps UP to
        // exactly 10.0 (scale = 10/1 exact, all arithmetic Sterbenz-exact),
        // tying it — the first-index tie rule moves the pick to j=0.
        let q = [1.0f32];
        let k = [9.6f32, 0.0, 10.0];
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let res = probe_key_channels(key_probe(&q, &k, 1, 3, 1, None), &cfg, &mut sc);
        assert_eq!(res[0].pick_flip_rate, 1.0);
        assert_eq!(res[0].damage, 1.0);
    }

    #[test]
    fn topk_overlap_half_on_middle_push() {
        // d=1, seq=3, top_k=2: reference top-2 = {0, 2} (scores 1.0, 0.5,
        // 0.52); the interior 0.52 snaps DOWN to exactly 0.5 (scale = 0.5
        // exact), so the perturbed top-2 = {0, 1} — overlap 1/2.
        let q = [1.0f32];
        let k = [1.0f32, 0.5, 0.52];
        let cfg = ProbeConfig { bits: 1, top_k: 2 };
        let mut sc = ProbeScratch::default();
        let res = probe_key_channels(key_probe(&q, &k, 1, 3, 1, None), &cfg, &mut sc);
        assert_eq!(res[0].damage, 0.5);
        // The k=1 read is untouched: the argmax (1.0) never moves.
        assert_eq!(res[0].pick_flip_rate, 0.0);
    }

    /// The V-side fixture: d=2, seq=4, channel 0 =
    /// [1.0, 0.7, 0.85, 0.7] (interior 0.85 snaps UP to 1.0 at 1 bit),
    /// channel 1 constant 0.5 (quantization-exact).
    fn value_fixture() -> [f32; 8] {
        [
            1.0f32, 0.5, 0.7, 0.5, 0.85, 0.5, 0.7, 0.5, // [seq × d] row-major
        ]
    }

    #[test]
    fn value_side_signal_channel_dominates() {
        let v = value_fixture();
        let mut sc = ProbeScratch::default();
        let res = probe_value_channels(
            ValueTensors {
                v_ref: &v,
                v_dequant: None,
                n_q: 1,
                seq: 4,
                d: 2,
                weights: None,
                bits: 1,
            },
            &mut sc,
        );
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].channel, 0);
        assert!(res[0].damage > 0.0);
        assert_eq!(res[1].damage, 0.0);
        assert!(res.iter().all(|r| r.pick_flip_rate == 0.0));
    }

    #[test]
    fn value_side_respects_attention_weights() {
        // Same column; weights concentrated on an error-carrying position
        // register damage, weights on an error-free position read zero.
        // Position 1 is the column MIN (always exactly on-grid, err = 0);
        // position 2 is the interior point (snaps up, err > 0).
        let v = value_fixture();
        let clean = [0.0f32, 1.0, 0.0, 0.0]; // position 1: the min, err 0
        let noisy = [0.0f32, 0.0, 1.0, 0.0]; // position 2: interior, err +0.15
        let mut sc = ProbeScratch::default();
        let a = probe_value_channels(
            ValueTensors {
                v_ref: &v,
                v_dequant: None,
                n_q: 1,
                seq: 4,
                d: 2,
                weights: Some(&clean),
                bits: 1,
            },
            &mut sc,
        );
        let b = probe_value_channels(
            ValueTensors {
                v_ref: &v,
                v_dequant: None,
                n_q: 1,
                seq: 4,
                d: 2,
                weights: Some(&noisy),
                bits: 1,
            },
            &mut sc,
        );
        assert_eq!(a[0].damage, 0.0);
        assert!(b[0].damage > 0.0);
    }

    #[test]
    fn value_attribution_matches_reduction_on_known_grid() {
        // Hand-build the quantized column for the V fixture: 0.85 snaps to
        // 1.0, everything else exact — both error columns identical.
        let v = value_fixture();
        let mut dq = v;
        dq[4] = 1.0; // position 2, channel 0
        let mut sc = ProbeScratch::default();
        let attr = probe_value_channels(
            ValueTensors {
                v_ref: &v,
                v_dequant: Some(&dq),
                n_q: 1,
                seq: 4,
                d: 2,
                weights: None,
                bits: 0,
            },
            &mut sc,
        );
        let red = probe_value_channels(
            ValueTensors {
                v_ref: &v,
                v_dequant: None,
                n_q: 1,
                seq: 4,
                d: 2,
                weights: None,
                bits: 1,
            },
            &mut sc,
        );
        assert_eq!(attr, red);
        assert!(attr[0].damage > 0.0);
    }

    #[test]
    fn degenerate_shapes() {
        let cfg = ProbeConfig::default();
        let mut sc = ProbeScratch::default();
        // seq = 0: nothing measurable (shapes still declared).
        assert!(
            probe_key_channels(key_probe(&[0.0f32; 8], &[], 2, 0, 4, None), &cfg, &mut sc)
                .is_empty()
        );
        // d = 0: no channels.
        assert!(probe_key_channels(key_probe(&[], &[], 2, 4, 0, None), &cfg, &mut sc).is_empty());
        // n_q = 0: zero-evidence rows for every channel.
        let k = vec![0.5f32; 4 * 3];
        let res = probe_key_channels(key_probe(&[], &k, 0, 4, 3, None), &cfg, &mut sc);
        assert_eq!(res.len(), 3);
        assert!(res.iter().all(|r| r.damage == 0.0));
        // seq = 1: constant single-position column is exact.
        let res = probe_key_channels(key_probe(&[1.0f32], &[2.0], 1, 1, 1, None), &cfg, &mut sc);
        assert_eq!(res[0].damage, 0.0);
        // V side mirrors the same guards.
        assert!(
            probe_value_channels(
                ValueTensors {
                    v_ref: &[],
                    v_dequant: None,
                    n_q: 2,
                    seq: 0,
                    d: 4,
                    weights: None,
                    bits: 1,
                },
                &mut sc
            )
            .is_empty()
        );
        assert!(
            probe_value_channels(
                ValueTensors {
                    v_ref: &[],
                    v_dequant: None,
                    n_q: 2,
                    seq: 4,
                    d: 0,
                    weights: None,
                    bits: 1,
                },
                &mut sc
            )
            .is_empty()
        );
        let v = vec![0.5f32; 4 * 3];
        let res = probe_value_channels(
            ValueTensors {
                v_ref: &v,
                v_dequant: None,
                n_q: 0,
                seq: 4,
                d: 3,
                weights: None,
                bits: 1,
            },
            &mut sc,
        );
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn ties_sort_channel_ascending() {
        // All-constant cache: every channel reads exactly 0 damage, so the
        // output order is decided entirely by the channel-ascending
        // tie-break.
        let (d, seq, n_q) = (5usize, 8usize, 3usize);
        let k = vec![0.25f32; seq * d];
        let q = vec![0.5f32; n_q * d];
        let cfg = ProbeConfig { bits: 1, top_k: 1 };
        let mut sc = ProbeScratch::default();
        let res = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        for (i, r) in res.iter().enumerate() {
            assert_eq!(r.channel, i, "ties must sort channel-ascending");
        }
    }

    #[test]
    fn random_fixture_attribution_zero_when_dq_is_ref() {
        // A seeded random-shaped fixture (not just the hand-built ones):
        // attribution against the untouched reference must read exactly
        // zero damage everywhere, at a non-trivial shape.
        let (d, seq, n_q) = (8usize, 24usize, 5usize);
        let mut rng = Lcg::new(0x960_013);
        let mut k = vec![0.0f32; seq * d];
        for x in k.iter_mut() {
            *x = rng.next_f32();
        }
        let mut q = vec![0.0f32; n_q * d];
        for x in q.iter_mut() {
            *x = rng.next_f32();
        }
        let cfg = ProbeConfig { bits: 3, top_k: 4 };
        let mut sc = ProbeScratch::default();
        let res = probe_key_channels(key_probe(&q, &k, n_q, seq, d, Some(&k)), &cfg, &mut sc);
        assert_eq!(res.len(), d);
        assert!(res.iter().all(|r| r.damage == 0.0));
        // And the reduction mode at 3 bits moves the scores on real
        // spread: the GUARANTEED nonzero signal is the distortion
        // companion (nonzero errors → nonzero ‖Δrow‖); damage itself is a
        // rank-change count and may legitimately read zero at 3 bits.
        let red = probe_key_channels(key_probe(&q, &k, n_q, seq, d, None), &cfg, &mut sc);
        assert!(red.iter().any(|r| r.mean_rel_distortion > 0.0));
    }
}
