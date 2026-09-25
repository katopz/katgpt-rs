//! Per-input-channel activation moments per linear layer — the
//! activation-diagonal collector (Issue 886 P0; Research 588 §2.1, the
//! AWQ `s_X` / llama.cpp imatrix `⟨x_j²⟩` artifact). Feature
//! `act_channel_moments`.
//!
//! One offline calibration pass observes the INPUT vector of every tapped
//! linear layer and accumulates, per input channel `j`,
//! `{Σ|x_j|, Σx_j²}` plus the observation count. [`ActChannelMoments::freeze`]
//! turns the sums into the committed product, an [`ActChannelDiagonal`]:
//! `mean|x_j|` (AWQ's saliency statistic) and `E[x_j²]` (the imatrix /
//! diagonal-of-GPTQ-Hessian weight) per channel per layer. The consumer is
//! `katgpt-types::TernaryGroupWeights::quantize_from_f32_act_aware`
//! (feature `act_aware_fit`), which takes one layer's diagonal as a plain
//! `&[f32]` — the dependency points from here to there, never back.
//!
//! # Laws (shared with the Issue 883 `fitted_anchor_table` family)
//!
//! - **Streaming sufficient statistics, f64**: a calibration corpus is
//!   10⁶–10⁹ token-layer observations; f32 running sums lose the tail. The
//!   observe loop is `|x|` → f64 → two adds + one mul per element.
//! - **Alloc-free observe (G4)**: every buffer is allocated at construction;
//!   `observe` / `observe_batch` touch only pre-sized slices. `freeze`,
//!   `to_bytes` and `from_bytes` allocate — they run once per calibration.
//! - **Poison control**: a non-finite observation panics (a tap-point bug,
//!   never data — the same refusal as `StreamingMeanTable`).
//! - **Layer widths are per layer**: `q/k/v/gate/up` see `d_model`, `o`
//!   sees `n_heads·head_dim`, `down` sees `d_ff` — one table holds them all.
//!
//! # Commitment (the `StaticCalTable` family, deterministic spelling)
//!
//! [`ActChannelDiagonal`] carries a BLAKE3 commitment over a **canonical
//! little-endian byte image** (magic, version, per-layer `(width, count)`,
//! then every `mean|x|` f32, then every `E[x²]` f32 — explicit
//! `to_le_bytes`, so the digest is platform-independent, unlike
//! `StaticCalTable`'s native-endian hash). `to_bytes` = image ‖ digest;
//! `from_bytes` re-derives the digest and refuses a mismatch. Binding the
//! table to the WEIGHTS it was measured against is the caller's
//! `calibration_staleness::SnapshotBound<ActChannelDiagonal>` — the
//! commitment names the table, not the checkpoint.
//!
//! # Co-collection with the Issue 883 pass
//!
//! The 883 pass taps K/V **outputs** of the projections; this collector
//! wants linear-layer **inputs** (the normed hidden state feeding q/k/v,
//! the attention output feeding `o_proj`, the SwiGLU product feeding
//! `down_proj`). Same forward, different tap points: call
//! [`ActChannelMoments::observe`] beside `LayeredVkCalibration::observe_layer`
//! in the same tapped loop. 883's grand `Σx²` is over K/V and is NOT a
//! substitute for this diagonal.

use std::fmt;

/// Canonical-image magic (`KACD` = Katgpt Activation Channel Diagonal).
pub const ACT_DIAGONAL_MAGIC: [u8; 4] = *b"KACD";
/// Canonical-image format version.
pub const ACT_DIAGONAL_VERSION: u32 = 1;

/// Streaming per-input-channel `{Σ|x|, Σx²}` accumulator over a set of
/// linear layers (possibly of different input widths).
pub struct ActChannelMoments {
    /// `offsets[l]..offsets[l+1]` is layer `l`'s channel range.
    offsets: Vec<usize>,
    counts: Vec<u64>,
    sum_abs: Vec<f64>,
    sum_sq: Vec<f64>,
}

impl ActChannelMoments {
    /// Allocate an accumulator for `widths.len()` linear layers, layer `l`
    /// having `widths[l]` input channels. The only allocation site.
    #[must_use]
    pub fn new(widths: &[usize]) -> Self {
        let offsets = prefix_offsets(widths);
        let total = *offsets.last().unwrap_or(&0);
        Self {
            offsets,
            counts: vec![0; widths.len()],
            sum_abs: vec![0.0; total],
            sum_sq: vec![0.0; total],
        }
    }

    /// Number of tracked linear layers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.counts.len()
    }

    /// Input width of layer `layer`.
    #[must_use]
    pub fn width(&self, layer: usize) -> usize {
        self.offsets[layer + 1] - self.offsets[layer]
    }

    /// Observations (activation vectors) seen by layer `layer`.
    #[must_use]
    pub fn count(&self, layer: usize) -> u64 {
        self.counts[layer]
    }

    /// Observe one input vector `x` of layer `layer` (length = its width).
    /// Alloc-free. Panics on a width mismatch or a non-finite element.
    #[inline]
    pub fn observe(&mut self, layer: usize, x: &[f32]) {
        let (lo, hi) = (self.offsets[layer], self.offsets[layer + 1]);
        assert_eq!(x.len(), hi - lo, "act_channel_moments: width mismatch at layer {layer}");
        accumulate(&mut self.sum_abs[lo..hi], &mut self.sum_sq[lo..hi], x);
        self.counts[layer] += 1;
    }

    /// Observe a row-major batch `xs = [n × width]` of layer `layer` in one
    /// call (a prefill chunk). Alloc-free; same laws as [`Self::observe`].
    pub fn observe_batch(&mut self, layer: usize, xs: &[f32]) {
        let (lo, hi) = (self.offsets[layer], self.offsets[layer + 1]);
        let w = hi - lo;
        if w == 0 {
            assert!(xs.is_empty(), "act_channel_moments: zero-width layer {layer} got data");
            return;
        }
        assert_eq!(xs.len() % w, 0, "act_channel_moments: batch not a multiple of width {w}");
        let (sa, ss) = (&mut self.sum_abs[lo..hi], &mut self.sum_sq[lo..hi]);
        for x in xs.chunks_exact(w) {
            accumulate(sa, ss, x);
        }
        self.counts[layer] += (xs.len() / w) as u64;
    }

    /// Freeze into the committed diagonal table (allocates; offline). A
    /// layer with no observations freezes to all-zero moments, which the
    /// `act_aware_fit` consumer reads as "no information ⇒ baseline fit".
    #[must_use]
    pub fn freeze(&self) -> ActChannelDiagonal {
        let total = self.sum_abs.len();
        let mut mean_abs = vec![0.0f32; total];
        let mut mean_sq = vec![0.0f32; total];
        for l in 0..self.layers() {
            let n = self.counts[l];
            if n == 0 {
                continue;
            }
            let inv = 1.0 / n as f64;
            for j in self.offsets[l]..self.offsets[l + 1] {
                mean_abs[j] = (self.sum_abs[j] * inv) as f32;
                mean_sq[j] = (self.sum_sq[j] * inv) as f32;
            }
        }
        ActChannelDiagonal::from_parts(self.offsets.clone(), self.counts.clone(), mean_abs, mean_sq)
    }
}

/// The hot loop: `Σ|x| += |x|`, `Σx² += x²` in f64, then the poison check.
#[inline]
fn accumulate(sa: &mut [f64], ss: &mut [f64], x: &[f32]) {
    let mut nonfinite = false;
    for ((a, s), &v) in sa.iter_mut().zip(ss.iter_mut()).zip(x) {
        let v64 = f64::from(v.abs());
        *a += v64;
        *s += v64 * v64;
        nonfinite |= !v.is_finite();
    }
    if nonfinite {
        panic!("act_channel_moments: non-finite activation (tap-point bug, not data)");
    }
}

fn prefix_offsets(widths: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(widths.len() + 1);
    let mut acc = 0usize;
    offsets.push(0);
    for &w in widths {
        acc += w;
        offsets.push(acc);
    }
    offsets
}

/// The frozen, committed activation diagonal: per layer, per input channel,
/// `mean|x_j|` and `E[x_j²]`, plus the per-layer observation count and a
/// BLAKE3 commitment over the canonical image.
#[derive(Clone, Debug, PartialEq)]
pub struct ActChannelDiagonal {
    offsets: Vec<usize>,
    counts: Vec<u64>,
    mean_abs: Vec<f32>,
    mean_sq: Vec<f32>,
    commitment: [u8; 32],
}

/// Why [`ActChannelDiagonal::from_bytes`] refused an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActDiagonalDecodeError {
    /// Shorter than its own header/length fields claim.
    Truncated,
    /// Magic bytes are not `KACD`.
    BadMagic,
    /// A version this build does not read.
    UnsupportedVersion(u32),
    /// Trailing bytes after the digest.
    TrailingBytes,
    /// A moment is negative or non-finite.
    InvalidMoment,
    /// The trailing BLAKE3 digest does not match the image.
    CommitmentMismatch,
}

impl fmt::Display for ActDiagonalDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "act channel diagonal decode: {self:?}")
    }
}

impl std::error::Error for ActDiagonalDecodeError {}

impl ActChannelDiagonal {
    fn from_parts(
        offsets: Vec<usize>,
        counts: Vec<u64>,
        mean_abs: Vec<f32>,
        mean_sq: Vec<f32>,
    ) -> Self {
        let mut t = Self { offsets, counts, mean_abs, mean_sq, commitment: [0; 32] };
        t.commitment = *blake3::hash(&t.canonical_image()).as_bytes();
        t
    }

    /// A table whose every channel has the same moments — the G3 control
    /// (a uniform diagonal must reproduce the activation-blind baseline).
    #[must_use]
    pub fn uniform(widths: &[usize], mean_abs: f32, mean_sq: f32, count: u64) -> Self {
        let offsets = prefix_offsets(widths);
        let total = *offsets.last().unwrap_or(&0);
        Self::from_parts(
            offsets,
            vec![count; widths.len()],
            vec![mean_abs; total],
            vec![mean_sq; total],
        )
    }

    /// Number of layers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.counts.len()
    }

    /// Input width of layer `layer`.
    #[must_use]
    pub fn width(&self, layer: usize) -> usize {
        self.offsets[layer + 1] - self.offsets[layer]
    }

    /// Observations behind layer `layer`'s moments.
    #[must_use]
    pub fn count(&self, layer: usize) -> u64 {
        self.counts[layer]
    }

    /// `mean|x_j|` for layer `layer` — AWQ's `s_X`.
    #[must_use]
    pub fn mean_abs(&self, layer: usize) -> &[f32] {
        &self.mean_abs[self.offsets[layer]..self.offsets[layer + 1]]
    }

    /// `E[x_j²]` for layer `layer` — the imatrix weight (the diagonal of the
    /// input second-moment matrix).
    #[must_use]
    pub fn mean_sq(&self, layer: usize) -> &[f32] {
        &self.mean_sq[self.offsets[layer]..self.offsets[layer + 1]]
    }

    /// BLAKE3 commitment over the canonical image.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Re-derive the commitment and compare (in-memory corruption check).
    #[must_use]
    pub fn verify(&self) -> bool {
        *blake3::hash(&self.canonical_image()).as_bytes() == self.commitment
    }

    /// The canonical little-endian image the commitment is taken over.
    fn canonical_image(&self) -> Vec<u8> {
        let layers = self.layers();
        let total = self.mean_abs.len();
        let mut b = Vec::with_capacity(12 + layers * 12 + total * 8);
        b.extend_from_slice(&ACT_DIAGONAL_MAGIC);
        b.extend_from_slice(&ACT_DIAGONAL_VERSION.to_le_bytes());
        b.extend_from_slice(&(layers as u32).to_le_bytes());
        for l in 0..layers {
            b.extend_from_slice(&(self.width(l) as u32).to_le_bytes());
            b.extend_from_slice(&self.counts[l].to_le_bytes());
        }
        for v in &self.mean_abs {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.mean_sq {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// Serialize: canonical image ‖ 32-byte BLAKE3 digest. Deterministic —
    /// the same moments produce the same bytes on every platform.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = self.canonical_image();
        b.extend_from_slice(&self.commitment);
        b
    }

    /// Decode and verify an image produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ActDiagonalDecodeError> {
        let mut r = Reader { b: bytes, pos: 0 };
        if r.take(4)? != ACT_DIAGONAL_MAGIC {
            return Err(ActDiagonalDecodeError::BadMagic);
        }
        let version = r.u32()?;
        if version != ACT_DIAGONAL_VERSION {
            return Err(ActDiagonalDecodeError::UnsupportedVersion(version));
        }
        let layers = r.u32()? as usize;
        // Bound the header by what the image can actually hold (12 bytes per
        // layer) before allocating from an untrusted length field.
        if layers > bytes.len() / 12 {
            return Err(ActDiagonalDecodeError::Truncated);
        }
        let mut widths = Vec::with_capacity(layers);
        let mut counts = Vec::with_capacity(layers);
        for _ in 0..layers {
            widths.push(r.u32()? as usize);
            counts.push(r.u64()?);
        }
        let offsets = prefix_offsets(&widths);
        let total = *offsets.last().unwrap_or(&0);
        if total > bytes.len() / 8 {
            return Err(ActDiagonalDecodeError::Truncated);
        }
        let mut mean_abs = Vec::with_capacity(total);
        for _ in 0..total {
            mean_abs.push(r.f32()?);
        }
        let mut mean_sq = Vec::with_capacity(total);
        for _ in 0..total {
            mean_sq.push(r.f32()?);
        }
        let image_len = r.pos;
        let digest = r.take(32)?;
        if r.pos != bytes.len() {
            return Err(ActDiagonalDecodeError::TrailingBytes);
        }
        if mean_abs.iter().chain(&mean_sq).any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(ActDiagonalDecodeError::InvalidMoment);
        }
        let commitment = *blake3::hash(&bytes[..image_len]).as_bytes();
        if commitment[..] != digest[..] {
            return Err(ActDiagonalDecodeError::CommitmentMismatch);
        }
        Ok(Self { offsets, counts, mean_abs, mean_sq, commitment })
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ActDiagonalDecodeError> {
        let end = self.pos.checked_add(n).ok_or(ActDiagonalDecodeError::Truncated)?;
        let s = self.b.get(self.pos..end).ok_or(ActDiagonalDecodeError::Truncated)?;
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ActDiagonalDecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64, ActDiagonalDecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    fn f32(&mut self) -> Result<f32, ActDiagonalDecodeError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known answer: a synthetic diagonal with planted per-channel scales.
    /// Channel j sees the fixed cycle `c_j · {+1, −2, +3, −4}` ⇒
    /// `mean|x| = 2.5·c_j`, `E[x²] = 7.5·c_j²` exactly.
    #[test]
    fn synthetic_diagonal_known_answer() {
        let widths = [6usize, 3];
        let mut m = ActChannelMoments::new(&widths);
        let cycle = [1.0f32, -2.0, 3.0, -4.0];
        for rep in 0..25 {
            for &c in &cycle {
                let x0: Vec<f32> = (0..6).map(|j| c * (j as f32 + 1.0)).collect();
                m.observe(0, &x0);
                let x1: Vec<f32> = (0..3).map(|j| c * 10.0f32.powi(j)).collect();
                // Mix the two entry points: half batch, half single.
                if rep % 2 == 0 {
                    m.observe(1, &x1);
                } else {
                    m.observe_batch(1, &x1);
                }
            }
        }
        let d = m.freeze();
        assert_eq!(d.count(0), 100);
        assert_eq!(d.count(1), 100);
        for j in 0..6 {
            let c = j as f32 + 1.0;
            assert!((d.mean_abs(0)[j] - 2.5 * c).abs() < 1e-5 * c);
            assert!((d.mean_sq(0)[j] - 7.5 * c * c).abs() < 1e-5 * c * c);
        }
        for j in 0..3 {
            let c = 10.0f32.powi(j as i32);
            assert!((d.mean_abs(1)[j] - 2.5 * c).abs() <= 1e-5 * c);
            assert!((d.mean_sq(1)[j] - 7.5 * c * c).abs() <= 1e-4 * c * c);
        }
    }

    #[test]
    fn batch_equals_singles() {
        let w = 5;
        let xs: Vec<f32> = (0..w * 7).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut a = ActChannelMoments::new(&[w]);
        let mut b = ActChannelMoments::new(&[w]);
        a.observe_batch(0, &xs);
        for x in xs.chunks_exact(w) {
            b.observe(0, x);
        }
        assert_eq!(a.freeze(), b.freeze());
    }

    #[test]
    fn empty_layer_freezes_to_zero() {
        let m = ActChannelMoments::new(&[4, 2]);
        let d = m.freeze();
        assert!(d.mean_abs(0).iter().all(|&v| v == 0.0));
        assert_eq!(d.count(1), 0);
        assert!(d.verify());
    }

    #[test]
    fn roundtrip_and_commitment() {
        let mut m = ActChannelMoments::new(&[3, 4]);
        m.observe(0, &[1.0, -2.0, 0.5]);
        m.observe(1, &[0.1, 0.2, -0.3, 4.0]);
        let d = m.freeze();
        let bytes = d.to_bytes();
        let back = ActChannelDiagonal::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(back, d);
        assert_eq!(back.to_bytes(), bytes, "deterministic re-serialization");
        // Every single-byte flip is refused (header, moments, digest).
        for i in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[i] ^= 0x01;
            assert!(ActChannelDiagonal::from_bytes(&bad).is_err(), "flip at {i} accepted");
        }
        assert_eq!(
            ActChannelDiagonal::from_bytes(&bytes[..bytes.len() - 1]),
            Err(ActDiagonalDecodeError::Truncated)
        );
        let mut long = bytes.clone();
        long.push(0);
        assert_eq!(ActChannelDiagonal::from_bytes(&long), Err(ActDiagonalDecodeError::TrailingBytes));
    }

    /// Pinned digest of a fixed fixture — the canonical image is a format;
    /// a change to it must be a deliberate version bump.
    #[test]
    fn canonical_digest_pinned() {
        let d = ActChannelDiagonal::uniform(&[2, 3], 0.5, 0.25, 7);
        let hex: String = d.commitment().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, PINNED_DIGEST, "{hex}");
    }
    const PINNED_DIGEST: &str = "242a154410cfb9edfd3c0ec9d93ecf63cd7f869fe2b3c5b7f3f1666306644957";

    #[test]
    #[should_panic(expected = "non-finite")]
    fn nonfinite_panics() {
        let mut m = ActChannelMoments::new(&[2]);
        m.observe(0, &[1.0, f32::NAN]);
    }

    #[test]
    #[should_panic(expected = "width mismatch")]
    fn width_mismatch_panics() {
        let mut m = ActChannelMoments::new(&[3]);
        m.observe(0, &[1.0, 2.0]);
    }
}
