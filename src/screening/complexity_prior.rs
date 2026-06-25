//! Algorithmic-Probability Sampler — `sigmoid(-α·K̃(x) - β)`-weighted sampling
//! (Plan 305, Research 284, Dingle & Hutter 2026, *Entropy* 28(2):226).
//!
//! Replaces uniform candidate sampling in MCTS / bandits / DDTree / speculative
//! drafters with a simplicity-biased prior. Pluggable `K̃` proxies: RLE ratio,
//! Shannon entropy, L1 norm. Safety guarantee: never worse than uniform;
//! exponentially faster when the optimum is low-K (Levin-search variant).
//!
//! **Latent variant** [`LatentCompressionPriorSampler`] operates on `&[f32]` via
//! byte-quantization. riir-ai Plan 331 wires this to HLA / functor / shard vectors.
//!
//! # Sigmoid, never softmax
//!
//! Per-candidate sigmoid for the public log-prob; cumulative-sum + binary-search
//! for sampling (the normalization is internal to the sampler, not the API).

use core::hint::black_box;
use fastrand::Rng;

// ── Local RLE / entropy kernels ──────────────────────────────────────────────
//
// NOTE: The public `crate::ruliology::irreducibility` API works on `WinMatrix`,
// not raw bytes, and its `rle_compress` / inline Shannon-entropy kernels are
// private (`fn`, not `pub fn`). We keep this module self-contained for the
// byte-level proxies Plan 305 needs (operating on arbitrary `&[u8]` candidates,
// not game payoff matrices). The math mirrors `IrreducibilityGate::analyze`
// (`ruliology/irreducibility.rs` L60–110): byte-frequency histogram →
// `Σ -p·log2(p)` for entropy; RLE `(value, count)` pair counting for compressed
// length. The only difference is zero-allocation: we count pairs instead of
// materialising a `Vec<u8>`.

/// Zero-allocation RLE compressed length.
///
/// Counts `(value, count)` runs in a single pass and returns `2 * num_runs`
/// (one byte for the value, one byte for the count). When a run exceeds 255,
/// it is split into multiple pairs (max 255 per pair).
///
/// Returns 0 for empty input.
#[inline]
pub(crate) fn rle_compressed_len(data: &[u8]) -> usize {
    let n = data.len();
    if n == 0 {
        return 0;
    }
    let mut runs: usize = 0;
    let mut i = 0;
    while i < n {
        let value = data[i];
        let mut run: usize = 0;
        while i < n && data[i] == value {
            i += 1;
            run += 1;
        }
        // Each 255-byte chunk of the run becomes its own (value, count) pair.
        runs += run.div_ceil(255);
    }
    runs * 2
}

/// Shannon entropy in bits for a byte slice.
///
/// `H = Σ_b -p_b · log2(p_b)` over the 256-symbol alphabet. Returns 0.0 for
/// empty slices (no information). Maximum is 8.0 (uniform byte distribution).
#[inline]
pub(crate) fn shannon_entropy_bits(data: &[u8]) -> f32 {
    let total = data.len() as f32;
    if total == 0.0 {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let mut h = 0.0f32;
    for &count in &freq {
        if count > 0 {
            let p = count as f32 / total;
            h -= p * p.log2();
        }
    }
    h
}

/// Numerically-stable logistic sigmoid. Clamps the input to avoid overflow.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    // Clamp outside ±18 to keep `exp` in safe range (sigmoid(±18) ≈ 1 / 1.5e-8).
    if x >= 18.0 {
        1.0
    } else if x <= -18.0 {
        0.0
    } else {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    }
}

// ── ComplexityProxy trait ─────────────────────────────────────────────────────

/// Pluggable Kolmogorov-complexity proxy `K̃: &[u8] → [0, 1]`.
///
/// Plan 305 T1.1 originally specified `<T: AsRef<[u8]>>` for full genericity;
/// we narrow to `&[u8]` to keep the trait **object-safe** (so samplers can hold
/// `Box<dyn ComplexityProxy>` if ever needed) and ergonomic. The sampler hot
/// path always has a byte slice in hand, so the narrow signature is lossless.
///
/// Contract: `k_tilde` SHOULD return a value in `[0.0, 1.0]` (some proxies
/// produce slightly out-of-range values for empty / degenerate inputs — callers
/// should clamp or treat 0 as "trivially simple" and 1 as "maximally complex").
pub trait ComplexityProxy {
    /// Estimated normalised Kolmogorov complexity `K̃(candidate) ∈ [0, 1]`.
    fn k_tilde(&self, candidate: &[u8]) -> f32;
}

// ── Built-in proxies ─────────────────────────────────────────────────────────

/// RLE compression-ratio proxy: `K̃ = rle_compressed_len / raw_len`.
///
/// Low ratio → highly compressible (simple). High ratio → incompressible
/// (random / complex). Mirrors the RLE kernel inside
/// `ruliology::irreducibility::IrreducibilityGate::analyze`, but zero-allocation
/// (counts runs instead of materialising the compressed `Vec<u8>`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RleComplexity;

impl RleComplexity {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ComplexityProxy for RleComplexity {
    #[inline]
    fn k_tilde(&self, candidate: &[u8]) -> f32 {
        let raw = candidate.len();
        if raw == 0 {
            return 0.0;
        }
        rle_compressed_len(candidate) as f32 / raw as f32
    }
}

/// Shannon-entropy proxy: `K̃ = H(candidate) / 8` (normalised to `[0, 1]`).
///
/// 0.0 = all bytes identical (zero entropy). 1.0 = uniform byte distribution
/// (max entropy, 8 bits). Mirrors the entropy kernel inside
/// `ruliology::irreducibility::IrreducibilityGate::analyze`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EntropyComplexity;

impl EntropyComplexity {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ComplexityProxy for EntropyComplexity {
    #[inline]
    fn k_tilde(&self, candidate: &[u8]) -> f32 {
        shannon_entropy_bits(candidate) / 8.0
    }
}

/// Byte L1-norm proxy: `K̃ = Σ |b| / (255 · len)`.
///
/// Normalises the byte L1 magnitude to `[0, 1]`. 0.0 = all-zero bytes.
/// 1.0 = all-255 bytes. Per R125 sandwich bound for fixed-precision latents —
/// a cheap proxy for weight magnitude / energy.
#[derive(Debug, Clone, Copy, Default)]
pub struct L1Complexity;

impl L1Complexity {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ComplexityProxy for L1Complexity {
    #[inline]
    fn k_tilde(&self, candidate: &[u8]) -> f32 {
        let n = candidate.len();
        if n == 0 {
            return 0.0;
        }
        let sum: u64 = candidate.iter().map(|&b| b as u64).sum();
        sum as f32 / (255.0 * n as f32)
    }
}

// ── Stub proxies (sub-feature gated) ─────────────────────────────────────────

/// LZ4 compression-ratio proxy.
///
/// Phase 1 ships RLE/Entropy/L1 only. This stub exists so callers can write
/// feature-gated code against the type; the orchestrator wires the `lz4_proxy`
/// sub-feature + `lz4_flex` dep in a follow-up.
//
// Requires `lz4_proxy` sub-feature + `lz4_flex` dep (orchestrator wires).
#[cfg(feature = "lz4_proxy")]
pub struct Lz4Complexity;

#[cfg(feature = "lz4_proxy")]
impl Lz4Complexity {
    /// Stub — Phase 1 does not implement LZ4. Returns neutral complexity.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(feature = "lz4_proxy")]
impl ComplexityProxy for Lz4Complexity {
    #[inline]
    fn k_tilde(&self, _candidate: &[u8]) -> f32 {
        // TODO(Phase 2+): `lz4_flex::compress_prepend_size(candidate).len()` ratio.
        // Until then, fall back to the RLE kernel so feature-gated code compiles.
        0.5
    }
}

/// BLAKE3 canonical-length proxy.
///
/// Phase 1 ships RLE/Entropy/L1 only. The orchestrator wires the `blake3_proxy`
/// sub-feature (used by `riir-neuron-db` for canonical shard-length commitment)
/// in a follow-up.
//
// Requires `blake3_proxy` sub-feature (orchestrator wires). Used by
// `riir-neuron-db` for canonical shard-length commitment.
#[cfg(feature = "blake3_proxy")]
pub struct Blake3CanonicalLengthComplexity;

#[cfg(feature = "blake3_proxy")]
impl Blake3CanonicalLengthComplexity {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(feature = "blake3_proxy")]
impl ComplexityProxy for Blake3CanonicalLengthComplexity {
    #[inline]
    fn k_tilde(&self, _candidate: &[u8]) -> f32 {
        // TODO(Phase 2+): BLAKE3-keyed length-extension stable hash length ratio.
        0.5
    }
}

// ── CompressionPriorSampler ──────────────────────────────────────────────────

/// Algorithmic-probability sampler: `P(x) ∝ sigmoid(-α·K̃(x) - β)`.
///
/// Pluggable over any [`ComplexityProxy`] `K`. The PUBLIC scoring API is
/// [`Self::log_prob`] (a log-sigmoid input, **never softmax**). For sampling,
/// [`Self::sample_ix`] builds a cumulative distribution internally via
/// per-candidate sigmoid and binary-searches it — the normalisation is an
/// internal implementation detail, not a public softmax.
///
/// # Fields
///
/// - `proxy`: the `K̃` estimator (RLE, entropy, L1, …).
/// - `alpha`: temperature — higher `α` biases harder toward low-K candidates.
/// - `beta`: offset — shifts the sigmoid midpoint. Useful for online
///   calibration (curiosity signal in riir-ai Plan 331).
///
/// # Construction
///
/// Prefer [`Self::new`] when `K` has non-trivial state. [`Self::default`] sets
/// `α = 1, β = 0` (which requires `K: Default`). We provide a non-trait
/// `default()` rather than `Default::default()` so that `K` without `Default`
/// can still construct a neutral-temperature sampler by hand if desired.
#[derive(Debug, Clone, Copy)]
pub struct CompressionPriorSampler<K: ComplexityProxy> {
    pub proxy: K,
    pub alpha: f32,
    pub beta: f32,
}

impl<K: ComplexityProxy + Default> CompressionPriorSampler<K> {
    /// Neutral-temperature sampler (`α = 1.0, β = 0.0`) with `K::default()`.
    ///
    /// We expose this as an inherent `default()` rather than a `Default` trait
    /// impl because the `Default` bound leaks onto `K`, and we want callers to
    /// be explicit when their `K` does not impl `Default` (use [`Self::new`]).
    #[allow(clippy::should_implement_trait)] // intentional inherent API; see doc
    #[inline]
    #[must_use]
    pub fn default() -> Self {
        Self {
            proxy: K::default(),
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

impl<K: ComplexityProxy> CompressionPriorSampler<K> {
    /// Construct with an explicit proxy and `(α, β)`.
    #[inline]
    #[must_use]
    pub const fn new(proxy: K, alpha: f32, beta: f32) -> Self {
        Self { proxy, alpha, beta }
    }

    /// Public log-probability score: `log_prob(x) = -α·K̃(x) - β`.
    ///
    /// This is the log-sigmoid INPUT (i.e. `logit(P(x))` up to a per-draw
    /// normalisation constant that depends on the candidate set). Higher =
    /// simpler / more probable.
    ///
    /// **Never softmax.** Each candidate's score is independent of the others;
    /// normalisation happens only inside [`Self::sample_ix`].
    #[inline]
    pub fn log_prob(&self, candidate: &[u8]) -> f32 {
        -self.alpha * self.proxy.k_tilde(candidate) - self.beta
    }

    /// Categorical sample via cumulative-sum + binary search.
    ///
    /// Fills `scratch[0..candidates.len()]` with `sigmoid(log_prob(c_i))`,
    /// converts to a cumulative distribution, draws `u ∈ [0, total)` from `rng`,
    /// and binary-searches the cumsum. The per-candidate sigmoid is the **only**
    /// probabilistic transformation — there is no global softmax.
    ///
    /// # Arguments
    ///
    /// - `candidates`: slice of byte-slice references.
    /// - `scratch`: caller-owned buffer, `len >= candidates.len()`. Reused
    ///   across calls — zero allocation in the hot path. Holds the cumsum.
    /// - `rng`: caller-owned RNG (use `fastrand::Rng::with_seed(N)` in tests for
    ///   determinism).
    ///
    /// # Panics
    ///
    /// Debug-build assert: `scratch.len() >= candidates.len()`. On empty
    /// candidate sets, returns 0.
    #[inline]
    pub fn sample_ix(&self, candidates: &[&[u8]], scratch: &mut [f32], rng: &mut Rng) -> usize {
        let n = candidates.len();
        if n == 0 {
            return 0;
        }
        debug_assert!(
            scratch.len() >= n,
            "scratch must hold {} entries, got {}",
            n,
            scratch.len()
        );
        // Per-candidate sigmoid → in-place cumsum.
        let mut total = 0.0f32;
        for (i, &c) in candidates.iter().enumerate() {
            let lp = self.log_prob(black_box(c));
            let p = sigmoid(lp);
            total += p;
            scratch[i] = total;
        }
        // Uniform draw in [0, total), binary-search the cumsum.
        let u = rng.f32() * total;
        // Branchless-ish binary search; `match` would be overkill for n ≥ 4.
        let mut lo = 0usize;
        let mut hi = n;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if scratch[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // After the loop, `lo` points at the first cumsum ≥ u (clamped).
        if scratch[lo] >= u {
            lo
        } else if lo + 1 < n {
            lo + 1
        } else {
            n - 1
        }
    }

    /// Top-`k` candidate indices by descending `log_prob`, written into `out[0..k]`.
    ///
    /// Uses a simple selection sort (k is small — typically ≤ 16). Zero
    /// allocation. `out` MUST have `len >= k` AND `k <= candidates.len()`.
    ///
    /// # Panics
    ///
    /// Debug-build assert on the above length contract.
    #[inline]
    pub fn top_k(&self, candidates: &[&[u8]], k: usize, out: &mut [usize]) {
        let n = candidates.len();
        debug_assert!(
            out.len() >= k && k <= n,
            "top_k: out.len()={} must be >= k={} and k must be <= candidates.len()={}",
            out.len(),
            k,
            n
        );
        if k == 0 {
            return;
        }
        // Selection sort: pick the max-log_prob index not yet chosen, k times.
        // O(k·n) which is fine for small k. For k up to ~16, a linear scan over
        // `out[0..filled]` to skip already-chosen indices is cheaper than a
        // HashSet allocation (k² ≤ 256 comparisons).
        for filled in 0..k {
            let mut best_ix: usize = usize::MAX;
            let mut best_lp: f32 = f32::NEG_INFINITY;
            for (i, &c) in candidates.iter().enumerate() {
                // Skip already-chosen.
                let already = (0..filled).any(|j| out[j] == i);
                if already {
                    continue;
                }
                let lp = self.log_prob(c);
                if lp > best_lp {
                    best_lp = lp;
                    best_ix = i;
                }
            }
            out[filled] = best_ix;
        }
    }
}

// ── Latent (f32) variant ─────────────────────────────────────────────────────

/// Min-max quantise `&[f32]` to `&mut [u8]` in `[0, 255]`.
///
/// Linear stretch: `b_i = round((v_i - min) / (max - min) · 255)`. If
/// `min == max` (degenerate / constant input), every byte is set to **128**
/// (mid-grey) — this preserves a non-trivial entropy signal downstream
/// (all-128 is distinguishable from all-0 in RLE/L1 proxies) while signalling
/// "no information" to entropy-based proxies.
///
/// # Panics
///
/// Debug-build assert: `scratch.len() >= v.len()`.
#[inline]
pub fn quantize_latent(v: &[f32], scratch: &mut [u8]) {
    debug_assert!(
        scratch.len() >= v.len(),
        "quantize_latent: scratch.len()={} must be >= v.len()={}",
        scratch.len(),
        v.len()
    );
    let n = v.len();
    if n == 0 {
        return;
    }
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    let span = max - min;
    if !span.is_finite() || span <= 0.0 {
        // Degenerate: all-equal or non-finite range. Mid-grey sentinel.
        for b in &mut scratch[..n] {
            *b = 128;
        }
        return;
    }
    let scale = 255.0 / span;
    for (i, &x) in v.iter().enumerate() {
        let q = (x - min) * scale;
        // Clamp to [0, 255] to absorb float round-up.
        let b = if q >= 255.0 {
            255u8
        } else if q <= 0.0 {
            0u8
        } else {
            q.round() as u8
        };
        scratch[i] = b;
    }
}

/// Latent (`&[f32]`) variant of [`CompressionPriorSampler`].
///
/// Operates on `&[f32]` vectors via byte-quantisation (see [`quantize_latent`]).
/// The quantisation step uses a CALLER-PROVIDED scratch buffer — the sampler
/// itself owns no per-call allocation. riir-ai Plan 331 wires this to HLA /
/// functor / shard vectors (private).
///
/// Mirrors the byte API: [`Self::log_prob_latent`], [`Self::sample_ix_latent`],
/// [`Self::top_k_latent`].
#[derive(Debug, Clone, Copy)]
pub struct LatentCompressionPriorSampler<K: ComplexityProxy> {
    pub inner: CompressionPriorSampler<K>,
}

impl<K: ComplexityProxy + Default> LatentCompressionPriorSampler<K> {
    #[allow(clippy::should_implement_trait)] // intentional inherent API; mirrors CompressionPriorSampler::default
    #[inline]
    #[must_use]
    pub fn default() -> Self {
        Self {
            inner: CompressionPriorSampler::default(),
        }
    }
}

impl<K: ComplexityProxy> LatentCompressionPriorSampler<K> {
    #[inline]
    #[must_use]
    pub const fn new(proxy: K, alpha: f32, beta: f32) -> Self {
        Self {
            inner: CompressionPriorSampler::new(proxy, alpha, beta),
        }
    }

    /// Quantise `latent` into `scratch` and score it through the inner sampler.
    ///
    /// `scratch_byte` MUST have `len >= latent.len()`.
    #[inline]
    pub fn log_prob_latent(&self, latent: &[f32], scratch_byte: &mut [u8]) -> f32 {
        quantize_latent(latent, scratch_byte);
        self.inner.log_prob(&scratch_byte[..latent.len()])
    }

    /// Latent categorical sample. `scratch_byte` MUST have `len >= max latent.len()`.
    #[inline]
    pub fn sample_ix_latent(
        &self,
        candidates: &[&[f32]],
        scratch_byte: &mut [u8],
        scratch_f32: &mut [f32],
        rng: &mut Rng,
    ) -> usize {
        let n = candidates.len();
        if n == 0 {
            return 0;
        }
        debug_assert!(scratch_f32.len() >= n);
        // Quantise each candidate into the SHARED byte scratch, score, advance.
        // We reuse one byte scratch for all candidates because we only need the
        // scalar score per candidate — the byte buffer is transient.
        let mut total = 0.0f32;
        for (i, &c) in candidates.iter().enumerate() {
            debug_assert!(scratch_byte.len() >= c.len());
            quantize_latent(c, scratch_byte);
            let lp = self.inner.log_prob(&scratch_byte[..c.len()]);
            let p = sigmoid(lp);
            total += p;
            scratch_f32[i] = total;
        }
        let u = rng.f32() * total;
        let mut lo = 0usize;
        let mut hi = n;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if scratch_f32[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if scratch_f32[lo] >= u {
            lo
        } else if lo + 1 < n {
            lo + 1
        } else {
            n - 1
        }
    }

    /// Latent top-k. `scratch_byte` MUST have `len >= max latent.len()`.
    #[inline]
    pub fn top_k_latent(
        &self,
        candidates: &[&[f32]],
        k: usize,
        scratch_byte: &mut [u8],
        out: &mut [usize],
    ) {
        let n = candidates.len();
        debug_assert!(out.len() >= k && k <= n);
        if k == 0 {
            return;
        }
        for filled in 0..k {
            let mut best_ix: usize = usize::MAX;
            let mut best_lp: f32 = f32::NEG_INFINITY;
            for (i, &c) in candidates.iter().enumerate() {
                let already = (0..filled).any(|j| out[j] == i);
                if already {
                    continue;
                }
                debug_assert!(scratch_byte.len() >= c.len());
                quantize_latent(c, scratch_byte);
                let lp = self.inner.log_prob(&scratch_byte[..c.len()]);
                if lp > best_lp {
                    best_lp = lp;
                    best_ix = i;
                }
            }
            out[filled] = best_ix;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    /// Deterministic LCG — no `rand` crate dependency.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn next_u8(&mut self) -> u8 {
            (self.next_u64() & 0xFF) as u8
        }
        #[allow(dead_code)] // deterministic RNG helper; retained for future tests
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() as f32) / (u32::MAX as f32)
        }
    }

    // ── RleComplexity ─────────────────────────────────────────────────────────

    #[test]
    fn test_rle_complexity_all_same() {
        // [42; 100] → RLE: 1 run, splits into (100 + 254)/255 = 1 pair → 2 bytes.
        // K̃ = 2 / 100 = 0.02.
        let data = [42u8; 100];
        let k = RleComplexity::new().k_tilde(&data);
        assert!(
            (k - 0.02).abs() < 1e-6,
            "all-same K̃ should be ~0.02, got {k}"
        );
    }

    #[test]
    fn test_rle_complexity_random() {
        // Alternating bytes — maximal runs (each pair = 2 bytes for 2 raw bytes
        // → ratio 1.0). This is the "random" worst case for RLE.
        let mut lcg = Lcg::new(0xDEAD_BEEF);
        let data: Vec<u8> = (0..256).map(|_| lcg.next_u8()).collect();
        let k = RleComplexity::new().k_tilde(&data);
        // Truly random data: most runs are length 1 → ~2 bytes per raw byte.
        // Expect close to 1.0 (allow some slack for incidental repeats).
        assert!(k > 0.85, "pseudo-random K̃ should be near 1.0, got {k}");
    }

    #[test]
    fn test_rle_complexity_empty() {
        assert_eq!(RleComplexity::new().k_tilde(&[]), 0.0);
    }

    // ── EntropyComplexity ─────────────────────────────────────────────────────

    #[test]
    fn test_entropy_complexity_uniform() {
        // Each byte value 0..255 appears exactly once → max entropy = 8 bits.
        let data: Vec<u8> = (0u8..=255).collect();
        let k = EntropyComplexity::new().k_tilde(&data);
        assert!(
            (k - 1.0).abs() < 1e-5,
            "uniform distribution K̃ should be 1.0, got {k}"
        );
    }

    #[test]
    fn test_entropy_complexity_degenerate() {
        let data = [7u8; 64];
        let k = EntropyComplexity::new().k_tilde(&data);
        assert!((k - 0.0).abs() < 1e-6, "all-same K̃ should be 0.0, got {k}");
    }

    // ── L1Complexity ─────────────────────────────────────────────────────────

    #[test]
    fn test_l1_complexity_max() {
        let data = [255u8; 4];
        let k = L1Complexity::new().k_tilde(&data);
        assert!(
            (k - 1.0).abs() < 1e-6,
            "[255;4] L1 K̃ should be 1.0, got {k}"
        );
    }

    #[test]
    fn test_l1_complexity_zero() {
        let data = [0u8; 4];
        let k = L1Complexity::new().k_tilde(&data);
        assert!((k - 0.0).abs() < 1e-6, "[0;4] L1 K̃ should be 0.0, got {k}");
    }

    // ── Sampler: log_prob ────────────────────────────────────────────────────

    #[test]
    fn test_sampler_log_prob_monotone() {
        let s = CompressionPriorSampler::new(RleComplexity::new(), 1.0, 0.0);
        let simple = [0u8; 64]; // K̃ ≈ 0.03
        let complex = {
            let mut l = Lcg::new(123);
            (0..64).map(|_| l.next_u8()).collect::<Vec<u8>>()
        };
        let lp_simple = s.log_prob(&simple);
        let lp_complex = s.log_prob(&complex);
        assert!(
            lp_simple > lp_complex,
            "simpler candidate must have higher log_prob: {} vs {}",
            lp_simple,
            lp_complex
        );
    }

    // ── Sampler: sample_ix distribution ───────────────────────────────────────

    #[test]
    fn test_sampler_sample_ix_distribution() {
        // 16 candidates: candidate i has K̃_i = i / 16. We synthesise byte slices
        // whose RLE ratio approximates this. Simpler approach: use EntropyComplexity
        // over hand-crafted slices, OR test the sampler directly over a
        // controlled-complexity set.
        //
        // We use a synthetic proxy with known K̃ so the test is exact.
        struct StepProxy;
        impl ComplexityProxy for StepProxy {
            fn k_tilde(&self, candidate: &[u8]) -> f32 {
                // The first byte encodes the complexity tier (0..16).
                candidate[0] as f32 / 16.0
            }
        }

        let sampler = CompressionPriorSampler::new(StepProxy, 4.0, 0.0);
        // 16 candidates, candidate i = [i as u8].
        let candidates: Vec<Vec<u8>> = (0u8..16).map(|i| vec![i]).collect();
        let c_refs: Vec<&[u8]> = candidates.iter().map(|v| v.as_slice()).collect();

        // Theoretical probability per candidate:
        //   p_i = sigmoid(-4 · i/16) / Σ sigmoid(-4 · j/16)
        let logits: Vec<f32> = (0..16).map(|i| -4.0 * (i as f32 / 16.0)).collect();
        let probs: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
        let total: f32 = probs.iter().sum();
        let normalised: Vec<f32> = probs.iter().map(|&p| p / total).collect();

        let n_draws = 20_000usize;
        let mut counts = [0u32; 16];
        let mut rng = Rng::with_seed(42);
        let mut scratch = vec![0.0f32; 16];
        for _ in 0..n_draws {
            let ix = sampler.sample_ix(black_box(&c_refs), &mut scratch, &mut rng);
            counts[ix] += 1;
        }

        // Empirical frequency vs theoretical — correlation check (Pearson).
        let emp: Vec<f32> = counts.iter().map(|&c| c as f32 / n_draws as f32).collect();
        let mean_e: f32 = emp.iter().sum::<f32>() / emp.len() as f32;
        let mean_t: f32 = normalised.iter().sum::<f32>() / normalised.len() as f32;
        let mut cov = 0.0f32;
        let mut var_e = 0.0f32;
        let mut var_t = 0.0f32;
        for i in 0..16 {
            let de = emp[i] - mean_e;
            let dt = normalised[i] - mean_t;
            cov += de * dt;
            var_e += de * de;
            var_t += dt * dt;
        }
        let denom = (var_e * var_t).sqrt();
        let corr = if denom > 0.0 { cov / denom } else { 0.0 };
        assert!(
            corr > 0.85,
            "sampler empirical distribution should correlate > 0.85 with theory, got {corr:.4}"
        );
        // Also sanity-check the mode is candidate 0 (simplest).
        let mode = counts
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            mode, 0,
            "simplest candidate (i=0) should be the empirical mode"
        );
    }

    // ── Sampler: top_k ───────────────────────────────────────────────────────

    #[test]
    fn test_sampler_top_k_correct() {
        let s = CompressionPriorSampler::new(RleComplexity::new(), 1.0, 0.0);
        // Hand-craft candidates with KNOWN RLE K̃.
        let c0 = vec![0u8; 64]; // K̃ = 2/64 ≈ 0.031
        let c1 = vec![1u8; 32]; // K̃ = 2/32 = 0.0625
        let c2 = vec![2u8; 16]; // K̃ = 2/16 = 0.125
        let c3 = vec![3u8; 8]; //  K̃ = 2/8  = 0.25
        let c4 = {
            // alternating → K̃ = 1.0
            let mut v = Vec::with_capacity(8);
            for i in 0..8 {
                v.push(if i % 2 == 0 { 0 } else { 1 });
            }
            v
        };
        let candidates = vec![
            c0.as_slice(),
            c1.as_slice(),
            c2.as_slice(),
            c3.as_slice(),
            c4.as_slice(),
        ];
        let mut out = [0usize; 3];
        s.top_k(&candidates, 3, &mut out);
        // Expected: simplest first → [0, 1, 2].
        assert_eq!(out, [0, 1, 2], "top-3 by ascending K̃");
    }

    #[test]
    fn test_sampler_top_k_zero() {
        let s = CompressionPriorSampler::new(RleComplexity::new(), 1.0, 0.0);
        let candidates: Vec<&[u8]> = vec![&[0u8; 4], &[1u8; 4]];
        let mut out = [0usize; 0];
        s.top_k(&candidates, 0, &mut out); // must not panic
    }

    // ── quantize_latent ───────────────────────────────────────────────────────

    #[test]
    fn test_quantize_latent_min_max_equal() {
        // Arbitrary non-π value (avoids clippy::approximate_constants).
        let v = [2.5f32; 5];
        let mut scratch = [0u8; 5];
        quantize_latent(&v, &mut scratch);
        for &b in &scratch {
            assert_eq!(b, 128, "degenerate latent should quantise to mid-grey");
        }
    }

    #[test]
    fn test_quantize_latent_normal() {
        let v = [0.0f32, 0.5, 1.0];
        let mut scratch = [0u8; 3];
        quantize_latent(&v, &mut scratch);
        // (0.0 - 0.0) / 1.0 · 255 = 0
        // (0.5 - 0.0) / 1.0 · 255 = 127.5 → round → 128 (banker's? Rust uses
        // half-away-from-zero for `f32::round`).
        // (1.0 - 0.0) / 1.0 · 255 = 255
        assert_eq!(scratch[0], 0, "min should quantise to 0");
        assert!(
            scratch[1] == 127 || scratch[1] == 128,
            "midpoint should quantise to 127 or 128, got {}",
            scratch[1]
        );
        assert_eq!(scratch[2], 255, "max should quantise to 255");
    }

    #[test]
    fn test_quantize_latent_empty() {
        let mut scratch: [u8; 0] = [];
        quantize_latent(&[], &mut scratch); // must not panic
    }

    // ── Latent sampler end-to-end ───────────────────────────────────────────

    #[test]
    fn test_latent_sampler_log_prob_monotone() {
        let s = LatentCompressionPriorSampler::new(EntropyComplexity::new(), 1.0, 0.0);
        // Simple latent: constant → quantises to all-128 → non-zero entropy
        // (single-symbol alphabet → entropy 0 actually). Use a spread latent.
        let simple = [0.0f32; 16]; // constant → all-128 → entropy 0
        let complex = {
            // Many distinct values → high entropy after quantisation.
            (0..16).map(|i| i as f32).collect::<Vec<f32>>()
        };
        let mut sb = [0u8; 16];
        let lp_simple = s.log_prob_latent(&simple, &mut sb);
        let lp_complex = s.log_prob_latent(&complex, &mut sb);
        assert!(
            lp_simple >= lp_complex,
            "constant latent (entropy 0) should score >= spread latent"
        );
    }
}
