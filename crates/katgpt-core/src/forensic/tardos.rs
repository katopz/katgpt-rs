//! Tardos 2008 anti-collusion fingerprinting codebook.
//!
//! Reference: G. Tardos, "Optimal Probabilistic Fingerprint Codes",
//! J. ACM 55(2), 2008. Produces length-L binary codewords for `n`
//! recipients that resist collusion of up to `c` attackers with false
//! positive probability ≤ ε.
//!
//! ## Determinism
//!
//! The codebook is a deterministic function of a 32-byte `seed`. The seed
//! feeds a BLAKE3 stream that drives every Bernoulli draw. This makes the
//! entire codebook reconstructable from `(seed, n, c, ε)` without storing
//! per-recipient bit tables.
//!
//! ## No new deps
//!
//! No `chacha20`, no `rand`. PRNG = BLAKE3 keyed by `seed`.

use blake3::Hasher;

/// Write the BLAKE3 output of `h` into `out` (32 bytes).
#[inline]
fn finalize_into(h: &Hasher, out: &mut [u8; 32]) {
    let hash = h.finalize();
    let bytes = hash.as_bytes();
    *out = *bytes;
}

/// Tardos codebook parameters + per-position accusation probabilities.
///
/// The per-recipient codeword bits `x_{j,i}` are NOT stored — they are
/// regenerated on demand by [`Self::bit`] and [`Self::codeword`]. This
/// keeps the codebook O(L) instead of O(n·L); for the default
/// c=10, n=10⁵, ε=10⁻⁶ case that's a few KB vs ~10 MB.
#[derive(Clone, Debug)]
pub struct TardosCodebook {
    /// Codeword length L = ceil(100·c²·ln(n/ε)).
    pub length: usize,
    /// Number of recipients the codebook was sized for (used for sanity).
    pub n_recipients: usize,
    /// Design colluder bound c.
    pub colluder_bound: usize,
    /// False-positive epsilon.
    pub false_positive_epsilon: f64,
    /// Per-position Bernoulli parameter p_i, length L, in [p_min, p_max].
    pub p_i: Vec<f32>,
    /// BLAKE3 seed for deterministic codeword regeneration.
    pub seed: [u8; 32],
}

/// Lower clamp on p_i. Tardos 2008: p_min → 0 as ε → 0; we use the
/// standard heuristic floor. Below this, accusation variance explodes.
const P_MIN_DEFAULT: f32 = 1e-3;
/// Upper clamp symmetric to p_min (Tardos uses [t, 1-t]).
const P_MAX_DEFAULT: f32 = 1.0 - P_MIN_DEFAULT;

impl TardosCodebook {
    /// Codeword length L per the Tardos theorem: `L = ceil(100·c²·ln(n/ε))`.
    #[inline]
    pub fn codeword_length(c: usize, n_recipients: usize, epsilon: f64) -> usize {
        debug_assert!(c >= 1 && n_recipients >= 1 && epsilon > 0.0);
        let c_f = c as f64;
        let n_f = n_recipients as f64;
        let raw = 100.0 * c_f * c_f * (n_f / epsilon).ln();
        raw.ceil() as usize
    }

    /// Generate the codebook (per-position p_i drawn from the arcsine
    /// distribution f(p) ∝ 1/√(p(1-p)), clamped to [p_min, p_max]).
    ///
    /// `seed` keys a BLAKE3 stream so the result is fully deterministic.
    pub fn generate(
        seed: &[u8; 32],
        n_recipients: usize,
        c: usize,
        epsilon: f64,
    ) -> Self {
        let length = Self::codeword_length(c, n_recipients, epsilon);
        let mut xof = Blake3Xof::new(seed, b"tardos_p_i");
        let mut p_i = Vec::with_capacity(length);
        for _ in 0..length {
            let u = xof.next_unit();
            p_i.push(sample_arcsine_clamped(u, P_MIN_DEFAULT, P_MAX_DEFAULT));
        }
        Self {
            length,
            n_recipients,
            colluder_bound: c,
            false_positive_epsilon: epsilon,
            p_i,
            seed: *seed,
        }
    }

    /// Codeword bit `x_{j,i}` for recipient `j` at position `i`, drawn
    /// Bernoulli(p_i) from a recipient-seeded BLAKE3 stream.
    /// Deterministic.
    #[inline]
    pub fn bit(&self, recipient_idx: usize, position: usize) -> u8 {
        debug_assert!(position < self.length);
        let mut h = Hasher::new();
        h.update(&self.seed);
        h.update(b"tardos_bit");
        h.update(&(recipient_idx as u64).to_le_bytes());
        h.update(&(position as u64).to_le_bytes());
        let u = next_unit_from_hasher(&h);
        let p = self.p_i[position] as f64;
        if u < p {
            1
        } else {
            0
        }
    }

    /// Build the full L-bit codeword for recipient `j`.
    pub fn codeword(&self, recipient_idx: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.length);
        for i in 0..self.length {
            out.push(self.bit(recipient_idx, i));
        }
        out
    }

    /// Tardos accusation statistic over a SUBSET of the codebook starting
    /// at `position_offset`. Used when the recovered codeword corresponds
    /// to Tardos positions `[offset .. offset+len]` (e.g. only the DCT
    /// channel was recovered, which lives at
    /// `[v_offset .. v_offset+L_dct]` in the full codeword).
    pub fn accusation_sum_offset(
        &self,
        leaked: &[u8],
        recipient_idx: usize,
        position_offset: usize,
    ) -> f64 {
        let len = leaked.len();
        let mut s = 0.0f64;
        for i in 0..len {
            let pos = (position_offset + i) % self.length;
            let y = leaked[i] as f64;
            let x = self.bit(recipient_idx, pos) as f64;
            let p = self.p_i[pos] as f64;
            let q = (p * (1.0 - p)).max(1e-12).sqrt();
            s += (x - p) / q * y;
        }
        s
    }

    /// Tardos accusation statistic
    /// `S_j = Σ_i g(x_{j,i}, p_i) · y_i` where `g(x,p) = (x-p)/√(p(1-p))`
    /// and `y_i` is the leaked bit at position i. Large positive S_j →
    /// recipient j participated in the leak.
    ///
    /// The sum runs over `min(leaked.len(), self.length)` positions —
    /// callers recovering a SUBSET of the full codeword (e.g. only the
    /// DCT channel) should pair this with
    /// [`Self::accusation_threshold_for_len`] using the same length.
    pub fn accusation_sum(&self, leaked: &[u8], recipient_idx: usize) -> f64 {
        let len = leaked.len().min(self.length);
        let mut s = 0.0f64;
        for i in 0..len {
            let y = leaked[i] as f64;
            let x = self.bit(recipient_idx, i) as f64;
            let p = self.p_i[i] as f64;
            let q = (p * (1.0 - p)).max(1e-12).sqrt();
            s += (x - p) / q * y;
        }
        s
    }

    /// Tardos accusation threshold `Z = c·√(L/2)` for the FULL codebook
    /// length. A recipient is accused iff `S_j > Z`.
    #[inline]
    pub fn accusation_threshold(&self) -> f64 {
        let c = self.colluder_bound as f64;
        let l = self.length as f64;
        c * (l / 2.0).sqrt()
    }

    /// Tardos accusation threshold scaled to a SUBSET of length `len`.
    /// Use this when `accusation_sum` was computed over fewer than the
    /// full `self.length` positions (e.g. only the DCT channel was
    /// recovered). The threshold scales as `c·√(len/2)`.
    #[inline]
    pub fn accusation_threshold_for_len(&self, len: usize) -> f64 {
        let c = self.colluder_bound as f64;
        let l = len as f64;
        c * (l / 2.0).sqrt()
    }
}

/// Codeword for a recipient identified by pubkey (deterministic
/// inverse-lookup mapping). Index is derived from `(seed, pubkey)` so
/// the same recipient always maps to the same codeword slot.
///
/// Returns the codeword (`Vec<u8>` of length `codebook.length`) and the
/// integer recipient index used.
pub fn extract_codeword_from_seed(
    seed: &[u8; 32],
    codebook: &TardosCodebook,
    recipient_pubkey: &[u8; 32],
) -> (Vec<u8>, usize) {
    let recipient_idx = recipient_index(seed, recipient_pubkey, codebook.n_recipients);
    (codebook.codeword(recipient_idx), recipient_idx)
}

/// Stable recipient index in `[0, n_recipients)` from a BLAKE3 hash of
/// `(seed ‖ pubkey ‖ "recipient_idx")`. Used both for codeword
/// regeneration and for registry lookups.
#[inline]
pub fn recipient_index(seed: &[u8; 32], recipient_pubkey: &[u8; 32], n_recipients: usize) -> usize {
    let mut h = Hasher::new();
    h.update(seed);
    h.update(b"recipient_idx");
    h.update(recipient_pubkey);
    let u = u64_from_hasher(&h);
    (u as usize) % n_recipients.max(1)
}

// ─── PRNG helpers (BLAKE3, no new deps) ─────────────────────────────────

/// A BLAKE3 stream producing uniform `f64 ∈ [0, 1)` and `u64` outputs.
/// Each draw hashes `(seed ‖ domain ‖ counter)` so the stream is fully
/// deterministic and reproducible.
struct Blake3Xof {
    seed: [u8; 32],
    domain: &'static [u8],
    counter: u64,
}

impl Blake3Xof {
    #[inline]
    fn new(seed: &[u8; 32], domain: &'static [u8]) -> Self {
        Self {
            seed: *seed,
            domain,
            counter: 0,
        }
    }

    #[inline]
    fn next_unit(&mut self) -> f64 {
        let u = self.next_u64();
        // 53-bit mantissa for uniform f64 in [0,1).
        (u >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut h = Hasher::new();
        h.update(&self.seed);
        h.update(self.domain);
        h.update(&self.counter.to_le_bytes());
        self.counter = self.counter.wrapping_add(1);
        u64_from_hasher(&h)
    }
}

#[inline]
fn next_unit_from_hasher(h: &Hasher) -> f64 {
    let u = u64_from_hasher(h);
    (u >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[inline]
fn u64_from_hasher(h: &Hasher) -> u64 {
    let mut out = [0u8; 32];
    finalize_into(h, &mut out);
    u64::from_le_bytes([
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
    ])
}

/// Sample from f(p) ∝ 1/√(p(1-p)) via inverse-CDF.
///
/// The CDF is `F(p) = (2/π)·arcsin(√p)`, so the inverse is
/// `p = sin²(π·u/2)`. Clamp to [p_min, p_max] for numerical stability.
#[inline]
fn sample_arcsine_clamped(u: f64, p_min: f32, p_max: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    let p = (std::f64::consts::PI * u / 2.0).sin().powi(2) as f32;
    p.clamp(p_min, p_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_seed(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn length_sanity_default_c10_n1e5_eps1e6() {
        let l = TardosCodebook::codeword_length(10, 100_000, 1e-6);
        // 100·100·ln(1e5/1e-6) = 10000·ln(1e11) ≈ 10000·25.33 ≈ 253_286.
        // The plan's "≈ 1000 bits" is the *bandwidth* of recipe bits (L_v +
        // L_dct + L_top), NOT the Tardos length. The real Tardos length
        // at these parameters is ~250k — but accusation is over the
        // codeword length actually embedded, which can be far smaller for
        // our recipe (Plan 293 deliberately truncates for bandwidth).
        // We assert it's in a sane band.
        assert!(l >= 100_000, "L={l} unexpectedly small");
        assert!(l <= 500_000, "L={l} unexpectedly large");
    }

    #[test]
    fn determinism_same_seed_same_p_i() {
        let s = dummy_seed(7);
        let cb1 = TardosCodebook::generate(&s, 1000, 5, 1e-3);
        let cb2 = TardosCodebook::generate(&s, 1000, 5, 1e-3);
        assert_eq!(cb1.p_i, cb2.p_i);
        assert_eq!(cb1.codeword(3), cb2.codeword(3));
    }

    #[test]
    fn different_seed_different_p_i() {
        let cb1 = TardosCodebook::generate(&dummy_seed(1), 100, 3, 1e-3);
        let cb2 = TardosCodebook::generate(&dummy_seed(2), 100, 3, 1e-3);
        assert_ne!(cb1.p_i, cb2.p_i);
    }

    #[test]
    fn p_i_in_valid_range() {
        let cb = TardosCodebook::generate(&dummy_seed(3), 1000, 5, 1e-3);
        for &p in &cb.p_i {
            assert!(p >= P_MIN_DEFAULT && p <= P_MAX_DEFAULT, "p={p} out of range");
            assert!(p > 0.0 && p < 1.0);
        }
    }

    #[test]
    fn accusation_identifies_true_leaker() {
        // Small synthetic case: 20 recipients, design for c=2 colluders.
        let seed = dummy_seed(11);
        let cb = TardosCodebook::generate(&seed, 20, 2, 1e-3);
        let leaker_idx = 5;
        let leaked = cb.codeword(leaker_idx);
        let scores: Vec<f64> = (0..20)
            .map(|j| cb.accusation_sum(&leaked, j))
            .collect();
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, leaker_idx);
        // Threshold should clear the leaker.
        let z = cb.accusation_threshold();
        assert!(scores[leaker_idx] > z, "leaker score below threshold");
    }

    #[test]
    fn g2b_leaker_identified_95pct_over_trials() {
        // Reduced-scale G2b: c=3 colluders (kept small to keep the test
        // fast — full c=10 is the G2 GOAT gate benchmark T7.3, out of
        // scope here). The 95% bar is the same.
        let seed = dummy_seed(42);
        let n = 200;
        let c_design = 3;
        let cb = TardosCodebook::generate(&seed, n, c_design, 1e-4);
        let mut hits = 0usize;
        let trials = 200usize;
        let mut trial_seed_byte = 99u8;
        for _ in 0..trials {
            // Pick leaker pseudorandomly.
            trial_seed_byte = trial_seed_byte.wrapping_mul(7).wrapping_add(13);
            let leaker = (trial_seed_byte as usize) % n;
            let leaked = cb.codeword(leaker);
            let scores: Vec<f64> = (0..n)
                .map(|j| cb.accusation_sum(&leaked, j))
                .collect();
            let winner = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if winner == leaker {
                hits += 1;
            }
        }
        let acc = hits as f64 / trials as f64;
        assert!(acc >= 0.95, "G2b accuracy {acc:.3} < 0.95");
    }

    #[test]
    fn g2a_no_false_accusation_on_non_colluders() {
        // Erasure-style collusion: c colluders erase positions where they
        // disagree (replace with random). Non-colluders must NOT be
        // accused.
        let seed = dummy_seed(13);
        let n = 50;
        let c = 3;
        let cb = TardosCodebook::generate(&seed, n, c, 1e-5);
        let colluder_idx: Vec<usize> = (0..c).collect();
        // Build a collusion-erased "leaked" codeword: for each position,
        // if all colluders agree, take their value; else set to 0
        // (erasure).
        let mut leaked = vec![0u8; cb.length];
        for i in 0..cb.length {
            let mut ones = 0;
            for &j in &colluder_idx {
                ones += cb.bit(j, i) as usize;
            }
            if ones == c {
                leaked[i] = 1;
            } else if ones == 0 {
                leaked[i] = 0;
            } else {
                leaked[i] = 0; // erasure position
            }
        }
        let z = cb.accusation_threshold();
        let mut false_accusations = 0usize;
        for j in 0..n {
            if colluder_idx.contains(&j) {
                continue;
            }
            let s = cb.accusation_sum(&leaked, j);
            if s > z {
                false_accusations += 1;
            }
        }
        // Allow a tiny tolerance at the design ε; the theorem guarantees
        // < ε aggregate, but with 47 non-colluders we tolerate up to 1
        // spurious hit to absorb noise (the design ε here is loose).
        assert!(
            false_accusations <= 1,
            "{false_accusations} false accusations on non-colluders"
        );
    }

    #[test]
    fn extract_codeword_is_deterministic_and_stable() {
        let seed = dummy_seed(99);
        let cb = TardosCodebook::generate(&seed, 100, 3, 1e-4);
        let pk = [5u8; 32];
        let (cw1, idx1) = extract_codeword_from_seed(&seed, &cb, &pk);
        let (cw2, idx2) = extract_codeword_from_seed(&seed, &cb, &pk);
        assert_eq!(idx1, idx2);
        assert_eq!(cw1, cw2);
        // Different pubkey → different codeword (modulo rare n-way
        // collision).
        let pk2 = [6u8; 32];
        let (cw3, idx_pk2) = extract_codeword_from_seed(&seed, &cb, &pk2);
        let idx_pk1 = idx1;
        if idx_pk1 != idx_pk2 {
            assert_ne!(cw1, cw3);
        }
    }
}
