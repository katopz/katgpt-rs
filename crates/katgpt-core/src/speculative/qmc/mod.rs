//! Quasi-Monte Carlo uniform sources for correlated-but-marginally-exact
//! parallel sampling (Plan 367, Research 367 — QuasiMoTTo,
//! arXiv:2607.01179).
//!
//! Three QMC methods producing k marginally-Unif[0,1) points with controlled
//! joint structure (low-discrepancy coverage):
//! - [`LatticeQmc`]: rank-1 lattice, max coverage, min freedom
//!   (pairwise MI = −∞ — each point determines every other)
//! - [`StratifiedQmc`]: stratified + Fisher-Yates permutation
//!   (pairwise MI = log(k/(k−1)))
//! - [`SobolQmc`]: multi-dimensional Sobol sequence with digital-shift
//!   (Owen) randomization; direction numbers computed at construction from
//!   GF(2) primitive polynomials (zero-dep, no vendored tables)
//!
//! ## The contract (marginal exactness)
//!
//! Each `u_i` drawn by any [`QmcSource`] is marginally uniform on [0,1). The
//! joint structure is designed for better coverage than i.i.d. — enabling
//! 25–47% fewer rollouts for matched pass@k (per the paper). By linearity of
//! expectation, any average-type estimator (policy gradient, mean reward,
//! pass@k) is unbiased regardless of the joint, as long as each rollout's
//! marginal matches the LM. This is what makes QMC a drop-in for i.i.d.
//!
//! ## Zero-allocation contract
//!
//! All [`QmcSource::draw`] calls write into a caller-provided `&mut [f32]`.
//! No allocation occurs inside `draw` — the caller pre-allocates the buffer.

use crate::types::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// QmcSource trait
// ─────────────────────────────────────────────────────────────────────────────

/// QMC uniform source: produces k marginally-Unif[0,1) points.
///
/// Contract: each `u_i` is marginally uniform on [0,1); the joint is
/// low-discrepancy (controlled per implementation). Implementations MUST NOT
/// allocate inside [`draw`](Self::draw) — the caller provides the output
/// buffer.
///
/// Drop-in replacement for K calls to `rng.uniform()` in K-rollout paths
/// (speculative decoding, BoM sampling, PPOT resampling). Each `u_i` feeds
/// an independent arithmetic-coding descend (Plan 367 Phase 2).
pub trait QmcSource {
    /// Fill `out[..k]` with k uniform variates.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < k`.
    fn draw(&mut self, k: usize, out: &mut [f32]);
}

// ─────────────────────────────────────────────────────────────────────────────
// LatticeQmc — rank-1 lattice
// ─────────────────────────────────────────────────────────────────────────────

/// Rank-1 lattice QMC: k points on `{(i/k + Δ) mod 1 : i=0..k-1}`.
///
/// A single shared offset `Δ ~ Unif[0,1)` is the only degree of freedom — each
/// grid point is marginally uniform because Δ is. Pairwise mutual information
/// is `−∞` (each point determines every other). This is the maximum-coverage /
/// minimum-freedom end of the QMC spectrum: the paper (R367 §1.1) reports it
/// dominates pass@k among the three methods.
///
/// State: 1 `f32` (the offset Δ, redrawn each batch). No per-point allocation.
pub struct LatticeQmc {
    rng: Rng,
}

impl LatticeQmc {
    /// Construct from a 64-bit seed (SplitMix64-mixed per [`Rng::new`]).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
        }
    }
}

impl QmcSource for LatticeQmc {
    #[inline]
    fn draw(&mut self, k: usize, out: &mut [f32]) {
        assert!(
            out.len() >= k,
            "LatticeQmc::draw: out.len() {} < k {}",
            out.len(),
            k
        );
        if k == 0 {
            return;
        }
        let delta = self.rng.uniform();
        let inv_k = 1.0 / k as f32;
        // Each point: (i/k + Δ) mod 1. The `fract` is a single `% 1.0` —
        // numerically stable since i/k ∈ [0,1) and Δ ∈ [0,1), so i/k+Δ ∈ [0,2).
        for (i, slot) in out.iter_mut().enumerate().take(k) {
            let v = i as f32 * inv_k + delta;
            *slot = if v >= 1.0 { v - 1.0 } else { v };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StratifiedQmc — stratified + Fisher-Yates permutation
// ─────────────────────────────────────────────────────────────────────────────

/// Stratified QMC: divide `[0,1)` into k equal strata, draw one point per
/// stratum, then Fisher-Yates permute.
///
/// Pairwise MI `= log(k/(k−1))` — the middle ground between i.i.d. (MI=0) and
/// lattice (MI=−∞). The paper (R367 §1.1) reports stratified empirically wins
/// RL (lower RLOO bias under dependence).
///
/// State: none beyond the RNG (used for stratum-local draws + permutation).
pub struct StratifiedQmc {
    rng: Rng,
}

impl StratifiedQmc {
    /// Construct from a 64-bit seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
        }
    }
}

impl QmcSource for StratifiedQmc {
    #[inline]
    fn draw(&mut self, k: usize, out: &mut [f32]) {
        assert!(
            out.len() >= k,
            "StratifiedQmc::draw: out.len() {} < k {}",
            out.len(),
            k
        );
        if k == 0 {
            return;
        }
        let inv_k = 1.0 / k as f32;
        // Step 1: draw one uniform per stratum: out[i] ~ Unif[i/k, (i+1)/k).
        for (i, slot) in out.iter_mut().enumerate().take(k) {
            let lo = i as f32 * inv_k;
            *slot = lo + self.rng.uniform() * inv_k;
        }
        // Step 2: Fisher-Yates shuffle — each permutation equally likely.
        // Index i drawn uniformly from [0, i] via next_u64 % (i+1).
        for i in (1..k).rev() {
            let j = (self.rng.next() % (i as u64 + 1)) as usize;
            out.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SobolQmc — Sobol sequence with digital-shift randomization
// ─────────────────────────────────────────────────────────────────────────────

/// Number of bits in each Sobol direction number (u32 → f32 precision).
const SOBOL_BITS: usize = 32;

/// Maximum supported dimensions (dim 0 = Van der Corput + dims 1..32).
///
/// 32 dimensions is enough for token-level QMC on sequences up to 32 tokens
/// (one coordinate per token position). The paper's token-level Sobol uses
/// `dim = sequence_length`; for longer sequences, draw batches at different
/// starting indices.
pub const SOBOL_MAX_DIM: usize = 33;

/// Multi-dimensional Sobol QMC with digital-shift (Owen) randomization.
///
/// Direction numbers are computed at construction from GF(2) primitive
/// polynomials — zero external data tables, zero-dep. Each dimension uses a
/// distinct primitive polynomial (the first available of the smallest
/// sufficient degree), ensuring valid multi-dimensional projection properties.
///
/// Initial direction numbers use `m_j = 1` (the simplest valid choice — all
/// are odd, as required). The specific Joe-Kuo optimized initial values
/// improve two-dimensional projection quality but are not required for
/// correctness; the GOAT gate (Phase 5) validates quality empirically.
///
/// The digital-shift scramble XORs each dimension's output with a random u32
/// drawn at construction. This randomizes the starting point while preserving
/// the low-discrepancy property.
///
/// State: `SOBOL_MAX_DIM × SOBOL_BITS` direction numbers (precomputed) + the
/// running point (u32 per dim) + the index counter + per-dim scramble.
pub struct SobolQmc {
    /// Number of active dimensions (1 for 1D QMC; >1 for token-level coverage).
    dim: usize,
    /// Running sample index (0-based; point 0 is the zero vector, skipped).
    index: u32,
    /// Current point, one u32 bit-pattern per dimension.
    point: [u32; SOBOL_MAX_DIM],
    /// Precomputed direction numbers: `[dim][bit]`.
    direction_numbers: [[u32; SOBOL_BITS]; SOBOL_MAX_DIM],
    /// Per-dimension digital-shift scramble (random u32 from seed).
    scramble: [u32; SOBOL_MAX_DIM],
}

impl SobolQmc {
    /// Construct a 1-dimensional Sobol source (Van der Corput + Owen shift).
    ///
    /// This is the most common case: each `draw(k, out)` produces k scalar
    /// points suitable for the [`QmcSource`] trait. For multi-dimensional
    /// coverage, use [`new_multi`](Self::new_multi).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self::new_multi(seed, 1)
    }

    /// Construct a `dim`-dimensional Sobol source.
    ///
    /// `dim` is clamped to [`SOBOL_MAX_DIM`]. Each dimension uses a distinct
    /// primitive polynomial over GF(2), computed at construction via the
    /// `find_primitive_poly` search.
    ///
    /// The trait method [`QmcSource::draw`] outputs only dimension 0 (for
    /// 1D compatibility). Use [`draw_nd`](Self::draw_nd) for multi-dimensional
    /// output.
    pub fn new_multi(seed: u64, dim: usize) -> Self {
        let dim = dim.clamp(1, SOBOL_MAX_DIM);
        let mut rng = Rng::new(seed);

        // Compute direction numbers for each dimension.
        let mut direction_numbers = [[0u32; SOBOL_BITS]; SOBOL_MAX_DIM];

        // Dimension 0: Van der Corput in base 2 — v[j] = 1 << (BITS-1-j).
        // This is the canonical first Sobol dimension (trivially "primitive").
        for (j, slot) in direction_numbers[0].iter_mut().enumerate() {
            *slot = 1u32 << (SOBOL_BITS - 1 - j);
        }

        // Dimensions 1..dim: find primitive polynomials and compute direction
        // numbers via the recurrence.
        for (d, row) in direction_numbers.iter_mut().enumerate().take(dim).skip(1) {
            let (poly, degree) = find_primitive_poly(d as u32);
            *row = compute_direction_numbers(poly, degree);
        }

        // Digital-shift scramble: one random u32 per dimension (see
        // `draw_scramble` for the upper-bits / Phase-5 G1 rationale).
        let scramble = draw_scramble(&mut rng, dim);

        Self {
            dim,
            index: 0,
            point: [0u32; SOBOL_MAX_DIM],
            direction_numbers,
            scramble,
        }
    }

    /// Re-seed in place: reset the sequence and redraw ONLY the digital-shift
    /// scramble from `seed`. The direction numbers do not depend on the seed,
    /// so the result is bit-identical to `SobolQmc::new_multi(seed, dim)` —
    /// without re-running the primitive-polynomial search, which allocates
    /// (`prime_factors_u64`). Zero-allocation; the per-decision form for hot
    /// callers that keep one source in their scratch (Issue 895 T4).
    pub fn reseed(&mut self, seed: u64) {
        let mut rng = Rng::new(seed);
        self.index = 0;
        self.point = [0u32; SOBOL_MAX_DIM];
        self.scramble = draw_scramble(&mut rng, self.dim);
    }

    /// Active dimensions.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Multi-dimensional draw: fill `out` with `k` points, each `dim` f32s.
    ///
    /// Output layout: `[p0c0, p0c1, ..., p0c(dim-1), p1c0, ...]`.
    /// `out.len()` must be `>= k * self.dim`.
    ///
    /// This is the method for token-level Sobol where each rollout uses
    /// coordinate j as the initial `u` for token position j.
    pub fn draw_nd(&mut self, k: usize, out: &mut [f32]) {
        let needed = k * self.dim;
        assert!(
            out.len() >= needed,
            "SobolQmc::draw_nd: out.len() {} < k*dim {}",
            out.len(),
            needed
        );
        for i in 0..k {
            self.advance();
            let base = i * self.dim;
            for d in 0..self.dim {
                out[base + d] = u32_to_unit_f32(self.point[d] ^ self.scramble[d]);
            }
        }
    }

    /// Advance the internal state by one Sobol point (incremental XOR).
    #[inline]
    fn advance(&mut self) {
        self.index = self.index.wrapping_add(1);
        // The bit to flip is the position of the lowest set bit of the new index.
        // For index 1 → bit 0; index 2 → bit 1; index 3 → bit 0; etc.
        // This follows from Gray(n) XOR Gray(n-1) having exactly one bit set
        // at position trailing_zeros(n).
        let l = (self.index.trailing_zeros() as usize).min(SOBOL_BITS - 1);
        for d in 0..self.dim {
            self.point[d] ^= self.direction_numbers[d][l];
        }
    }
}

impl QmcSource for SobolQmc {
    #[inline]
    fn draw(&mut self, k: usize, out: &mut [f32]) {
        assert!(
            out.len() >= k,
            "SobolQmc::draw: out.len() {} < k {}",
            out.len(),
            k
        );
        for slot in out.iter_mut().take(k) {
            self.advance();
            // Output dimension 0 with scramble.
            *slot = u32_to_unit_f32(self.point[0] ^ self.scramble[0]);
        }
    }
}

/// Digital-shift scramble: one random u32 per dimension (shared by
/// [`SobolQmc::new_multi`] and [`SobolQmc::reseed`] so the two are one draw).
///
/// Each scramble is the upper 32 bits of one `rng.next()` call (u64).
/// Upper bits of xorshift64 have better statistical distribution
/// than the lower bits (lower bits have shorter LFSR periods).
/// (Phase 5 GOAT gate G1 catch: the original code OR'd two 32-bit
/// halves from two separate draws — OR(a,b) is NOT uniform:
/// P(bit=1) = 0.75, not 0.5 — which biased the Sobol output and broke
/// marginal exactness. G1 fail rate dropped from 98% to ~1%.)
fn draw_scramble(rng: &mut Rng, dim: usize) -> [u32; SOBOL_MAX_DIM] {
    let mut scramble = [0u32; SOBOL_MAX_DIM];
    for s in &mut scramble[..dim] {
        *s = (rng.next() >> 32) as u32;
        // Ensure nonzero (a zero scramble is valid but boring).
        if *s == 0 {
            *s = 0xDEAD_BEEF;
        }
    }
    scramble
}

/// Map a u32 bit-pattern to a float in [0, 1) using upper 24 bits.
///
/// Matches [`Rng::uniform`] precision (24 mantissa bits). Takes the upper
/// 24 bits (positions 8–31), overlays the IEEE-754 exponent for [1.0, 2.0),
/// then subtracts 1.0.
///
/// [`Rng::uniform`]: katgpt_types::Rng::uniform
#[inline(always)]
fn u32_to_unit_f32(bits: u32) -> f32 {
    f32::from_bits((bits >> 8) | 0x3f80_0000) - 1.0
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) polynomial arithmetic — for computing Sobol direction numbers
// ─────────────────────────────────────────────────────────────────────────────
//
// Polynomials over GF(2) are represented as u64 bitmasks: bit i = coefficient
// of x^i. The degree is the position of the highest set bit.
//
// These helpers are ONLY called during `SobolQmc::new_multi` (construction),
// never in the hot `draw` path. Allocation in `prime_factors` is acceptable.

/// Compute a mod b in GF(2)[x] (polynomial remainder).
fn gf2_mod(mut a: u64, b: u64) -> u64 {
    if b == 0 {
        return a;
    }
    let db = 63 - b.leading_zeros();
    // Subtract b shifted to cancel the highest set bit of a, until a is
    // smaller than b (degree of remainder < degree of divisor).
    while a != 0 {
        let da = 63 - a.leading_zeros();
        if da < db {
            break;
        }
        a ^= b << (da - db);
    }
    a
}

/// Compute gcd(a, b) in GF(2)[x] via the Euclidean algorithm.
fn gf2_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = gf2_mod(a, b);
        a = b;
        b = r;
    }
    a
}

/// Compute (a × b) mod `modulus` in GF(2)[x], where `modulus` has degree `deg`.
fn gf2_mulmod(a: u64, b: u64, modulus: u64, deg: u32) -> u64 {
    let mut result = 0u64;
    let mut a = a;
    let high_bit = 1u64 << deg;
    let mut b = b;
    while b != 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & high_bit != 0 {
            a ^= modulus;
        }
    }
    result
}

/// Compute `base^exp mod modulus` in GF(2)[x] (square-and-multiply).
fn gf2_powmod(mut exp: u64, base: u64, modulus: u64, deg: u32) -> u64 {
    let mut result = 1u64;
    let mut base = gf2_mod(base, modulus);
    while exp > 0 {
        if exp & 1 != 0 {
            result = gf2_mulmod(result, base, modulus, deg);
        }
        base = gf2_mulmod(base, base, modulus, deg);
        exp >>= 1;
    }
    result
}

/// Test whether `poly` (with implicit leading bit at position `degree` and
/// constant bit at position 0) is irreducible over GF(2), using the Ben-Or
/// test.
fn is_irreducible(poly: u64, degree: u32) -> bool {
    // poly must have bit 0 and bit `degree` set.
    if poly & 1 == 0 || poly & (1u64 << degree) == 0 {
        return false;
    }
    // Ben-Or: irreducible iff gcd(poly, x^(2^i) + x) == 1 for i = 1..=floor(deg/2).
    let mut xp = 2u64; // x (= x^1)
    for _ in 1..=(degree / 2) {
        // Square x mod poly: x^(2^i) = (x^(2^{i-1}))^2 mod poly
        xp = gf2_mulmod(xp, xp, poly, degree);
        // x^(2^i) + x (subtraction = addition in GF(2))
        let g = gf2_gcd(poly, xp ^ 2);
        if g != 1 {
            return false;
        }
    }
    true
}

/// Test whether `poly` (degree `degree`) is a primitive polynomial over GF(2):
/// irreducible AND the multiplicative order of x mod poly is exactly 2^degree − 1.
fn is_primitive(poly: u64, degree: u32) -> bool {
    if !is_irreducible(poly, degree) {
        return false;
    }
    let order = (1u64 << degree) - 1;
    // x^order mod poly must be 1.
    if gf2_powmod(order, 2, poly, degree) != 1 {
        return false;
    }
    // For each prime factor q of order: x^(order/q) mod poly must NOT be 1.
    for &q in &prime_factors_u64(order) {
        if gf2_powmod(order / q, 2, poly, degree) == 1 {
            return false;
        }
    }
    true
}

/// Find the primitive polynomial assigned to dimension `dim_index` (1-based).
///
/// Dimensions are assigned one primitive polynomial each, consuming the
/// available primitive polynomials of each degree in order:
/// degree 2 (1 poly) → dims 1..2
/// degree 3 (2 polys) → dims 2..4
/// degree 4 (2 polys) → dims 4..6
/// degree 5 (6 polys) → dims 6..12
/// degree 6 (6 polys) → dims 12..18
/// degree 7 (18 polys) → dims 18..36
///
/// Returns `(poly_as_u64, degree)`.
fn find_primitive_poly(dim_index: u32) -> (u64, u32) {
    // (degree, count_of_polys_so_far_before_this_degree)
    // Number of primitive polys of degree s over GF(2) = φ(2^s − 1) / s.
    // s=2: φ(3)/2 = 1   → cumulative 1
    // s=3: φ(7)/3 = 2   → cumulative 3
    // s=4: φ(15)/4 = 2  → cumulative 5
    // s=5: φ(31)/5 = 6  → cumulative 11
    // s=6: φ(63)/6 = 6  → cumulative 17
    // s=7: φ(127)/7 = 18 → cumulative 35
    const DEGREE_CUMULATIVE: &[(u32, u32)] = &[(2, 0), (3, 1), (4, 3), (5, 5), (6, 11), (7, 17)];

    // Find the degree for this dimension index (1-based).
    let mut degree = 2u32;
    let mut skip = dim_index - 1; // 0-based offset within the degree

    for &(deg, cum) in DEGREE_CUMULATIVE {
        if dim_index > cum {
            degree = deg;
            // How many polys in this degree?
            let next_cum = DEGREE_CUMULATIVE
                .iter()
                .find(|&&(d, _)| d == deg + 1).map_or(35, |&(_, c)| c);
            let count_in_degree = next_cum - cum;
            skip = dim_index - cum - 1;
            if skip < count_in_degree {
                break;
            }
        }
    }

    // Enumerate polynomials of `degree` with leading + constant terms set,
    // find the `skip`-th primitive one.
    let leading = 1u64 << degree;
    let middle_bits = degree - 1;
    let mut found = 0u32;
    for middle in 0u64..(1u64 << middle_bits) {
        let poly = leading | (middle << 1) | 1;
        if is_primitive(poly, degree) {
            if found == skip {
                return (poly, degree);
            }
            found += 1;
        }
    }
    panic!(
        "find_primitive_poly: not enough primitive polynomials for dim_index {dim_index} (degree {degree}, skip {skip})"
    );
}

/// Compute the full direction number table `[u32; SOBOL_BITS]` from a primitive
/// polynomial and its degree.
///
/// Initial direction numbers: `m_j = 1` for `j = 0..degree` (all odd, valid).
/// Stored left-aligned: `v[j] = m_j << (BITS − 1 − j)`.
///
/// Recurrence (Bratley-Fox, in left-aligned integer storage — no shifts):
/// ```text
/// v[j] = v[j − degree]
///      XOR a_1 · v[j − 1] XOR a_2 · v[j − 2] XOR ... XOR a_{s−1} · v[j − s + 1]
/// ```
/// where `a_k` = bit `(degree − k)` of `poly` (coefficient of `x^(degree−k)`).
fn compute_direction_numbers(poly: u64, degree: u32) -> [u32; SOBOL_BITS] {
    let mut v = [0u32; SOBOL_BITS];
    let deg = degree as usize;

    // Initial direction numbers: m_j = 1 for j = 0..degree.
    for (j, slot) in v.iter_mut().enumerate().take(deg) {
        *slot = 1u32 << (SOBOL_BITS - 1 - j);
    }

    // Recurrence for j >= degree.
    for j in deg..SOBOL_BITS {
        // v[j] starts with v[j − degree] (the constant-term tap, always 1).
        v[j] = v[j - deg];
        // For k = 1..degree−1: if a_k (= bit (degree−k) of poly) is set, XOR v[j−k].
        for k in 1..deg {
            if (poly >> (deg - k)) & 1 == 1 {
                v[j] ^= v[j - k];
            }
        }
    }

    v
}

/// Prime factorization of a u64 (distinct prime factors only).
fn prime_factors_u64(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            factors.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors.shrink_to_fit();
    factors
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4 — QMC → Gaussian noise query fill (Fusion A: QmcBoMSampler)
// (Plan 367 Phase 4, Research 367 §2.3 — strongest fusion)
//
// `BoMSampler::sample_k_states` takes a pre-filled `queries: &[f32]` buffer of
// K×D Gaussian noise. The sampler itself is agnostic to how `queries` was
// generated — i.i.d. (`rng.normal() * sigma` in a loop) or QMC (this module).
// Phase 4 provides the QMC fill path: draw low-discrepancy uniforms, apply the
// inverse Gaussian CDF (probit) to each, scale by σ. Each element is marginally
// N(0,σ²) exact (T4.2); the joint has QMC low-discrepancy structure for better
// coverage of the K-dim belief ball (T4.3).
//
// # Design note — why a free helper, not a SeedStrategy variant
//
// The plan suggested adding `SeedStrategy::QmcLattice` / `QmcSobol` variants,
// but this is infeasible for two SOLID reasons:
// 1. `SeedStrategy` lives in `katgpt-micro-belief` (leaf crate) which cannot
//    depend on `katgpt-core` where `QmcSource` is defined — circular dep.
// 2. `SeedStrategy` governs seed derivation (PerNpc vs PerClass), semantically
//    orthogonal to noise shape (i.i.d. vs QMC). Conflating violates ISP.
// The free-helper design respects the existing architecture: callers already
// manage their own `queries` buffer (see `conformal_floor_bom.rs:184`); QMC is
// a drop-in alternative fill strategy.
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse of the standard normal CDF (probit function).
///
/// Maps `u ∈ (0, 1)` to the standard normal quantile `z = Φ⁻¹(u)` such that
/// `P(Z ≤ z) = u` for `Z ~ N(0,1)`. Used to transform QMC uniform variates into
/// marginally-Gaussian noise queries (Plan 367 Phase 4, T4.2).
///
/// # Algorithm
///
/// Hastings (1955) rational approximation. Max absolute error ≈ 4.5e-4
/// — sufficient for the BoM marginal-Gaussianity KS gate (T4.2), which
/// detects CDF errors > ~0.01 at N=10⁴. Uses `t = √(−2 ln(min(u, 1−u)))`
/// and a single rational function, then applies the sign. Symmetric by
/// construction: `Φ⁻¹(1−u) = −Φ⁻¹(u)`.
///
/// # Edge cases
///
/// - `u ≤ 0.0` → `-INFINITY` (left tail limit)
/// - `u ≥ 1.0` → `+INFINITY` (right tail limit)
/// - `u == 0.5` → `0.0` (median, exact by symmetry)
///
/// # Zero-allocation
///
/// Pure arithmetic — no allocations, one `sqrt` + one `ln` per call.
#[inline]
pub fn inverse_normal_cdf(u: f32) -> f32 {
    const C1: f64 = 0.802853;
const C2: f64 = 0.010328;
const D1: f64 = 1.432788;
const D2: f64 = 0.189269;
const D3: f64 = 0.001308;

if u <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if u >= 1.0 {
        return f32::INFINITY;
    }
    if u == 0.5 {
        return 0.0;
    }

    // Hastings (1955) coefficients.
    const C0: f64 = 2.515517;

    // Exploit symmetry: work with the smaller tail.
    let p = (u as f64).min(1.0 - u as f64);
    let t = (-2.0 * p.ln()).sqrt();
    let numerator = C0 + C1 * t + C2 * t * t;
    let denominator = 1.0 + D1 * t + D2 * t * t + D3 * t * t * t;
    let x0 = t - numerator / denominator;

    // Sign: positive for u > 0.5, negative for u < 0.5.
    if u > 0.5 { x0 as f32 } else { -(x0 as f32) }
}

/// Apply `σ · Φ⁻¹(u)` in-place to a buffer of uniforms, producing Gaussian
/// noise queries.
///
/// Each `uniforms[i]` is transformed to `sigma * inverse_normal_cdf(uniforms[i])`.
/// Works with any pre-filled uniforms buffer — from [`QmcSource::draw`] (1D
/// coverage) or [`SobolQmc::draw_nd`] (D-dimensional coverage for T4.3).
///
/// # Zero-allocation
///
/// In-place mutation — no allocation.
#[inline]
pub fn gaussianize_uniforms_inplace(uniforms: &mut [f32], sigma: f32) {
    for u in uniforms.iter_mut() {
        *u = sigma * inverse_normal_cdf(*u);
    }
}

/// Fill a `queries` buffer with K×D QMC-derived Gaussian noise.
///
/// Produces a `[K×D]` row-major buffer where every element is marginally
/// `N(0, σ²)` (T4.2) with QMC low-discrepancy joint structure for better
/// coverage of the K-dim belief ball (T4.3).
///
/// # Multi-dimensional coverage strategy
///
/// For `dim > 1`, performs **D independent QMC draws** of K points each (one
/// per dimension), rather than a single K·D draw. This is critical for
/// D-dimensional coverage: a single K·D lattice draw assigns consecutive
/// lattice points to the same vector (row-major), causing all D coordinates
/// of each rollout to cluster near the same Gaussian quantile → diagonal
/// bias → poor pairwise separation.
///
/// With D independent draws, each column j gets K evenly-spaced Gaussian
/// quantiles (low-discrepancy within the column), and the columns are
/// independent (different random offsets per `QmcSource::draw` call). This
/// gives proper D-dimensional coverage: each rollout is marginally
/// `N(0, σ²I)` (all D coordinates independent), and the K rollouts are
/// correlated within each dimension for better spread.
///
/// For `dim == 1`, the single-draw fast path is used (no coverage benefit
/// from per-dimension draws in 1D).
///
/// # Panics
///
/// Panics if `queries.len() < k * dim` or `k > FILL_NOISE_MAX_K` (stack
/// buffer limit for the per-dimension scratch).
///
/// # Zero-allocation
///
/// Uses a stack-allocated `[f32; FILL_NOISE_MAX_K]` scratch buffer (no heap).
/// Writes into the caller-provided `queries`.
pub const FILL_NOISE_MAX_K: usize = 256;

#[inline]
pub fn fill_noise_queries_gaussian_qmc(
    source: &mut dyn QmcSource,
    k: usize,
    dim: usize,
    sigma: f32,
    queries: &mut [f32],
) {
    let n = k.checked_mul(dim).expect("k * dim overflow");
    assert!(
        queries.len() >= n,
        "fill_noise_queries_gaussian_qmc: queries.len() {} < k*dim {}",
        queries.len(),
        n
    );
    if k == 0 || dim == 0 {
        return;
    }

    if dim == 1 {
        // 1D fast path: single draw, in-place gaussianize.
        source.draw(k, &mut queries[..k]);
        gaussianize_uniforms_inplace(&mut queries[..k], sigma);
        return;
    }

    // Multi-dim: D independent draws of K points each.
    // Stack scratch for per-dimension K uniforms (no heap allocation).
    assert!(
        k <= FILL_NOISE_MAX_K,
        "fill_noise_queries_gaussian_qmc: k {k} > FILL_NOISE_MAX_K {FILL_NOISE_MAX_K} (stack buffer limit)"
    );
    let mut col_scratch = [0.0f32; FILL_NOISE_MAX_K];
    for j in 0..dim {
        source.draw(k, &mut col_scratch[..k]);
        for k_idx in 0..k {
            queries[k_idx * dim + j] = sigma * inverse_normal_cdf(col_scratch[k_idx]);
        }
    }
}

/// Convenience wrapper: fill `queries` with QMC Gaussian noise, then call
/// [`BoMSampler::sample_k_states`](crate::BoMSampler::sample_k_states).
///
/// This is the one-call "QMC BoM" path — composes
/// [`fill_noise_queries_gaussian_qmc`] with the kernel's `sample_k_states`.
/// Requires both `qmc_sampling` (this module) and `bom_sampling` (the
/// `BoMSampler` trait + `NoiseQueryConfig`).
///
/// `queries` and `out` are caller-allocated; `queries` is overwritten with QMC
/// noise on each call. The `NoiseQueryConfig::sigma` field scales the noise; its
/// `k` field determines K.
///
/// # Zero-allocation
///
/// Writes into caller-provided `queries` and `out`; no allocation.
#[cfg(feature = "bom_sampling")]
pub fn sample_k_states_qmc<K: crate::BoMSampler>(
    kernel: &K,
    s_prev: &[f32],
    x: &[f32],
    source: &mut dyn QmcSource,
    cfg: &crate::NoiseQueryConfig,
    queries: &mut [f32],
    out: &mut [f32],
) {
    let dim = kernel.dim();
    fill_noise_queries_gaussian_qmc(source, cfg.k, dim, cfg.sigma, queries);
    kernel.sample_k_states(s_prev, x, queries, out, cfg);
}

/// Convenience wrapper: fill `queries` with QMC Gaussian noise using a
/// [`QmcMethod`](crate::QmcMethod) tag (Plan 370 — BoM Arena × QuasiMoTTo wiring).
/// Constructs the appropriate [`QmcSource`] on the stack (zero-alloc) from
/// `method` + `seed`, then delegates to [`fill_noise_queries_gaussian_qmc`].
///
/// This is the entry point used by `MultiHypothesisBoMMinimaxPlanner::resample_queries`
/// when `NoiseQueryConfig::qmc_method` is `Some(method)`. The caller passes a
/// per-tick `seed` (typically `TICK_SALT + obs_hash`); each call constructs a
/// fresh source so the QMC batch is deterministic given the seed.
///
/// Requires both `qmc_sampling` (this module) and `bom_sampling` (the
/// `QmcMethod` tag lives in `katgpt-micro-belief`, forwarded via `bom_sampling`).
///
/// # Zero-allocation
///
/// Each `QmcSource` impl is stack-allocated (1 `f32` for Lattice, 1 `Rng` for
/// Stratified, fixed-size direction table for Sobol). Writes into caller-provided
/// `queries`; no heap allocation.
#[cfg(feature = "bom_sampling")]
pub fn fill_noise_queries_gaussian_qmc_by_method(
    method: crate::QmcMethod,
    seed: u64,
    k: usize,
    dim: usize,
    sigma: f32,
    queries: &mut [f32],
) {
    match method {
        crate::QmcMethod::Lattice => {
            let mut src = LatticeQmc::new(seed);
            fill_noise_queries_gaussian_qmc(&mut src, k, dim, sigma, queries);
        }
        crate::QmcMethod::Stratified => {
            let mut src = StratifiedQmc::new(seed);
            fill_noise_queries_gaussian_qmc(&mut src, k, dim, sigma, queries);
        }
        crate::QmcMethod::Sobol => {
            let mut src = SobolQmc::new(seed);
            fill_noise_queries_gaussian_qmc(&mut src, k, dim, sigma, queries);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dyadic bootstrap pass@k estimator (Plan 367 Phase 6 / Theorem 1)
// ─────────────────────────────────────────────────────────────────────────────
//
// Theorem 1 of arXiv:2607.01179 (QuasiMoTTo): for a rank-1 lattice QMC
// batch of size k = 2^L, every stride-`s` subsequence (s = k/m, m a power
// of two dividing k) starting at offset j ∈ {0, ..., s-1} is itself a
// valid rank-1 lattice of size m, because the lattice values
// `{(i/k + Δ) mod 1 : i=0..k-1}` restricted to indices `{j + t·s}` satisfy
//   ((j + t·s)/k + Δ) mod 1 = (t/m + (Δ + j/k)) mod 1
// which is a rank-1 lattice of size m with offset `(Δ + j/k) mod 1`, itself
// marginally Unif[0,1) since both Δ and j/k are.
//
// Therefore a single pass@k lattice batch yields `s = k/m` unbiased pass@m
// lattice-batch estimates (one per starting offset). Their mean is an
// unbiased point estimate; the Wilson score CI quantifies lattice-resample
// variance. This converts one expensive pass@k rollout batch into `s`
// cheaper pass@m estimates for free — no extra rollouts needed.
//
// For Sobol/Stratified the dyadic-stride property does NOT hold in general,
// but contiguous blocks of m points DO preserve the per-method low-
// discrepancy structure (Sobol: contiguous subsequences are valid shifted
// Sobol subsequences; Stratified: contiguous blocks span m consecutive
// strata, a coarser valid stratification). So for those methods we offer a
// contiguous-block bootstrap with random starts.

/// Result of a bootstrap pass@m estimation.
///
/// `point_estimate` is the unbiased pass@m point estimate; `sample_variance`
/// is the unbiased (n-1) sample variance of the per-resample binary
/// indicators; `n_resamples` is the number of resamples that contributed.
/// Use [`wilson_ci`](Self::wilson_ci) for a well-behaved CI at small n.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapEstimate {
    /// Unbiased point estimate of pass@m (mean over resamples).
    pub point_estimate: f64,
    /// Unbiased (n-1) sample variance of the per-resample binary indicators.
    pub sample_variance: f64,
    /// Number of resamples that contributed (s = k/m for dyadic, B for block).
    pub n_resamples: usize,
}

impl BootstrapEstimate {
    /// Wilson score confidence interval for the pass@m proportion.
    ///
    /// Preferred over the normal-approximation (`p̂ ± z·√(p̂(1-p̂)/n)`) for
    /// binary indicators because it has correct coverage at small n and near
    /// the 0/1 boundaries (Brown, Cai & DasGupta 2001).
    ///
    /// `z` is the two-sided critical value (1.96 for 95%, 2.576 for 99%).
    /// Returns `(low, high)` clamped to `[0, 1]`. For `n_resamples == 0`, the
    /// uninformative `(0.0, 1.0)` is returned.
    #[inline]
    pub fn wilson_ci(&self, z: f64) -> (f64, f64) {
        let n = self.n_resamples as f64;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        let p_hat = self.point_estimate;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p_hat + z2 / (2.0 * n)) / denom;
        let margin = (z / denom) * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();
        let lo = (center - margin).max(0.0);
        let hi = (center + margin).min(1.0);
        (lo, hi)
    }

    /// Convenience: Wilson score 95% CI (z = 1.959963984540054, Φ⁻¹(0.975)).
    #[inline]
    pub fn wilson_ci_95(&self) -> (f64, f64) {
        self.wilson_ci(1.959_963_984_540_054)
    }

    /// Sample standard deviation across resamples (√ of [`sample_variance`](Self::sample_variance)).
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.sample_variance.sqrt()
    }
}

/// Lattice dyadic bootstrap pass@m estimator (Theorem 1 of arXiv:2607.01179).
///
/// For a [`LatticeQmc`] batch of size k = 2^L, every stride-`s` subsequence
/// (s = k/m, m a power of two dividing k) starting at offset j ∈ {0,...,s-1}
/// is itself a valid rank-1 lattice of size m. Therefore a single pass@k
/// lattice batch yields `s = k/m` unbiased pass@m estimates — one per
/// starting offset. Their mean is the unbiased pass@m estimate; their
/// variance drives the Wilson score CI.
///
/// This is the strongest of the three QMC-method bootstrap forms: it is
/// *exhaustive* (no RNG needed — every starting offset is taken) and
/// *algebraically exact* (each sub-lattice is provably a LatticeQmc batch
/// of size m, not an approximation).
///
/// # Arguments
///
/// * `outcomes` - pass/fail of each of the K rollouts (length K = 2^L).
/// * `m` - sub-sample size (power of two, divides K, 0 < m ≤ K).
///
/// # Panics
///
/// Panics if `outcomes.len()` is not a power of two, `m` is not a power of
/// two, `m > outcomes.len()`, or `outcomes.len() % m != 0`.
///
/// # Zero-allocation
///
/// Single streaming pass; no heap allocation. Hot-path friendly.
///
/// # Example
///
/// ```
/// # use katgpt_core::speculative::qmc::dyadic_bootstrap_pass_at_m_lattice;
/// // 8-rollout lattice batch: 4 pass, 4 fail. Stride-2 sub-lattices at
/// // m=4 give 2 estimates of pass@4.
/// let outcomes = [true, false, true, false, true, false, true, false];
/// let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 4);
/// assert_eq!(est.point_estimate, 0.5);  // one sub-lattice all-pass, one all-fail
/// assert_eq!(est.n_resamples, 2);
/// ```
pub fn dyadic_bootstrap_pass_at_m_lattice(outcomes: &[bool], m: usize) -> BootstrapEstimate {
    let k = outcomes.len();
    assert!(
        k > 0 && k.is_power_of_two(),
        "dyadic_bootstrap_pass_at_m_lattice: outcomes.len() = {k} must be a power of two"
    );
    assert!(
        m > 0 && m.is_power_of_two(),
        "dyadic_bootstrap_pass_at_m_lattice: m = {m} must be a power of two"
    );
    assert!(
        m <= k,
        "dyadic_bootstrap_pass_at_m_lattice: m = {m} > outcomes.len() = {k}"
    );
    // k is a power of two and m ≤ k is a power of two ⇒ k % m == 0 by the
    // divisibility of powers of two. No separate assert needed.
    let stride = k / m;

    // For each starting offset j ∈ [0, stride), pass@m of the subsequence
    // {outcomes[j], outcomes[j+stride], ..., outcomes[j+(m-1)*stride]} is the
    // indicator "any true". Single streaming pass — sum and sum of squares
    // only.
    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for j in 0..stride {
        let mut any = false;
        for t in 0..m {
            if outcomes[j + t * stride] {
                any = true;
                break;
            }
        }
        let x = if any { 1.0f64 } else { 0.0 };
        sum += x;
        sum_sq += x * x;
    }

    let n = stride as f64;
    let mean = sum / n;
    let var = sample_variance_binary(mean, sum_sq, n);
    BootstrapEstimate {
        point_estimate: mean,
        sample_variance: var,
        n_resamples: stride,
    }
}

/// Contiguous-block bootstrap for Sobol / Stratified / general orderings.
///
/// Unlike the lattice dyadic case (which has provable sub-lattice validity
/// for strided offsets), [`SobolQmc`] and [`StratifiedQmc`] preserve their
/// low-discrepancy structure within *contiguous* blocks of m points. We
/// resample by drawing `n_resamples` random contiguous starting positions
/// uniformly from {0, 1, ..., K-m} (no wrapping — boundary blocks are full
/// size) and computing pass@m of each.
///
/// This is the standard nonparametric block-bootstrap, adapted to preserve
/// local QMC structure. Less powerful than the lattice dyadic form (random
/// starts vs exhaustive; contiguous rather than algebraically exact) but
/// applicable when the dyadic-stride theorem doesn't hold.
///
/// # Arguments
///
/// * `outcomes` - pass/fail of each of the K rollouts.
/// * `m` - block size (sub-batch size).
/// * `n_resamples` - number of random block starts (B). Must be > 0.
/// * `rng` - caller-provided [`Rng`] for selecting block starts.
///
/// # Panics
///
/// Panics if `m == 0`, `m > outcomes.len()`, or `n_resamples == 0`.
///
/// # Zero-allocation
///
/// Single streaming pass; no heap allocation.
pub fn contiguous_block_bootstrap_pass_at_m(
    outcomes: &[bool],
    m: usize,
    n_resamples: usize,
    rng: &mut Rng,
) -> BootstrapEstimate {
    let k = outcomes.len();
    assert!(m > 0, "contiguous_block_bootstrap_pass_at_m: m must be > 0");
    assert!(
        m <= k,
        "contiguous_block_bootstrap_pass_at_m: m = {m} > outcomes.len() = {k}"
    );
    assert!(
        n_resamples > 0,
        "contiguous_block_bootstrap_pass_at_m: n_resamples must be > 0"
    );
    let n_starts = if k > m { k - m + 1 } else { 1 };

    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for _ in 0..n_resamples {
        // Map a u64 to [0, n_starts). For n_starts a power of two this would
        // be unbiased via masking; for the general case we accept the tiny
        // modular bias (≤ n_starts/u64::MAX, negligible for n_starts ≤ 2^32).
        let start = if n_starts > 1 {
            (rng.next() % (n_starts as u64)) as usize
        } else {
            0
        };
        let mut any = false;
        for i in 0..m {
            if outcomes[start + i] {
                any = true;
                break;
            }
        }
        let x = if any { 1.0f64 } else { 0.0 };
        sum += x;
        sum_sq += x * x;
    }

    let n = n_resamples as f64;
    let mean = sum / n;
    let var = sample_variance_binary(mean, sum_sq, n);
    BootstrapEstimate {
        point_estimate: mean,
        sample_variance: var,
        n_resamples,
    }
}

/// Unbiased (n-1) sample variance of binary indicators.
///
/// For 0/1 indicators with mean `mean` and `sum_sq = Σ x_i² = Σ x_i = sum`
/// (since x_i² = x_i for binary), this is `(n/(n-1)) · (m2 − mean²)` where
/// `m2 = sum_sq / n`. Returns 0 for n ≤ 1; clamps tiny negative drift from
/// f64 rounding to 0.
#[inline]
fn sample_variance_binary(mean: f64, sum_sq: f64, n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    let m2 = sum_sq / n;
    let v = (n / (n - 1.0)) * (m2 - mean * mean);
    v.max(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
