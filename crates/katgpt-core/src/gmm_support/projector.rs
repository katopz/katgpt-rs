//! Johnson–Lindenstrauss projector, revived at E ≥ 32 (Plan 618 T3).
//!
//! # History (Issue 139 — carried so nobody re-shrinks it)
//!
//! The shipped [`JlProjectionMatrix`](crate::shard_embedding::JlProjectionMatrix)
//! (Plan 230) is **deprecated at m=8**: the JL lower bound is violated by
//! over 200× there (1.4–6% NN preservation vs the documented 90% target;
//! Issue 139, 2026-07-16). The projection is SOUND at `E ≥ 256` for the
//! matrix counts this stack uses, and the fit-space law measured at the
//! first consumer (riir-reflex Issue 066 PRE-CHECK: JL k=64 rescues
//! Gaussianity on 253/254 label pools, k-stable across 64/32/16) puts the
//! useful band at **E ≈ 32–64**. This module therefore revives the
//! construction as a const-generic `E ≥ 32` primitive and REFUSES to
//! compile for `E < 32` (a compile-error guard, not a runtime warning —
//! the m=8 mistake was a shape decision, and shape decisions belong in
//! types).
//!
//! # Construction
//!
//! Seeded Rademacher: each matrix entry is `±1/√E` (bit from a SplitMix64
//! stream, row-major). Norm-preserving in expectation:
//! `E[‖Px‖²] = ‖x‖²`.
//!
//! # Packed sign-bit storage (verdict round 2)
//!
//! The `±1/√E` entries are stored as ONE BIT per entry (packed
//! little-endian into `u64` words) and applied with add/sub only — at
//! batch-1 decode the projection is memory-bandwidth-bound and sign bits
//! make its weight reads near-free (E·D/8 bytes vs E·D·4), the natural
//! fit for the ternary/Bonsai direction. The `1/√E` scale is applied
//! once per output row.
//!
//! # Determinism
//!
//! Same `(D, E, seed)` → identical packed words → identical BLAKE3
//! commitment → bit-identical projections (pure add/sub/mul, no libm on
//! the apply path — the scale is computed once at construction with one
//! `sqrt`, an exactly-rounded IEEE op).

use blake3::Hasher;

/// Compile-time floor on the output dimension (the Issue 139 law).
const MIN_E: usize = 32;

/// Seeded Rademacher JL projector `R^D → R^E` with packed sign-bit storage.
///
/// `E < 32` fails to compile by construction (see module docs).
///
/// Storage is ONE packed `Vec<u64>` allocated at construction (the
/// offline half — `E · ceil(D/64)` words); the apply path is
/// slice-indexed over it and allocates nothing.
#[derive(Debug, Clone)]
pub struct JlProjector<const D: usize, const E: usize> {
    /// Packed sign bits, row-major: row `e` occupies
    /// `signs[e·words..(e+1)·words]`, bit `d` of row `e` is bit `d % 64`
    /// of word `signs[e·words + d/64]`.
    signs: Vec<u64>,
    /// `1/√E`, applied once per output row.
    scale: f32,
    commitment: [u8; 32],
}

impl<const D: usize, const E: usize> JlProjector<D, E> {
    /// Compile-time shape guard (an associated const so its `assert!`
    /// const-evaluates at monomorphization — the only stable pattern that
    /// both carries generics and hard-errors): `E >= 32` (the Issue 139
    /// law) and `D > 0`. Instantiating the projector with a violating shape
    /// is a COMPILE error, not a runtime surprise.
    const SHAPE_GUARD: () = assert!(
        E >= MIN_E && D > 0,
        "JlProjector requires E >= 32 (Issue 139) and D > 0"
    );

    /// Generate from `seed`. Deterministic: the packed sign words are the
    /// SplitMix64 stream read row-major.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let () = Self::SHAPE_GUARD;
        let words = D.div_ceil(64);
        let mut rng = SplitMix64::new(seed);
        let mut signs = vec![0u64; E * words];
        for w in signs.iter_mut() {
            *w = rng.next_u64();
        }
        let mut p = Self {
            signs,
            scale: 1.0 / (E as f32).sqrt(),
            commitment: [0u8; 32],
        };
        p.commit();
        p
    }

    /// Project `x` into `out` (add/sub only over the sign bits; the
    /// `1/√E` scale applied once per row). Zero-alloc.
    #[inline]
    pub fn project(&self, x: &[f32; D], out: &mut [f32; E]) {
        let words = D.div_ceil(64);
        for (e, out_e) in out.iter_mut().enumerate() {
            let row = &self.signs[e * words..(e + 1) * words];
            let mut acc = 0.0f32;
            let mut d = 0usize;
            for &word in row.iter() {
                // Full words except possibly the last (D may not be a
                // multiple of 64) — bounded by the d < D check.
                for b in 0..64 {
                    if d >= D {
                        break;
                    }
                    let v = x[d];
                    if (word >> b) & 1 == 1 {
                        acc += v;
                    } else {
                        acc -= v;
                    }
                    d += 1;
                }
            }
            *out_e = acc * self.scale;
        }
    }

    /// Project into a stack buffer (the zero-alloc convenience form).
    #[inline]
    #[must_use]
    pub fn project_array(&self, x: &[f32; D]) -> [f32; E] {
        let mut out = [0f32; E];
        self.project(x, &mut out);
        out
    }

    /// BLAKE3 commitment over `D || E || sign words` (LE u64, pinned
    /// order) — different shapes cannot collide.
    #[must_use]
    pub fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    /// Re-compute the commitment; `false` = tampered or drifted matrix.
    #[must_use]
    pub fn verify(&self) -> bool {
        self.compute_commitment() == self.commitment
    }

    fn commit(&mut self) {
        self.commitment = self.compute_commitment();
    }

    fn compute_commitment(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&(D as u64).to_le_bytes());
        hasher.update(&(E as u64).to_le_bytes());
        for &w in self.signs.iter() {
            hasher.update(&w.to_le_bytes());
        }
        let mut out = [0u8; 32];
        hasher.finalize_xof().fill(&mut out);
        out
    }
}

/// SplitMix64 — the deterministic stream (the codebook module's algorithm,
/// kept private there; duplicated here so neither feature implies the
/// other for a 20-line PRNG).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)` from the top 24 bits.
    #[inline]
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
}
