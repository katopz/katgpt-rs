//! Shared deterministic fixtures for the Plan-638 `cond_geometry` gates.
//!
//! House pattern (the `sst_certified_g4_alloc` fixture discipline): a
//! local seeded LCG + Box–Muller — no global RNG (the unseeded-global-RNG
//! gate), identical seed ⇒ bit-identical populations. This module is not
//! a test target; it is included by the gated gate files via `#[path]`.
//!
//! `allow(dead_code)`: each gate binary includes the whole fixture and
//! uses the slice it needs (the `subspace_intervention` shared-fixture
//! discipline) — the unused remainder is not dead, it is per-binary
//! reach.
#![allow(dead_code)]

/// Deterministic f64/f32 stream — the LCG constants are the
/// `sst_certified_g4_alloc` house pair.
pub struct Lcg(u64);

impl Lcg {
    /// New stream from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// One N(0,1) draw (Box–Muller; deterministic under the stream).
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = core::f64::consts::TAU * u2;
        (r * theta.cos()) as f32
    }

    /// One uniform draw in (0, 1] — the radius draw for the ball sampler.
    pub fn next_unit(&mut self) -> f64 {
        self.next_f64().max(1e-12)
    }
}

const MAX_DIM: usize = 64;

fn gauss_direction(rng: &mut Lcg, d: usize) -> [f64; MAX_DIM] {
    debug_assert!((1..=MAX_DIM).contains(&d));
    loop {
        let mut dir = [0.0f64; MAX_DIM];
        let mut norm2 = 0.0f64;
        for v in dir.iter_mut().take(d) {
            *v = rng.next_gauss() as f64;
            norm2 += *v * *v;
        }
        if norm2 > 1e-24 {
            let norm = norm2.sqrt();
            for v in dir.iter_mut().take(d) {
                *v /= norm;
            }
            return dir;
        }
        // Astronomically unlikely degenerate draw — redraw (the stream
        // advances, so the result stays deterministic per seed).
    }
}

/// `n` uniform samples in the d-ball of radius 1, embedded in `big_d`
/// (direction on S^(d−1) × radius u^(1/d), u ∈ (0,1] — the exact uniform
/// ball law; coordinates past `d` are zero).
///
/// # Panics
///
/// Asserts `1 ≤ d ≤ big_d ≤ 64` and `out.len() == n·big_d`.
pub fn uniform_in_dball_into(out: &mut [f32], n: usize, d: usize, big_d: usize, rng: &mut Lcg) {
    assert!(d >= 1 && d <= big_d && big_d <= MAX_DIM);
    assert_eq!(out.len(), n * big_d);
    for i in 0..n {
        let dir = gauss_direction(rng, d);
        let r = rng.next_unit().powf(1.0 / d as f64);
        let row = &mut out[i * big_d..(i + 1) * big_d];
        for (c, v) in row.iter_mut().enumerate() {
            *v = if c < d { (r * dir[c]) as f32 } else { 0.0 };
        }
    }
}

/// `n` uniform samples ON the unit sphere S^(d−1), embedded in `big_d`
/// (the L2N2 bias-class arm; coordinates past `d` are zero).
///
/// # Panics
///
/// Asserts `1 ≤ d ≤ big_d ≤ 64` and `out.len() == n·big_d`.
pub fn sphere_into(out: &mut [f32], n: usize, d: usize, big_d: usize, rng: &mut Lcg) {
    assert!(d >= 1 && d <= big_d && big_d <= MAX_DIM);
    assert_eq!(out.len(), n * big_d);
    for i in 0..n {
        let dir = gauss_direction(rng, d);
        let row = &mut out[i * big_d..(i + 1) * big_d];
        for (c, v) in row.iter_mut().enumerate() {
            *v = if c < d { dir[c] as f32 } else { 0.0 };
        }
    }
}
