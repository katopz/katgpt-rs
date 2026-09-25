//! T1 arm (b) — mass-conserving perturbation for COCHAIN belief fields
//! (belief flows on 2D zone maps, d = 2 — the DEC-legal regime), plus the T6
//! codifferential-divergence escape signal.
//!
//! # The invariant, by construction
//!
//! The divergence-free edge cochains are `ker δ₁ = im δ₂ ⊕ H¹` — coexact ⊕
//! harmonic (the complement of the exact part in the Hodge decomposition).
//! This arm never projects (no Poisson solve, no CG tolerance): it BUILDS ε
//! inside that subspace —
//!
//! ```text
//! ε = δ₂ ψ + Σ_h c_h z_h
//! ```
//!
//! with `ψ` a BLAKE3-drawn face potential (coexact part, via the shipped
//! `codifferential_into`) and `z_h` caller-supplied integer cycles that close
//! (`δ₁ z_h = 0` is verified exactly at construction) for the harmonic part.
//! `δ₁δ₂ = 0` holds algebraically; to make it hold in **f32, bitwise**, every
//! `ψ_f` and `c_h` is rounded to a common power-of-two grid `g ≤ σ·2⁻¹⁵`, so
//! every partial sum the two boundary operators form is an integer multiple
//! of `g` well inside the 24-bit mantissa — no rounding ever happens, and
//! `belief_mass_divergence(ε) == 0.0` exactly (the Bench 898 identity arm).
//! The admission bound is checked at construction.
//!
//! # What this does NOT claim
//!
//! Mass conservation is an architectural invariant only — Research 590 §Caveats:
//! no evidence (GRAM's or ours) that belief failure modes involve mass drift.
//! Never read the identity arm as a quality result. The arm refuses the μ≠0
//! table term ([`Perturbation::admits_guidance`] is `false`): a table
//! direction is not divergence-free and would break the invariant — and for
//! the same reason it owns the diversity init and the escape kick
//! ([`Perturbation::owns_init`], [`Perturbation::admits_kick`]): neither a
//! Sobol offset nor `apply_kick`'s isotropic unit vector is in `ker δ₁`.

use crate::dec::{CellComplex, CochainField, codifferential_into};
use crate::diversity::temp::blake3_noise_fill;

use super::init::mix64;
use super::perturb::Perturbation;

/// Maximum harmonic generators carried by [`HarmonicCycles`].
pub const MAX_CYCLES: usize = 32;

/// Integer harmonic-class generators: closed edge cycles `z_h` with
/// `δ₁ z_h = 0` exactly (e.g. a loop around a hole of the zone map).
#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicCycles {
    /// Flattened `(edge, sign)` entries of every cycle.
    entries: Vec<(u32, i8)>,
    /// `offsets[h]..offsets[h+1]` indexes cycle `h` in `entries`.
    offsets: Vec<u32>,
    /// Max number of cycles through one edge (admission bound input).
    max_per_edge: usize,
}

impl HarmonicCycles {
    /// Validate `cycles` against `cx`: every entry names an existing edge
    /// with sign ±1, and each cycle is closed (`δ₁ z = 0` over the integers).
    /// Returns `None` otherwise, or when there are more than [`MAX_CYCLES`].
    pub fn new(cx: &CellComplex, cycles: &[&[(usize, i8)]]) -> Option<Self> {
        if cycles.len() > MAX_CYCLES {
            return None;
        }
        let n_e = cx.n_edges();
        let n_v = cx.n_vertices();
        let mut coef = vec![0i32; n_e];
        let mut per_edge = vec![0usize; n_e];
        let mut div = vec![0i64; n_v];
        let mut entries = Vec::new();
        let mut offsets = vec![0u32];
        for cyc in cycles {
            coef.fill(0);
            for &(e, s) in cyc.iter() {
                if e >= n_e || !(s == 1 || s == -1) {
                    return None;
                }
                coef[e] += i32::from(s);
                per_edge[e] += 1;
                entries.push((e as u32, s));
            }
            div.fill(0);
            for &(v, e, s) in cx.boundary_entries(0) {
                div[v] += i64::from(s) * i64::from(coef[e]);
            }
            if div.iter().any(|&x| x != 0) {
                return None;
            }
            offsets.push(entries.len() as u32);
        }
        Some(Self {
            entries,
            offsets,
            max_per_edge: per_edge.into_iter().max().unwrap_or(0),
        })
    }

    /// Number of cycles.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// `true` when there are no cycles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Largest power of two `≤ x` for finite positive `x` (exact, bit-level).
#[inline]
fn pow2_floor(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7F80_0000)
}

/// Arm (b): mass-conserving ε for a rank-1 (edge) cochain field.
///
/// `h` passed to [`Perturbation::perturb`] is the edge-cochain data
/// (`cx.n_edges()` scalars). Scratch is owned (allocated once in
/// [`Self::new`]); the hot path is allocation-free.
pub struct MassConserving<'a> {
    cx: &'a CellComplex,
    cycles: Option<&'a HarmonicCycles>,
    face: CochainField,
    flow: CochainField,
}

/// Grid headroom: `ψ, c ∈ g·ℤ` with `|ψ|, |c| ≤ σ ≤ 2^e`, `g = 2^(e−GRID_BITS)`
/// ⇒ integers ≤ 2^GRID_BITS; the admission bound keeps every partial sum
/// under 2^24.
const GRID_BITS: i32 = 15;

impl<'a> MassConserving<'a> {
    /// Build over `cx` with optional harmonic generators. Returns `None` when
    /// the exactness admission bound fails:
    /// `max_vertex_degree · (max_faces_per_edge + max_cycles_per_edge) ≤ 2^(24−GRID_BITS−1)`.
    pub fn new(cx: &'a CellComplex, cycles: Option<&'a HarmonicCycles>) -> Option<Self> {
        let n_v = cx.n_vertices();
        let n_e = cx.n_edges();
        let mut deg = vec![0usize; n_v];
        for &(v, _, _) in cx.boundary_entries(0) {
            deg[v] += 1;
        }
        let mut faces_per_edge = vec![0usize; n_e];
        if cx.n_faces() > 0 {
            for &(e, _, _) in cx.boundary_entries(1) {
                faces_per_edge[e] += 1;
            }
        }
        let max_deg = deg.into_iter().max().unwrap_or(0);
        let max_fpe = faces_per_edge.into_iter().max().unwrap_or(0);
        let max_cpe = cycles.map_or(0, |c| c.max_per_edge);
        if max_deg * (max_fpe + max_cpe) > (1usize << (24 - GRID_BITS - 1)) {
            return None;
        }
        Some(Self {
            cx,
            cycles,
            face: CochainField::zeros(2, cx.n_faces(), 1),
            flow: CochainField::zeros(1, n_e, 1),
        })
    }

    /// Write the mass-conserving draw for `(seed, sigma)` into `out` (len
    /// `n_edges`), overwriting it. `sigma ≤ 0` / non-finite ⇒ all zeros.
    /// `δ₁(out) == 0` exactly.
    pub fn draw_into(&mut self, seed: u64, sigma: f32, out: &mut [f32]) {
        let n_e = self.cx.n_edges();
        let out = &mut out[..n_e];
        if !(sigma.is_finite() && sigma > 0.0) {
            out.fill(0.0);
            return;
        }
        // Common grid g = 2^(e − GRID_BITS), 2^e the power of two ≥ σ.
        let g = pow2_floor(sigma) * 2.0 / (1u32 << GRID_BITS) as f32;
        let q = |x: f32| (x / g).round() * g; // exact: g is a power of two
        // Coexact part: δ₂ψ.
        let face = &mut self.face.data;
        blake3_noise_fill(seed, sigma, face);
        for x in face.iter_mut() {
            *x = q(*x);
        }
        if !face.is_empty() {
            codifferential_into(self.cx, &self.face, &mut self.flow);
        } else {
            self.flow.data.fill(0.0);
        }
        out.copy_from_slice(&self.flow.data[..n_e]);
        // Harmonic part: Σ c_h z_h.
        if let Some(cyc) = self.cycles {
            let mut c = [0.0f32; MAX_CYCLES];
            let nc = cyc.len();
            blake3_noise_fill(mix64(seed ^ 0x4A12_B0C5), sigma, &mut c[..nc]);
            for (h, &ch_raw) in c[..nc].iter().enumerate() {
                let ch = q(ch_raw);
                let (a, b) = (cyc.offsets[h] as usize, cyc.offsets[h + 1] as usize);
                for &(e, s) in &cyc.entries[a..b] {
                    out[e as usize] += f32::from(s) * ch;
                }
            }
        }
    }
}

impl Perturbation for MassConserving<'_> {
    fn perturb(&mut self, h: &mut [f32], _delta: &[f32], noise: &mut [f32], seed: u64, sigma: f32) {
        if !(sigma.is_finite() && sigma > 0.0) {
            return;
        }
        self.draw_into(seed, sigma, noise);
        for (x, &e) in h.iter_mut().zip(noise.iter()) {
            *x += e;
        }
    }

    fn admits_guidance(&self) -> bool {
        false
    }

    fn owns_init(&self) -> bool {
        true
    }

    fn admits_kick(&self) -> bool {
        false
    }
}

/// T6 new claim (ii) — codifferential-divergence escape signal over an edge
/// update `Δflow`: the CIRCULATION fraction
///
/// ```text
/// s = 1 − ‖δ₁ Δ‖₁ / (2 ‖Δ‖₁)      ∈ [0, 1]
/// ```
///
/// (`‖δ₁Δ‖₁ ≤ 2‖Δ‖₁` since each edge touches two vertices). `s ≈ 1`: the
/// update churns belief round loops without moving mass — the trap-shaped
/// signature to confirm an escape on; `s ≈ 0`: mass is being redistributed
/// (progress). A zero update returns `NaN` (no evidence — never confirms).
/// Zero-allocation with owned scratch; plug in as `Hooks::probe`.
pub struct DivergenceProbe<'a> {
    cx: &'a CellComplex,
    flow: CochainField,
    div: CochainField,
}

impl<'a> DivergenceProbe<'a> {
    /// Probe over `cx` (allocates its two cochain buffers once).
    pub fn new(cx: &'a CellComplex) -> Self {
        Self {
            cx,
            flow: CochainField::zeros(1, cx.n_edges(), 1),
            div: CochainField::zeros(0, cx.n_vertices(), 1),
        }
    }

    /// Circulation fraction of `delta` (len `n_edges`).
    pub fn circulation(&mut self, delta: &[f32]) -> f32 {
        let n_e = self.cx.n_edges();
        self.flow.data.copy_from_slice(&delta[..n_e]);
        let l1: f32 = self.flow.data.iter().map(|x| x.abs()).sum();
        if !(l1.is_finite() && l1 > 0.0) {
            return f32::NAN;
        }
        codifferential_into(self.cx, &self.flow, &mut self.div);
        let dl1: f32 = self.div.data.iter().map(|x| x.abs()).sum();
        1.0 - dl1 / (2.0 * l1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec::belief_mass_divergence;

    #[test]
    fn coexact_draw_is_exactly_divergence_free_on_a_grid() {
        let cx = CellComplex::grid_2d(6, 5);
        let mut arm = MassConserving::new(&cx, None).unwrap();
        let mut out = vec![0.0f32; cx.n_edges()];
        for seed in 0..32u64 {
            for sigma in [1e-3f32, 0.1, 0.37, 1.0, 3.0] {
                arm.draw_into(seed, sigma, &mut out);
                assert!(out.iter().any(|&x| x != 0.0));
                let f = CochainField::from_vec(1, 1, out.clone());
                assert_eq!(belief_mass_divergence(&cx, &f), 0.0, "seed {seed} σ {sigma}");
            }
        }
    }

    #[test]
    fn harmonic_cycle_draw_is_exactly_divergence_free_on_a_ring() {
        // A 6-cycle graph with no faces: im δ₂ = 0, ker δ₁ = H¹ = span(ring).
        let edges: Vec<(usize, usize)> = (0..6).map(|i| (i, (i + 1) % 6)).collect();
        let cx = CellComplex::from_edges(6, &edges);
        let ring: Vec<(usize, i8)> = (0..6).map(|e| (e, 1)).collect();
        let cyc = HarmonicCycles::new(&cx, &[&ring]).unwrap();
        let mut arm = MassConserving::new(&cx, Some(&cyc)).unwrap();
        let mut out = vec![0.0f32; 6];
        for seed in 0..32u64 {
            arm.draw_into(seed, 0.5, &mut out);
            let f = CochainField::from_vec(1, 1, out.clone());
            assert_eq!(belief_mass_divergence(&cx, &f), 0.0);
            assert!(out.iter().all(|&x| x == out[0]), "a ring flow is constant");
        }
        // An open path is refused as a harmonic generator.
        let open: Vec<(usize, i8)> = (0..5).map(|e| (e, 1)).collect();
        assert!(HarmonicCycles::new(&cx, &[&open]).is_none());
    }

    #[test]
    fn arm_refuses_guidance_and_zero_sigma_is_a_noop() {
        let cx = CellComplex::grid_2d(3, 3);
        let mut arm = MassConserving::new(&cx, None).unwrap();
        assert!(!arm.admits_guidance());
        let mut h = vec![0.25f32; cx.n_edges()];
        let mut buf = vec![0.0f32; cx.n_edges()];
        arm.perturb(&mut h, &[], &mut buf, 1, 0.0);
        assert!(h.iter().all(|&x| x == 0.25));
    }

    #[test]
    fn circulation_probe_separates_churn_from_transport() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut arm = MassConserving::new(&cx, None).unwrap();
        let mut probe = DivergenceProbe::new(&cx);
        let mut churn = vec![0.0f32; cx.n_edges()];
        arm.draw_into(5, 0.3, &mut churn);
        assert_eq!(probe.circulation(&churn), 1.0, "pure circulation");
        let mut transport = vec![0.0f32; cx.n_edges()];
        transport[0] = 1.0; // one edge: a source and a sink
        assert!(probe.circulation(&transport) < 1e-6);
        assert!(probe.circulation(&vec![0.0; cx.n_edges()]).is_nan());
    }
}
