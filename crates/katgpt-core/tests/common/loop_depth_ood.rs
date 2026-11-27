//! `loop_depth_ood` — the Issue-1248 constructed fixture (T1) + its G0 span
//! census + fault-signature/R2/R3 instruments (T1–T3).
//!
//! A two-layer looped forward whose passes 1..d converge (direction
//! contraction under rmsnorm, alignment axis e₁) with ONE pass-indexed gate
//! module between the layers: healthy for the seen pass band `φ = k/d`
//! (ramp; sinusoid fallback spelling), collapsed at `φ > 1` — the Inkling
//! fault class (a seen direction whose influence flips) reproduced at a
//! LOCALIZED site (layer 2, the OOD-consuming layer). The gate write is the
//! pass's payload; its collapse reverses the loop's steady shift direction —
//! the source's `v₁` sign-flip trajectory.
//
// Shared by several test binaries (g0, r2_r3, and T4–T6 arms later): each
// binary sees the whole surface, so unused-for-that-binary items are
// expected — the scoped allow is the include shape, not a lint escape.
#![allow(dead_code)]
//!
//! The design is pre-registered in `.issues/1248_closed_form_loop_depth_extension.md`
//! §"T1 fixture design" (revised at implementation — revision note there).
//! All bars measured green 2026-11-27; the measured numbers live in the
//! issue. std-only: no model, no GPU, no lib-internal access.
//!
//! Pure constructed math — std-only, no model, no GPU. State math in f32;
//! census/eigen math in f64 with sequential folds (Issue-940 hygiene; a local
//! instrument, never a sync crossing).

// ---------------------------------------------------------------------------
// Deterministic RNG — the house SplitMix64 (self-contained copy, the
// established per-module convention; Box–Muller normals for fixtures).
// ---------------------------------------------------------------------------

pub struct SplitMix64 {
    state: u64,
    pending: Option<f64>,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            pending: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller (pairs consumed two at a time).
    pub fn next_normal(&mut self) -> f64 {
        if let Some(p) = self.pending.take() {
            return p;
        }
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.pending = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// Fixture constants — the pre-registered parameters (issue §"T1 fixture
// design"). One master seed; every derived quantity is seeded from it.
// ---------------------------------------------------------------------------

/// Master seed — the issue number. Changing it re-randomizes the fixture
/// geometry; the bars below are seed-tolerant by construction but the pin is
/// still recorded (determinism law: same seed, bit-identical fixture).
pub const MASTER_SEED: u64 = 1248;

pub const D: usize = 64; // state dimension
pub const D_GATE: usize = D + 1; // gate input = [h1; φ]
pub const D_TRAIN: usize = 3; // trained depth d
pub const K_LADDER: usize = 6; // ladder length (= d+3, the extrapolation bar)
pub const BATCH: usize = 12; // seeded starts (the prompt-batch analog)

const L1_ALIGN: f32 = 0.85; // layer-1 mode on the alignment axis e₁
const L1_OFF: f32 = 0.008; // layer-1 off-mode — KILLS the state's off-content
                           // each pass, so at the fault the layer-2 input collapses onto e₁ (0.03 was
                           // not enough: y's 80%-off share × 0.03 still left ~25% off in the fault mix)
const L2_M1: f32 = 0.3; // layer-2 mode on e₁ — small, so the fault output
                        // stays e₁-pure while the healthy output rides the write
const L2_M2: f32 = 0.9; // layer-2 mode on e₂
const L2_M3: f32 = 0.9; // layer-2 mode on e₃
const L2_OFF: f32 = 0.1; // layer-2 off-mode
                         // The gate write direction ŵ = normalize(w1·e₁ + w2·e₂ + w3·e₃) — positive
                         // on all three axes (the +e₁ weight is what puts ⟨e₁, A2ŵ⟩ > 0, the flip's
                         // sign), dominated by e₂/e₃ so its collapse swings the layer-2 output mix.
const WRITE_W1: f64 = 0.30;
const WRITE_W2: f64 = 0.55;
const WRITE_W3: f64 = 0.77;
const W_AMP: f32 = 40.0; // write amplitude — the write dominates the healthy
                         // h1g mix (~25 vs ~7); only the gate's collapse RATIO (κ) separates the fault
const GATE_PHI_GAIN: f32 = 0.3; // in-band drift term (the 0.566→0.638 shape)
const GATE_KAPPA: f32 = 26.0; // hinge steepness — κ/3 = 8.67 at φ = 4/3:
                              // g drops 0.70 → ≈ 0.0002 (a 3500× collapse — with W_AMP = 40 even the fault's
                              // residual write must LOSE to the state content, or the fault mix stays
                              // write-flavored; the ratio, not the amplitude, is the discriminator)
const WARM_PASSES: usize = 8; // deep-warm passes to the fixed point
const ALIGN_DEFICIT: f32 = 0.15; // shared e₁ deficit left in the warm state —
                                 // the healthy ladder's pull-back shifts (decaying, shared sign across rows)
const ROW_NOISE: f32 = 0.005; // per-row warm noise — keeps rows distinct
                              // without fattening the keep span

/// The pass-indicator encoding (verdict round-2 pin 1: CONTINUOUS, never
/// one-hot). Ramp = primary; Sinusoid = the compiled fallback spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassIndicator {
    Ramp,
    Sinusoid,
}

// ---------------------------------------------------------------------------
// Small linear-algebra helpers (f64, self-contained — the lib's Cholesky and
// Jacobi eigensolver are private module fns; test fixtures carry their own).
// ---------------------------------------------------------------------------

/// Cholesky factor of a symmetric positive-definite `n×n` matrix (row-major).
/// Panics loud on a non-SPD diagonal — the F9 conditioning refusal.
fn cholesky_lower(m: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut acc = m[i * n + j];
            for k in 0..j {
                acc -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(acc > 1e-12, "census matrix not SPD (diag {i} = {acc})");
                l[i * n + i] = acc.sqrt();
            } else {
                l[i * n + j] = acc / l[j * n + j];
            }
        }
    }
    l
}

/// Solve `A x = b` for SPD `A` via the Cholesky factor (two triangular solves).
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let l = cholesky_lower(a, n);
    // Forward: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[i * n + k] * y[k];
        }
        y[i] = acc / l[i * n + i];
    }
    // Backward: Lᵀ x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[k * n + i] * x[k];
        }
        x[i] = acc / l[i * n + i];
    }
    x
}

/// Cyclic Jacobi eigendecomposition of a symmetric `n×n` matrix (row-major,
/// destroyed in place). Returns the eigenvalues in DIAGONAL ORDER — index i
/// corresponds to eigenvector column i of `v_out` (NOT sorted; sort pairs at
/// the call site when an ordering is needed).
pub fn jacobi_symmetric(m: &mut [f64], v_out: &mut [f64], n: usize) -> Vec<f64> {
    for i in 0..n {
        for j in 0..n {
            v_out[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _ in 0..60 {
        // Largest off-diagonal magnitude.
        let mut off = 0.0f64;
        let (mut p, mut q) = (0usize, 1usize);
        for i in 0..n {
            for j in (i + 1)..n {
                let a = m[i * n + j].abs();
                if a > off {
                    off = a;
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-12 {
            break;
        }
        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // Rotate rows/columns p,q of m and the eigenvector matrix.
        for k in 0..n {
            let mkp = m[k * n + p];
            let mkq = m[k * n + q];
            m[k * n + p] = c * mkp - s * mkq;
            m[k * n + q] = s * mkp + c * mkq;
        }
        for k in 0..n {
            let mpk = m[p * n + k];
            let mqk = m[q * n + k];
            m[p * n + k] = c * mpk - s * mqk;
            m[q * n + k] = s * mpk + c * mqk;
        }
        for k in 0..n {
            let vkp = v_out[k * n + p];
            let vkq = v_out[k * n + q];
            v_out[k * n + p] = c * vkp - s * vkq;
            v_out[k * n + q] = s * vkp + c * vkq;
        }
    }
    let eigs: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    eigs
}

/// Gram–Schmidt orthonormalization of `rows` in place.
fn gram_schmidt(rows: &mut [Vec<f64>]) {
    for i in 0..rows.len() {
        let (prev, cur) = rows.split_at_mut(i);
        let cur = &mut cur[0];
        for pj in prev.iter() {
            let dot: f64 = cur.iter().zip(pj).map(|(a, b)| a * b).sum();
            for (a, b) in cur.iter_mut().zip(pj) {
                *a -= dot * b;
            }
        }
        let nrm = cur.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(nrm > 1e-8, "gram-schmidt: degenerate row {i}");
        for v in cur.iter_mut() {
            *v /= nrm;
        }
    }
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

/// The fixture's frozen geometry: the three seeded axes, the write direction,
/// the gate weights and the layer-2 eigenbasis. Built once from
/// [`MASTER_SEED`]; every field derives deterministically.
pub struct LoopDepthOod {
    /// The alignment axis e₁ (unit, seeded) — layer 1's mode; the healthy
    /// ladder's pull-back shifts run along it.
    pub e1: Vec<f32>,
    /// Second seeded axis (unit, ⊥ e₁) — carries write content + a layer-2
    /// mode.
    pub e2: Vec<f32>,
    /// Third seeded axis (unit, ⊥ e₁,e₂) — carries write content + a layer-2
    /// mode.
    pub e3: Vec<f32>,
    /// The gate write direction ŵ (unit) = normalize(w1·e₁ + w2·e₂ + w3·e₃).
    pub w_dir: Vec<f32>,
    /// Layer-2 eigenbasis (orthonormal rows; slots 0..3 = e₁, e₂, e₃).
    pub q2: Vec<Vec<f32>>,
    /// Gate input weights over [h1; φ] (last entry = φ weight).
    pub w: Vec<f32>,
    /// Gate bias.
    pub c: f32,
}

impl LoopDepthOod {
    /// Build the frozen geometry from the master seed.
    pub fn build() -> Self {
        let mut rng = SplitMix64::new(MASTER_SEED);
        // Seeded normal rows → Gram–Schmidt → the three axes + the layer-2
        // eigenbasis (slots 0..3 = e₁, e₂, e₃; the rest off-modes).
        let mut rows: Vec<Vec<f64>> = (0..D)
            .map(|_| (0..D).map(|_| rng.next_normal()).collect())
            .collect();
        gram_schmidt(&mut rows);
        let e1: Vec<f32> = rows[0].iter().map(|&v| v as f32).collect();
        let e2: Vec<f32> = rows[1].iter().map(|&v| v as f32).collect();
        let e3: Vec<f32> = rows[2].iter().map(|&v| v as f32).collect();
        let q2: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| r.iter().map(|&v| v as f32).collect())
            .collect();
        // The write direction: a positive mix of all three axes.
        let mut wd = vec![0.0f64; D];
        for i in 0..D {
            wd[i] = WRITE_W1 * rows[0][i] + WRITE_W2 * rows[1][i] + WRITE_W3 * rows[2][i];
        }
        let nrm: f64 = wd.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut wd {
            *v /= nrm;
        }
        let w_dir: Vec<f32> = wd.iter().map(|&v| v as f32).collect();

        // Gate weights: small seeded w_h, the φ gain, and the bias set so the
        // in-band gate sits ≈ 0.6 (the source's healthy band).
        let w_h: Vec<f32> = (0..D).map(|_| (rng.next_normal() as f32) * 0.02).collect();
        let mut w = w_h;
        w.push(GATE_PHI_GAIN);
        let c = 0.58f32;

        Self {
            e1,
            e2,
            e3,
            w_dir,
            q2,
            w,
            c,
        }
    }

    /// RMSNorm (gain-free): `x / rms(x)`, rms over the slice.
    pub fn rmsnorm(x: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms.max(1e-12).sqrt()) as f32;
        x.iter().map(|&v| v * inv).collect()
    }

    /// Layer 1: `h1 = A1·rmsnorm(x)`, `A1 = align·e₁e₁ᵀ + off·(I − e₁e₁ᵀ)` —
    /// the alignment axis' mode pulls the state's direction toward e₁.
    pub fn layer1(&self, x: &[f32]) -> Vec<f32> {
        let nx = Self::rmsnorm(x);
        let dot: f32 = nx.iter().zip(&self.e1).map(|(a, b)| a * b).sum();
        nx.iter()
            .zip(&self.e1)
            .map(|(&n, &ui)| L1_ALIGN * dot * ui + L1_OFF * (n - dot * ui))
            .collect()
    }

    /// Pass indicator — the PINNED continuous encoding: scalar ramp `k/d`.
    /// (Sinusoid fallback spelling provided separately; never one-hot.)
    pub fn pass_indicator_ramp(k: usize) -> f32 {
        k as f32 / D_TRAIN as f32
    }

    /// Sinusoid fallback spelling of the same indicator (not the GOAT arm's
    /// encoding). Seen band for d = 3: [sin(π/4), 1] = [0.707, 1.0].
    pub fn pass_indicator_sinusoid(k: usize) -> f32 {
        (std::f64::consts::PI * (k as f64) / ((D_TRAIN + 1) as f64)).sin() as f32
    }

    /// Lower edge of the seen band for an encoding.
    pub fn pass_band_lo(indicator: PassIndicator) -> f32 {
        match indicator {
            PassIndicator::Ramp => 1.0 / D_TRAIN as f32,
            PassIndicator::Sinusoid => Self::pass_indicator_sinusoid(1),
        }
    }

    /// Hinge penalty past the seen band — ramp: `κ·relu(φ−1)`; sinusoid:
    /// `κ·(relu(band_lo − φ) + relu(φ − 1))` (both edges of the band).
    pub fn band_hinge(phi: f32, indicator: PassIndicator) -> f32 {
        let lo = Self::pass_band_lo(indicator);
        GATE_KAPPA * (phi - 1.0).max(0.0)
            + match indicator {
                PassIndicator::Ramp => 0.0,
                PassIndicator::Sinusoid => GATE_KAPPA * (lo - phi).max(0.0),
            }
    }

    /// Gate pre-activation at input `[h1; φ]` — in-band mild drift, band
    /// hinge penalty past the seen band (the constructed fault).
    pub fn gate_preact_enc(&self, h1: &[f32], phi: f32, indicator: PassIndicator) -> f32 {
        let zh: f32 = h1.iter().zip(&self.w).map(|(a, b)| a * b).sum();
        zh + self.w[D] * phi + self.c - Self::band_hinge(phi, indicator)
    }

    /// Gate pre-activation on the primary (ramp) encoding.
    pub fn gate_preact(&self, h1: &[f32], phi: f32) -> f32 {
        self.gate_preact_enc(h1, phi, PassIndicator::Ramp)
    }

    /// Gate output and the gated state `h1g = h1 + g·W_AMP·ŵ` for an
    /// encoding — the write is the pass's payload (the source's memory-gate
    /// analog) and its collapse is the fault.
    pub fn gate_apply_enc(
        &self,
        h1: &[f32],
        phi: f32,
        indicator: PassIndicator,
    ) -> (f32, Vec<f32>) {
        let z = self.gate_preact_enc(h1, phi, indicator);
        let g = 1.0 / (1.0 + (-z as f64).exp()) as f32;
        let mut h1g = h1.to_vec();
        let amp = g * W_AMP;
        for (v, &wi) in h1g.iter_mut().zip(&self.w_dir) {
            *v += amp * wi;
        }
        (g, h1g)
    }

    /// Gate output and gated state on the primary (ramp) encoding.
    pub fn gate_apply(&self, h1: &[f32], phi: f32) -> (f32, Vec<f32>) {
        self.gate_apply_enc(h1, phi, PassIndicator::Ramp)
    }

    /// Layer 2: `y = Q2ᵀ diag(μ) Q2 rmsnorm(h1g)` — three carried modes
    /// (slots 0..3 = e₁, e₂, e₃), off-modes elsewhere.
    pub fn layer2(&self, h1g: &[f32]) -> Vec<f32> {
        let nv = Self::rmsnorm(h1g);
        // Project onto eigenvectors, scale, reconstruct.
        let mut coeffs = vec![0.0f32; D];
        for (i, qi) in self.q2.iter().enumerate() {
            let mu = match i {
                0 => L2_M1,
                1 => L2_M2,
                2 => L2_M3,
                _ => L2_OFF,
            };
            coeffs[i] = mu * nv.iter().zip(qi).map(|(a, b)| a * b).sum::<f32>();
        }
        let mut y = vec![0.0f32; D];
        for (i, qi) in self.q2.iter().enumerate() {
            for (j, &qij) in qi.iter().enumerate() {
                y[j] += coeffs[i] * qij;
            }
        }
        y
    }

    /// One pass of the loop at pass index `k` (1-based): layer 1 → gate →
    /// layer 2. Returns `(g, y)` — the gate telemetry and the pass output.
    pub fn pass(&self, x: &[f32], k: usize) -> (f32, Vec<f32>) {
        let phi = Self::pass_indicator_ramp(k);
        let h1 = self.layer1(x);
        let (_g, h1g) = self.gate_apply(&h1, phi);
        let y = self.layer2(&h1g);
        (_g, y)
    }

    /// Run the ladder for one start state: `K_LADDER` passes. Returns the
    /// per-pass gate values and outputs (index 0 = pass 1).
    pub fn ladder(&self, x0: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut gates = Vec::with_capacity(K_LADDER);
        let mut outs = Vec::with_capacity(K_LADDER);
        let mut x = x0.to_vec();
        for k in 1..=K_LADDER {
            let (g, y) = self.pass(&x, k);
            gates.push(g);
            x = y.clone();
            outs.push(y);
        }
        (gates, outs)
    }

    /// A seeded start state (unit-scale normal draw — the "prompt" analog).
    /// `usize::MAX - 1` is the CANONICAL basin seed (used by `warm_start`).
    pub fn start_state(&self, idx: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(MASTER_SEED ^ 0x5EED_0000_u64.wrapping_add(idx as u64));
        (0..D).map(|_| (rng.next_normal() as f32) * 0.5).collect()
    }

    /// A WARM start: the canonical basin point (the loop run to its fixed
    /// point from a canonical seed — row-independent), then a SHARED e₁
    /// deficit plus small per-row noise. Real serving never feeds a trained
    /// loop white noise; and the shared deficit is what gives the healthy
    /// ladder its pull-back shifts — decaying, one sign, every row — while
    /// the tiny noise keeps the batch rows distinct without fattening the
    /// keep span (the G0 margin lives in the thin span).
    pub fn warm_start(&self, idx: usize) -> Vec<f32> {
        let mut s = self.start_state(usize::MAX - 1);
        for _ in 0..WARM_PASSES {
            let h1 = self.layer1(&s);
            let (_g, h1g) = self.gate_apply(&h1, Self::pass_indicator_ramp(D_TRAIN));
            s = self.layer2(&h1g);
        }
        // Shared alignment EXCESS: displaced +ALIGN_DEFICIT along e₁, the
        // loop pulls the state back toward the fixed point (−e₁ shifts —
        // OPPOSITE the fault's +e₁-ward collapse shift, which is what makes
        // the shift-SVD read the source's sign-flip trajectory).
        for (v, &ei) in s.iter_mut().zip(&self.e1) {
            *v += ALIGN_DEFICIT * ei;
        }
        // Per-row noise (tiny).
        let mut rng = SplitMix64::new(MASTER_SEED ^ (0x0FF5_0000 + idx as u64));
        for v in s.iter_mut() {
            *v += (rng.next_normal() as f32) * ROW_NOISE;
        }
        s
    }
}

// ---------------------------------------------------------------------------
// G0 span census (the bar that runs before any arm)
// ---------------------------------------------------------------------------

/// One row of the per-row overlap census.
pub struct OverlapRow {
    pub row: usize,
    pub overlap: f64,
}

/// The G0 span census report — everything is REPORTED (the overlap number
/// ships beside every result; verdict round-2 pin 1).
pub struct G0Census {
    /// Per-edit-row overlap with span(X_keep): ‖P x‖ / ‖x‖, P the ridge
    /// projector `Xᵀ(X Xᵀ + εI)⁻¹X` computed through the row Gram.
    pub overlaps: Vec<OverlapRow>,
    /// Mean overlap across edit rows.
    pub overlap_mean: f64,
    /// Min overlap across edit rows.
    pub overlap_min: f64,
    /// Max overlap across edit rows (the binding margin against 1).
    pub overlap_max: f64,
    /// Numerical rank of the keep Gram (eigenvalues above the floor).
    pub keep_rank: usize,
    /// Smallest nonzero eigenvalue of the keep Gram.
    pub keep_min_nonzero_eig: f64,
    /// Numerical rank of the joint (keep + edit) Gram.
    pub joint_rank: usize,
    /// Smallest nonzero eigenvalue of the joint Gram.
    pub joint_min_nonzero_eig: f64,
}

/// Compute the G0 span census. `keep` = pass-1..d gate-module input rows
/// (flattened row-major), `edit` = pass-(d+1) input rows, `dim` = row width.
pub fn g0_span_census(keep: &[f32], edit: &[f32], dim: usize) -> G0Census {
    let n_keep = keep.len() / dim;
    let n_edit = edit.len() / dim;
    // Row Gram of the keep set (f64, sequential folds).
    let mut g = vec![0.0f64; n_keep * n_keep];
    for i in 0..n_keep {
        for j in 0..n_keep {
            let mut acc = 0.0f64;
            for t in 0..dim {
                acc += (keep[i * dim + t] as f64) * (keep[j * dim + t] as f64);
            }
            g[i * n_keep + j] = acc;
        }
    }
    // Eigendecompose the keep Gram (pairs sorted ascending for the rank/
    // min-nonzero census; columns of `vecs` stay aligned with `eigs[col]`).
    let mut gm = g.clone();
    let mut vecs = vec![0.0f64; n_keep * n_keep];
    let eigs_unsorted = jacobi_symmetric(&mut gm, &mut vecs, n_keep);
    let mut pairs: Vec<(f64, usize)> = eigs_unsorted.iter().copied().zip(0..n_keep).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let floor = 1e-10
        * eigs_unsorted
            .iter()
            .fold(0.0f64, |m, &v| m.max(v.abs()))
            .max(1.0);
    let keep_rank = pairs.iter().filter(|(v, _)| *v > floor).count();
    let keep_min_nonzero_eig = pairs
        .iter()
        .find(|(v, _)| *v > floor)
        .map(|(v, _)| *v)
        .unwrap_or(0.0);

    // Ridge projection through the row Gram: P x = Kᵀ (K Kᵀ + εI)⁻¹ K x.
    let eps = 1e-8;
    let mut a = g.clone();
    for i in 0..n_keep {
        a[i * n_keep + i] += eps;
    }
    let mut overlaps = Vec::with_capacity(n_edit);
    for r in 0..n_edit {
        // K x (n_keep)
        let mut kx = vec![0.0f64; n_keep];
        for i in 0..n_keep {
            let mut acc = 0.0f64;
            for t in 0..dim {
                acc += (keep[i * dim + t] as f64) * (edit[r * dim + t] as f64);
            }
            kx[i] = acc;
        }
        let alpha = solve_spd(&a, &kx, n_keep);
        // ‖P x‖² = αᵀ (K Kᵀ) α ; ‖x‖² direct.
        let mut px2 = 0.0f64;
        for i in 0..n_keep {
            for j in 0..n_keep {
                px2 += alpha[i] * g[i * n_keep + j] * alpha[j];
            }
        }
        let mut x2 = 0.0f64;
        for t in 0..dim {
            let v = edit[r * dim + t] as f64;
            x2 += v * v;
        }
        let overlap = (px2.max(0.0).sqrt()) / (x2.max(1e-30).sqrt());
        overlaps.push(OverlapRow { row: r, overlap });
    }
    let overlap_mean = overlaps.iter().map(|o| o.overlap).sum::<f64>() / n_edit as f64;
    let overlap_min = overlaps
        .iter()
        .map(|o| o.overlap)
        .fold(f64::INFINITY, f64::min);
    let overlap_max = overlaps.iter().map(|o| o.overlap).fold(0.0f64, f64::max);

    // Joint Gram census (keep + edit rows).
    let n_joint = n_keep + n_edit;
    let mut jm = vec![0.0f64; n_joint * n_joint];
    for i in 0..n_joint {
        for j in 0..n_joint {
            let mut acc = 0.0f64;
            for t in 0..dim {
                let vi = if i < n_keep {
                    keep[i * dim + t]
                } else {
                    edit[(i - n_keep) * dim + t]
                };
                let vj = if j < n_keep {
                    keep[j * dim + t]
                } else {
                    edit[(j - n_keep) * dim + t]
                };
                acc += (vi as f64) * (vj as f64);
            }
            jm[i * n_joint + j] = acc;
        }
    }
    let mut jmm = jm.clone();
    let mut jvecs = vec![0.0f64; n_joint * n_joint];
    let jeigs_unsorted = jacobi_symmetric(&mut jmm, &mut jvecs, n_joint);
    let mut jpairs: Vec<(f64, usize)> = jeigs_unsorted.iter().copied().zip(0..n_joint).collect();
    jpairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let jfloor = 1e-10
        * jeigs_unsorted
            .iter()
            .fold(0.0f64, |m, &v| m.max(v.abs()))
            .max(1.0);
    let joint_rank = jpairs.iter().filter(|(v, _)| *v > jfloor).count();
    let joint_min_nonzero_eig = jpairs
        .iter()
        .find(|(v, _)| *v > jfloor)
        .map(|(v, _)| *v)
        .unwrap_or(0.0);

    G0Census {
        overlaps,
        overlap_mean,
        overlap_min,
        overlap_max,
        keep_rank,
        keep_min_nonzero_eig,
        joint_rank,
        joint_min_nonzero_eig,
    }
}

// ---------------------------------------------------------------------------
// Fault-signature instruments (arm-1 baseline; the R2 minimal core)
// ---------------------------------------------------------------------------

/// Per-(layer, transition) pass-cosine ladder.
pub struct PassCosines {
    /// `layer1[k]` = cos(h1_{k+1}, h1_k) for transition k→k+1 (index 0 = 1→2).
    pub layer1: Vec<f32>,
    /// `layer2[k]` = cos(y_{k+1}, y_k).
    pub layer2: Vec<f32>,
}

/// Cosine of two slices (f64 fold, f32 result).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}

/// The fault-signature report from one ladder run (mean over the batch where
/// a scalar is needed; the raw per-batch values stay available for R2's
/// per-pass projection census).
pub struct FaultSignature {
    pub cosines: PassCosines,
    /// Gate means by pass (batch mean), length K_LADDER.
    pub gate_means: Vec<f32>,
    /// Per-pass mean shift norms ‖y_{k+1} − y_k‖ (length K_LADDER − 1).
    pub shift_norms: Vec<f32>,
    /// Mean over batch of per-batch ‖dY‖ values at the fault transition
    /// (d → d+1), i.e. `shift_norms[d-1]` batch-meaned.
    pub fault_shift_norm: f32,
}

/// Compute the fault-signature instruments over the batch. Returns the
/// report; the caller asserts the Inkling shape against the pre-registered
/// bars (issue §"T1 fixture design").
pub fn fault_signature(fixture: &LoopDepthOod) -> FaultSignature {
    let mut cos_l1 = vec![0.0f32; K_LADDER - 1];
    let mut cos_l2 = vec![0.0f32; K_LADDER - 1];
    let mut gate_means = vec![0.0f32; K_LADDER];
    let mut shift_norms = vec![0.0f32; K_LADDER - 1];
    for i in 0..BATCH {
        let x0 = fixture.warm_start(i);
        let (gates, outs) = fixture.ladder(&x0);
        for (k, g) in gates.iter().enumerate() {
            gate_means[k] += g / BATCH as f32;
        }
        // Layer-1 states are recomputed here so R2's localization census can
        // read them (the gate sits between the layers; layer 1 must stay
        // healthy at every depth).
        let mut x = x0.clone();
        let mut h1s: Vec<Vec<f32>> = Vec::with_capacity(K_LADDER);
        for k in 1..=K_LADDER {
            let h1 = fixture.layer1(&x);
            let phi = LoopDepthOod::pass_indicator_ramp(k);
            let (_g, h1g) = fixture.gate_apply(&h1, phi);
            h1s.push(h1);
            x = fixture.layer2(&h1g);
        }
        for t in 0..K_LADDER - 1 {
            cos_l1[t] += cosine(&h1s[t + 1], &h1s[t]) / BATCH as f32;
            cos_l2[t] += cosine(&outs[t + 1], &outs[t]) / BATCH as f32;
            let sn: f64 = outs[t]
                .iter()
                .zip(&outs[t + 1])
                .map(|(a, b)| {
                    let d = (a - b) as f64;
                    d * d
                })
                .sum::<f64>();
            shift_norms[t] += (sn.sqrt()) as f32 / BATCH as f32;
        }
    }
    FaultSignature {
        cosines: PassCosines {
            layer1: cos_l1,
            layer2: cos_l2,
        },
        gate_means,
        fault_shift_norm: shift_norms[D_TRAIN - 1],
        shift_norms,
    }
}

// ---------------------------------------------------------------------------
// R2 — the fault-localizer census (readout-screen class: offline, never a
// serving path). Per layer: the pass-shift matrix's leading left-singular
// direction v₁, the per-pass projections, and the typed fault verdict. On a
// real looped checkpoint this is the pre-flight gate for "is this checkpoint
// even a depth-extension candidate".
// ---------------------------------------------------------------------------

/// R2's per-layer census row.
pub struct LayerR2 {
    /// 0-based layer index (0 = the fault-free layer, 1 = the constructed
    /// fault site's consumer).
    pub layer: usize,
    /// ⟨s_k, v₁⟩ per transition, batch-meaned shifts; v₁ oriented at the
    /// largest-|projection| transition.
    pub v1_projections: Vec<f64>,
    /// The transition index the verdict names (index 0 = transition 1→2).
    /// `None` = the ladder reads healthy at this layer.
    pub faulted_at: Option<usize>,
}

/// R2's report: one row per layer, plus the site verdict.
pub struct R2Report {
    pub layers: Vec<LayerR2>,
    /// The fault SITE: the earliest (transition, then layer) faulted verdict.
    /// On a real checkpoint this is the pre-flight gate's answer — `None` =
    /// the ladder reads healthy everywhere.
    pub first_fault: Option<(usize, usize)>,
}

/// Batch-mean per-pass states of one signal (the ladder's layer states).
fn batch_mean_states(fixture: &LoopDepthOod, layer: usize) -> Vec<Vec<f64>> {
    let mut states = vec![vec![0.0f64; D]; K_LADDER];
    for i in 0..BATCH {
        let x0 = fixture.warm_start(i);
        let mut x = x0.clone();
        for (k, acc_state) in states.iter_mut().enumerate() {
            let h1 = fixture.layer1(&x);
            let phi = LoopDepthOod::pass_indicator_ramp(k + 1);
            let (_g, h1g) = fixture.gate_apply(&h1, phi);
            let s = if layer == 0 {
                h1.clone()
            } else {
                fixture.layer2(&h1g)
            };
            for (acc, v) in acc_state.iter_mut().zip(&s) {
                *acc += (*v as f64) / BATCH as f64;
            }
            x = fixture.layer2(&h1g);
        }
    }
    states
}

/// The shift-SVD census for ONE signal: batch-mean pass-shifts, v₁ as the
/// leading left-singular direction of S Sᵀ (Jacobi, f64, oriented at the
/// largest-|projection| transition), the per-pass projections.
pub fn r2_shift_census(states: &[Vec<f64>]) -> Vec<f64> {
    let n_trans = states.len() - 1;
    let mut shifts = vec![vec![0.0f64; D]; n_trans];
    for t in 0..n_trans {
        for j in 0..D {
            shifts[t][j] = states[t + 1][j] - states[t][j];
        }
    }
    // S Sᵀ (D×D, sequential f64 folds).
    let mut sst = vec![0.0f64; D * D];
    for i in 0..D {
        for j in 0..D {
            let acc: f64 = shifts.iter().map(|s| s[i] * s[j]).sum();
            sst[i * D + j] = acc;
        }
    }
    let mut v = vec![0.0f64; D * D];
    let mut m = sst;
    let eigs = jacobi_symmetric(&mut m, &mut v, D);
    let lead_col = eigs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut v1 = vec![0.0f64; D];
    for r in 0..D {
        v1[r] = v[r * D + lead_col];
    }
    let v1n: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v1 {
        *x /= v1n;
    }
    let mut projs: Vec<f64> = shifts
        .iter()
        .map(|s| s.iter().zip(&v1).map(|(a, b)| a * b).sum())
        .collect();
    // Orient v₁ at the largest-|projection| transition (the eigenvector's
    // global sign is conventional; the invariant is the trajectory shape).
    let lead_t = (0..n_trans)
        .max_by(|a, b| projs[*a].abs().partial_cmp(&projs[*b].abs()).unwrap())
        .unwrap_or(0);
    if projs[lead_t] < 0.0 {
        for p in &mut projs {
            *p = -*p;
        }
    }
    projs
}

/// The R2 verdict rule: a layer is FAULTED at its EARLIEST transition t
/// whose |proj_t| clears `R2_DOMINANCE` × the median |projection| of the
/// remaining transitions (the fault transition dominates the ladder's own
/// shift trajectory; a healthy layer's transitions are homogeneous). Later
/// dominant transitions at the same layer are downstream contamination (the
/// wave), not the site — the report's `first_fault` names the earliest.
const R2_DOMINANCE: f64 = 10.0;

pub fn r2_fault_localizer(fixture: &LoopDepthOod) -> R2Report {
    let mut layers = Vec::with_capacity(2);
    for layer in 0..2usize {
        let states = batch_mean_states(fixture, layer);
        let projs = r2_shift_census(&states);
        let n_trans = projs.len();
        // Earliest dominant transition (not merely the largest — later
        // spikes are the propagation wave).
        let mut faulted_at = None;
        for t in 0..n_trans {
            let mut others: Vec<f64> = projs
                .iter()
                .enumerate()
                .filter(|(u, _)| *u != t)
                .map(|(_, p)| p.abs())
                .collect();
            others.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = others[others.len() / 2];
            if projs[t].abs() >= R2_DOMINANCE * med {
                faulted_at = Some(t);
                break;
            }
        }
        layers.push(LayerR2 {
            layer,
            v1_projections: projs,
            faulted_at,
        });
    }
    // The site: earliest (transition, layer) among the faulted verdicts.
    let first_fault = layers
        .iter()
        .filter_map(|l| l.faulted_at.map(|t| (t, l.layer)))
        .min()
        .map(|(t, l)| (l, t));
    R2Report {
        layers,
        first_fault,
    }
}

// ---------------------------------------------------------------------------
// R3 — the gate-saturation alarm + graceful early exit. The in-loop OOD
// guard: read the gate BEFORE committing a pass; on a collapse below the
// floor, stop and return the passes computed so far — "override depth and
// gracefully stay shallow", never worse than refusing the depth.
// ---------------------------------------------------------------------------

/// The gate floor: a pass whose batch-mean gate output falls below this is
/// refused (the healthy in-band ladder sits at ≈ 0.6-0.7; the fault pass
/// reads ≈ 0.0002).
pub const GATE_FLOOR: f32 = 0.35;

/// The alarm's report.
pub struct GateAlarmReport {
    /// The 0-based pass index where the alarm fired (`None` = the whole
    /// ladder ran clean).
    pub fired_at: Option<usize>,
    /// Passes completed before the exit (the graceful exit keeps these).
    pub passes_completed: usize,
    /// The per-pass batch-mean gate telemetry the alarm read.
    pub gate_means: Vec<f32>,
    /// Per-row outputs for the COMPLETED passes (row-major: `outputs[r][k]`).
    pub outputs: Vec<Vec<Vec<f32>>>,
}

/// Run the ladder with the alarm armed: per pass, read the BATCH-MEAN gate
/// BEFORE committing the pass (phase 1 computes every row's gate; phase 2
/// commits the pass only if the mean clears the floor); on a collapse, stop
/// and return the completed passes. With `armed = false` the telemetry is
/// recorded but the alarm never fires (identical pass math, the disarm only
/// removes the refusal branch).
pub fn run_with_gate_alarm(fixture: &LoopDepthOod, armed: bool) -> GateAlarmReport {
    let mut states: Vec<Vec<f32>> = (0..BATCH).map(|i| fixture.warm_start(i)).collect();
    let mut outputs: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut gate_means = Vec::new();
    let mut fired_at = None;
    for k in 1..=K_LADDER {
        let phi = LoopDepthOod::pass_indicator_ramp(k);
        // Phase 1 — read the telemetry (the gate is computed from the pass's
        // layer-1 input, before the pass's output exists to consume).
        let mut pass_gates = [0.0f32; BATCH];
        let mut h1s = Vec::with_capacity(BATCH);
        for (r, x) in states.iter().enumerate() {
            let h1 = fixture.layer1(x);
            let z = fixture.gate_preact(&h1, phi);
            pass_gates[r] = 1.0 / (1.0 + (-z as f64).exp()) as f32;
            h1s.push(h1);
        }
        let mean = pass_gates.iter().sum::<f32>() / BATCH as f32;
        if armed && mean < GATE_FLOOR {
            fired_at = Some(k - 1);
            gate_means.push(mean);
            break;
        }
        gate_means.push(mean);
        // Phase 2 — commit the pass.
        let mut pass_outs = Vec::with_capacity(BATCH);
        for h1 in &h1s {
            let (_g, h1g) = fixture.gate_apply(h1, phi);
            pass_outs.push(fixture.layer2(&h1g));
        }
        for (x, y) in states.iter_mut().zip(&pass_outs) {
            *x = y.clone();
        }
        outputs.push(pass_outs);
    }
    GateAlarmReport {
        fired_at,
        passes_completed: outputs.len(),
        gate_means,
        outputs,
    }
}

// ---------------------------------------------------------------------------
// R4 — the spectral single-direction governor (the ladder's middle rung:
// damping modulates, the governor removes ONE direction, the edit re-writes
// weights). `x ← x − γ·(v₁ᵀx)·v₁` at the fault site, one frozen vector from
// R2's census, γ driven by gate telemetry: ≈ 0 while the gate reads healthy
// (byte-identical healthy passes), ramping as the gate collapses.
// ---------------------------------------------------------------------------

/// The governor's telemetry curve: γ_k = clamp(1 − g_{k−1}/GATE_FLOOR, 0, 1)
/// — the PREVIOUS pass's batch-mean gate (the telemetry-lag form; a healthy
/// pass 1..d reads γ = 0, a collapsed pass reads γ = 1, the first fault pass
/// reads γ from pass d's healthy telemetry — the governor cannot see the
/// fault before the loop produces it).
pub fn governor_gamma(prev_gate_mean: Option<f32>) -> f32 {
    match prev_gate_mean {
        None => 0.0,
        Some(g) => (1.0 - g / GATE_FLOOR).clamp(0.0, 1.0),
    }
}

/// The governor's report: the arm's outputs + per-pass telemetry.
pub struct GovernorRun {
    /// Per-pass batch-mean gate values (length K_LADDER).
    pub gate_means: Vec<f32>,
    /// The γ actually applied per pass (0 for pass 1 and every healthy pass).
    pub gammas: Vec<f32>,
    /// Per-row outputs (row-major: `outputs[r][k]`, all K_LADDER passes).
    pub outputs: Vec<Vec<Vec<f32>>>,
}

/// Run the full ladder with the governor armed past the trained depth: at
/// pass k > d the carried state is projected (γ from the lagged telemetry)
/// BEFORE the pass runs; passes 1..d are byte-identical to the unarmed
/// ladder (γ = 0 there by construction). `v1` is R2's frozen census vector
/// at the fault site, normalized (the caller passes `r2_shift_census`'s
/// output; the direction is unit by construction).
pub fn governor_run(fixture: &LoopDepthOod, v1: &[f64]) -> GovernorRun {
    assert_eq!(v1.len(), D, "v₁ must be a D-sized census vector");
    let nrm: f64 = v1.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!((nrm - 1.0).abs() < 1e-6, "v₁ not normalized: ‖·‖ = {nrm}");
    let v1f: Vec<f32> = v1.iter().map(|&c| c as f32).collect();
    let mut states: Vec<Vec<f32>> = (0..BATCH).map(|i| fixture.warm_start(i)).collect();
    let mut outputs = Vec::new();
    let mut gate_means = Vec::new();
    let mut gammas = Vec::new();
    let mut prev_mean: Option<f32> = None;
    for k in 1..=K_LADDER {
        let phi = LoopDepthOod::pass_indicator_ramp(k);
        let gamma = if k > D_TRAIN {
            governor_gamma(prev_mean)
        } else {
            0.0
        };
        gammas.push(gamma);
        let mut pass_gates = Vec::with_capacity(BATCH);
        let mut pass_outs = Vec::with_capacity(BATCH);
        for x in states.iter_mut() {
            // The intervention: project the divergent component out of the
            // carried message before the pass consumes it.
            if gamma > 0.0 {
                let proj: f32 = x.iter().zip(&v1f).map(|(a, b)| a * b).sum();
                for (xc, vc) in x.iter_mut().zip(&v1f) {
                    *xc -= gamma * proj * vc;
                }
            }
            let h1 = fixture.layer1(x);
            let z = fixture.gate_preact(&h1, phi);
            pass_gates.push(1.0 / (1.0 + (-z as f64).exp()) as f32);
            let (_g, h1g) = fixture.gate_apply(&h1, phi);
            pass_outs.push(fixture.layer2(&h1g));
        }
        let mean = pass_gates.iter().sum::<f32>() / BATCH as f32;
        gate_means.push(mean);
        prev_mean = Some(mean);
        for (x, y) in states.iter_mut().zip(&pass_outs) {
            *x = y.clone();
        }
        outputs.push(pass_outs);
    }
    GovernorRun {
        gate_means,
        gammas,
        outputs,
    }
}

// ---------------------------------------------------------------------------
// T5 — the FP-Solve edit arm: the loop-as-label protected-ridge solve, a
// rank-r overlay pair on the OoD-consuming layer's control matrix, an
// envelope-pinned mount (empty overlay demotes byte-identically).
// Serving posture = the fixture's own f32 forward; the solve runs f64
// (conditioning) and the artifact is rounded to f32 at apply (Issue-940:
// the capture posture and the apply posture are the same build).
// ---------------------------------------------------------------------------

/// The edit's pinned configuration (the pre-registered constants; `ridge_frac`
/// pins the 0.06 spelling — the source's own call sites drift 0.06..0.08,
/// Research 697 row 7).
pub struct EditConfig {
    /// λ_keep — the pass-domain keep weight (mean-Gram scale, both Grams
    /// normalized by their own row counts).
    pub keep_weight: f64,
    /// ridge_frac — λ₂ = ridge_frac · tr(G_edit)/D_in.
    pub ridge_frac: f64,
    /// The overlay's rank cap (thin-SVD truncation of the solved ΔW).
    pub rank_r: usize,
}

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            keep_weight: 1.0,
            ridge_frac: 0.06,
            rank_r: 8,
        }
    }
}

/// The capture on the serving posture: layer-2 inputs and outputs per row —
/// keep rows from passes 1..d, edit rows + per-row dY labels from pass d+1
/// (the loop's OWN pass-d output is the label — the loop-as-label law; the
/// input to the solve is the POST-norm vector the matrix actually
/// multiplies).
pub struct EditCapture {
    /// Keep input rows (post-rmsnorm), B·d rows × D.
    pub x_keep: Vec<f32>,
    /// Edit input rows (post-rmsnorm) at pass d+1, B rows × D.
    pub x_edit: Vec<f32>,
    /// Per-row output labels: y_d − y_{d+1}, B rows × D (f64, sequential
    /// folds of the f32 captures).
    pub dy: Vec<f64>,
    /// BLAKE3 of the capture bytes (inputs + labels, row-major, little-endian
    /// f32/f64 bits) — the envelope binds this.
    pub capture_digest: blake3::Hash,
}

/// Capture the edit's rows on the fixture's serving posture.
pub fn capture_edit(fixture: &LoopDepthOod) -> EditCapture {
    let mut x_keep = Vec::with_capacity(BATCH * D_TRAIN * D);
    let mut x_edit = Vec::with_capacity(BATCH * D);
    let mut dy = Vec::with_capacity(BATCH * D);
    for i in 0..BATCH {
        let mut x = fixture.warm_start(i);
        let mut y_prev: Option<Vec<f32>> = None;
        for k in 1..=K_LADDER {
            let h1 = fixture.layer1(&x);
            let phi = LoopDepthOod::pass_indicator_ramp(k);
            let (_g, h1g) = fixture.gate_apply(&h1, phi);
            let nv = LoopDepthOod::rmsnorm(&h1g);
            let y = fixture.layer2(&h1g);
            if k <= D_TRAIN {
                x_keep.extend_from_slice(&nv);
            }
            if k == D_TRAIN + 1 {
                x_edit.extend_from_slice(&nv);
                let y_d = y_prev.expect("pass d output must precede d+1");
                for (a, b) in y_d.iter().zip(&y) {
                    dy.push((*a as f64) - (*b as f64));
                }
            }
            y_prev = Some(y.clone());
            x = y;
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&bytemuck_bytes(&x_keep));
    hasher.update(&bytemuck_bytes(&x_edit));
    hasher.update(&f64_bytes(&dy));
    EditCapture {
        x_keep,
        x_edit,
        dy,
        capture_digest: hasher.finalize(),
    }
}

/// f32 slice → little-endian bytes (std `to_ne_bytes` is fine on one box;
/// the digest is a local pin, never a wire format — pinned as LE for
/// cross-box reproducibility anyway).
fn bytemuck_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for f in v {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for f in v {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

/// The solved edit + its rank-r overlay artifact.
pub struct EditArtifact {
    /// The full solved ΔW (f64, D×D row-major) — the pre-truncation form.
    pub dw_full: Vec<f64>,
    /// The rank-r overlay: U·Vᵀ (U: r×D, V: r×D, row-major), the THIN-SVD
    /// truncation of `dw_full` (the factor_tail discipline; dropped energy
    /// reported).
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    /// Σ of the KEPT singular values and of the DROPPED tail (the disclosed
    /// truncation energy).
    pub kept_energy: f64,
    pub dropped_energy: f64,
    /// The apply-side (f32-rounded) overlay rows: U, V cast to f32.
    pub u32: Vec<f32>,
    pub v32: Vec<f32>,
    /// BLAKE3 over (capture digest ‖ config ‖ the f32 overlay bytes) — the
    /// envelope pin.
    pub envelope_digest: blake3::Hash,
}

/// The protected-ridge solve: dWᵀ = (G_edit + λ_keep·G_keep + λ₂·I)⁻¹ ·
/// (X_editᵀ·dY / N_e), G_* the mean Grams (each normalized by its own row
/// count), λ₂ = ridge_frac · tr(G_edit)/D_in. Solves f64; returns the
/// rank-r artifact with the f32 apply-side rows.
pub fn fp_solve_edit(capture: &EditCapture, cfg: &EditConfig) -> EditArtifact {
    let n_keep = capture.x_keep.len() / D;
    let n_edit = capture.x_edit.len() / D;
    // Mean Grams (f64, sequential folds over rows).
    let mut g_edit = vec![0.0f64; D * D];
    for r in 0..n_edit {
        let row = &capture.x_edit[r * D..(r + 1) * D];
        gram_into(&mut g_edit, row);
    }
    for c in &mut g_edit {
        *c /= n_edit as f64;
    }
    let mut g_keep = vec![0.0f64; D * D];
    for r in 0..n_keep {
        let row = &capture.x_keep[r * D..(r + 1) * D];
        gram_into(&mut g_keep, row);
    }
    for c in &mut g_keep {
        *c /= n_keep as f64;
    }
    let tr: f64 = (0..D).map(|i| g_edit[i * D + i]).sum();
    let ridge = cfg.ridge_frac * tr / D as f64;
    // The protected M-matrix, then the RHS: X_editᵀ·dY / N_e.
    let mut m = vec![0.0f64; D * D];
    for i in 0..D {
        for j in 0..D {
            m[i * D + j] = g_edit[i * D + j]
                + cfg.keep_weight * g_keep[i * D + j]
                + if i == j { ridge } else { 0.0 };
        }
    }
    let mut rhs = vec![0.0f64; D * D];
    for r in 0..n_edit {
        let row = &capture.x_edit[r * D..(r + 1) * D];
        let dyr = &capture.dy[r * D..(r + 1) * D];
        // rhs += outer(row, dY_r) / N_e
        for i in 0..D {
            for j in 0..D {
                rhs[i * D + j] += (row[i] as f64) * dyr[j] / n_edit as f64;
            }
        }
    }
    // dWᵀ = M⁻¹ · rhs, one SPD solve per output column j. The APPLY needs
    // Δ = (dWᵀ)ᵀ = dYᵀ·X·M⁻¹ (Δ·x uses Δ[i,j]·x[j]; M symmetric) — the
    // transpose is taken BEFORE the rank-r fit so the overlay applies
    // U·Vᵀ ≈ Δ directly.
    let mut dw_t = vec![0.0f64; D * D];
    for j in 0..D {
        let col: Vec<f64> = (0..D).map(|i| rhs[i * D + j]).collect();
        let sol = solve_spd(&m, &col, D);
        for i in 0..D {
            dw_t[i * D + j] = sol[i];
        }
    }
    let mut dw_full = vec![0.0f64; D * D]; // Δ (the apply convention)
    for i in 0..D {
        for j in 0..D {
            dw_full[i * D + j] = dw_t[j * D + i];
        }
    }
    // Thin-SVD truncation via the eigendecomposition of dWᵀ's Gram —
    // dWᵀ = UΣVᵀ: the eigvecs of dWᵀdW give V, Σ; U = dWᵀ·V/Σ.
    let mut gram = vec![0.0f64; D * D];
    for i in 0..D {
        for j in 0..D {
            let acc: f64 = (0..D)
                .map(|t| dw_full[t * D + i] * dw_full[t * D + j])
                .sum();
            gram[i * D + j] = acc;
        }
    }
    let mut vecs = vec![0.0f64; D * D];
    let eigs = jacobi_symmetric(&mut gram, &mut vecs, D);
    let mut pairs: Vec<(f64, usize)> = eigs.iter().copied().zip(0..D).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // descending
    let r = cfg.rank_r.min(D);
    let mut u = vec![0.0f64; r * D];
    let mut v = vec![0.0f64; r * D];
    let mut kept_energy = 0.0f64;
    let mut dropped_energy = 0.0f64;
    for (rank, &(eig, col)) in pairs.iter().enumerate() {
        let sv = eig.max(0.0).sqrt();
        if rank < r {
            kept_energy += sv * sv;
            for d0 in 0..D {
                // V row = the eigenvector (column `col`); U row = Δ·v/σ —
                // (Δ·v)[d0] = Σ_t Δ[d0, t]·v[t]: both indices read the
                // APPLY-convention Δ (dw_full), never its transpose.
                v[rank * D + d0] = vecs[d0 * D + col];
                let u_acc: f64 = (0..D)
                    .map(|t| dw_full[d0 * D + t] * vecs[t * D + col])
                    .sum();
                u[rank * D + d0] = if sv > 1e-12 { u_acc / sv } else { 0.0 };
            }
        } else {
            dropped_energy += sv * sv;
        }
    }
    // f32 apply-side rows + the envelope digest (binds capture + config +
    // the artifact bytes).
    let u32: Vec<f32> = u.iter().map(|&c| c as f32).collect();
    let v32: Vec<f32> = v.iter().map(|&c| c as f32).collect();
    let mut hasher = blake3::Hasher::new();
    hasher.update(capture.capture_digest.as_bytes());
    hasher.update(&cfg.keep_weight.to_le_bytes());
    hasher.update(&cfg.ridge_frac.to_le_bytes());
    hasher.update(&(cfg.rank_r as u64).to_le_bytes());
    hasher.update(&f32_le_bytes(&u32));
    hasher.update(&f32_le_bytes(&v32));
    EditArtifact {
        dw_full,
        u,
        v,
        kept_energy,
        dropped_energy,
        u32,
        v32,
        envelope_digest: hasher.finalize(),
    }
}

fn gram_into(g: &mut [f64], row: &[f32]) {
    for i in 0..D {
        for j in 0..D {
            g[i * D + j] += (row[i] as f64) * (row[j] as f64);
        }
    }
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for f in v {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

/// The overlay-applied layer-2 forward: `y = (A2 + ΔW)·rmsnorm(h1g)` with
/// ΔW = U·Vᵀ reconstructed from the f32 rows (rank-r, never materialized
/// dense — the UNMERGED serving posture's cost, T7's census). The unarmed
/// `layer2` path is untouched — the empty-overlay demote is structural.
pub fn layer2_overlay(fixture: &LoopDepthOod, h1g: &[f32], u32: &[f32], v32: &[f32]) -> Vec<f32> {
    let nv = LoopDepthOod::rmsnorm(h1g);
    // The overlay's contribution: y_ow = U·(Vᵀ·nv).
    let r = u32.len() / D;
    let mut dots = vec![0.0f32; r];
    for (ri, vrow) in v32.chunks(D).enumerate() {
        dots[ri] = nv.iter().zip(vrow).map(|(a, b)| a * b).sum::<f32>();
    }
    let mut delta = vec![0.0f32; D];
    for (ri, urow) in u32.chunks(D).enumerate() {
        let c = dots[ri];
        for (d0, &uval) in urow.iter().enumerate() {
            delta[d0] += c * uval;
        }
    }
    // The base pass through the fixture's own layer-2, plus the delta.
    let mut y = fixture.layer2(h1g);
    for (yv, dv) in y.iter_mut().zip(&delta) {
        *yv += *dv;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture builds and the frozen geometry is deterministic: two
    /// builds from the same seed are bit-identical.
    #[test]
    fn fixture_build_is_deterministic() {
        let a = LoopDepthOod::build();
        let b = LoopDepthOod::build();
        assert_eq!(a.e1, b.e1);
        assert_eq!(a.e2, b.e2);
        assert_eq!(a.e3, b.e3);
        assert_eq!(a.w_dir, b.w_dir);
        assert_eq!(a.w, b.w);
        for (qa, qb) in a.q2.iter().zip(&b.q2) {
            assert_eq!(qa, qb);
        }
    }

    /// The seeded axes are orthonormal: ‖e₁‖ = ‖e₂‖ = ‖e₃‖ = ‖ŵ‖ = 1, the
    /// axes mutually ⊥; Q2 is orthogonal (first few rows).
    #[test]
    fn fixture_bases_orthonormal() {
        let fx = LoopDepthOod::build();
        for (name, ax) in [
            ("e1", &fx.e1),
            ("e2", &fx.e2),
            ("e3", &fx.e3),
            ("w", &fx.w_dir),
        ] {
            let n: f32 = ax.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "‖{name}‖ = {n}");
        }
        let e12: f32 = fx.e1.iter().zip(&fx.e2).map(|(a, b)| a * b).sum();
        let e13: f32 = fx.e1.iter().zip(&fx.e3).map(|(a, b)| a * b).sum();
        let e23: f32 = fx.e2.iter().zip(&fx.e3).map(|(a, b)| a * b).sum();
        assert!(e12.abs() < 1e-4, "⟨e1,e2⟩ = {e12}");
        assert!(e13.abs() < 1e-4, "⟨e1,e3⟩ = {e13}");
        assert!(e23.abs() < 1e-4, "⟨e2,e3⟩ = {e23}");
        for i in 0..4 {
            let ni: f32 = fx.q2[i].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((ni - 1.0).abs() < 1e-3, "‖q2[{i}]‖ = {ni}");
            for j in (i + 1)..4 {
                let q: f32 = fx.q2[i].iter().zip(&fx.q2[j]).map(|(a, b)| a * b).sum();
                assert!(q.abs() < 1e-3, "⟨q2[{i}],q2[{j}]⟩ = {q}");
            }
        }
    }

    /// The gate is healthy on the seen band and collapsed at the OOD pass:
    /// feed a WARM state (the serving posture) with ramp indicators
    /// φ ∈ {1/3, 1, 4/3}.
    #[test]
    fn gate_band_structure() {
        let fx = LoopDepthOod::build();
        let x0 = fx.warm_start(0);
        let h1 = fx.layer1(&x0);
        let g_lo = {
            let z = fx.gate_preact(&h1, LoopDepthOod::pass_indicator_ramp(1));
            1.0 / (1.0 + (-z as f64).exp()) as f32
        };
        let g_hi = {
            let z = fx.gate_preact(&h1, LoopDepthOod::pass_indicator_ramp(3));
            1.0 / (1.0 + (-z as f64).exp()) as f32
        };
        let g_ood = {
            let z = fx.gate_preact(&h1, LoopDepthOod::pass_indicator_ramp(4));
            1.0 / (1.0 + (-z as f64).exp()) as f32
        };
        assert!(g_lo > 0.5 && g_lo < 0.75, "g(1/3) = {g_lo}");
        assert!(g_hi > 0.5 && g_hi < 0.75, "g(1) = {g_hi}");
        assert!(g_ood < 0.15, "g(4/3) = {g_ood}");
    }

    /// The sinusoid fallback spelling reproduces the band structure on BOTH
    /// band edges (seen values healthy, the first out-of-band pass collapsed).
    #[test]
    fn gate_band_structure_sinusoid_fallback() {
        let fx = LoopDepthOod::build();
        let x0 = fx.warm_start(0);
        let h1 = fx.layer1(&x0);
        for k in 1..=D_TRAIN {
            let phi = LoopDepthOod::pass_indicator_sinusoid(k);
            let z = fx.gate_preact_enc(&h1, phi, PassIndicator::Sinusoid);
            let g = 1.0 / (1.0 + (-z as f64).exp()) as f32;
            assert!(g > 0.45, "sinusoid in-band g(k={k}) = {g}");
        }
        for k in D_TRAIN + 1..=D_TRAIN + 2 {
            let phi = LoopDepthOod::pass_indicator_sinusoid(k); // 0, then −0.707
            let z = fx.gate_preact_enc(&h1, phi, PassIndicator::Sinusoid);
            let g = 1.0 / (1.0 + (-z as f64).exp()) as f32;
            assert!(g < 0.15, "sinusoid OOD g(k={k}) = {g}");
        }
    }
}
