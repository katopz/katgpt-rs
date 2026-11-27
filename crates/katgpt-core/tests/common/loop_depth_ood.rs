//! `loop_depth_ood` — the Issue-1248 constructed fixture (T1).
//!
//! A two-layer looped forward whose passes 1..d converge (rank-1-dominant
//! direction contraction under rmsnorm) with ONE pass-indexed gate module
//! between the layers: healthy for the seen pass band `φ ∈ (0, 1]`, collapsed
//! at `φ > 1` — the Inkling fault class (a seen direction whose influence
//! flips) reproduced at a LOCALIZED site (layer 2, the OOD-consuming layer).
//!
//! The design is pre-registered in `.issues/1248_closed_form_loop_depth_extension.md`
//! §"T1 fixture design" (written before code, per T1): encoding = continuous
//! scalar ramp `φ = k/d` (sinusoid fallback, same κ), geometry = rank-1-dominant
//! contraction so the keep span is effectively 1-D and the gate-collapse
//! deficit (−0.58·b) leaves it by a real margin — the G0 overlap lands strictly
//! between the two rigged poles.
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
