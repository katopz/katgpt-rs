//! Issue 1239 G1/G3 — the conditioned-KARC correction-bank cell.
//!
//! Pre-registered in `.issues/1239_conditioned_karc_correction_bank.md`
//! (PINNED 2026-11-22 block): corpora, splits, arms, decision rule. This
//! harness runs the cell at the pinned small shape (D=3, K=4, M=8,
//! d_h=96, λ=1e-4) and prints the full verdict table. It asserts the
//! GATE outcomes (G3 abstain is a hard gate; the demote-to-null shape —
//! A5 beating A4 — panics); the promotion question (G1 WIN/TIE) is
//! adjudicated by the bench record, not by this test passing alone.
//!
//! Corpora (the PINNED block):
//! - **C1 regime-switching** — double-scroll ODE (the
//!   `karc_g1_dh18720.rs` parameters, dt 0.25 substepped RK4) + a
//!   condition-keyed constant drive on dV1 (blocks of 200 steps, c∈{0,1};
//!   DRIVE = 0.08 raw units, pinned at harness time and disclosed in the
//!   bench record — the issue pinned the mechanism, not the magnitude;
//!   0.5 explodes the attractor into sinh runaway, 0.08 is the largest
//!   bounded scan step),
//!   normalized to [−1,1] per coordinate on the noiseless seed-0
//!   trajectory, then 5 seeds of observation noise σ=0.01 (normalized
//!   units) = the noise-robustness arm.
//! - **C2 pure double-scroll** (c ≡ 0; corrections must abstain — G3).
//! - **C3 periodic** — 3-channel multi-sine, period 20 (c ≡ 0; G3 + the
//!   G5 SeasonalNaive floor reported beside it).
//!
//! Splits: fit 0–60 %, gate calibration 60–70 %, test 70–100 %.
//! One-step-ahead NRMSE (teacher-forced delay features); train NRMSE
//! reported beside (the Bench-992 gap check).
//!
//! Arms at matched bytes (the issue's ledger — asserted below):
//! A1 base 288 · A2 regime-mux (SELECT control) · A3 uniform keyed mean ·
//! A3b Σ/√2 scale control · A4 keyed σ-gated (486) · A5 single wide M=13
//! (468) · A6 two private M=7 forecasters, forecast mean (504) · A1c base +
//! per-condition bias (294) · A1k per-condition full readouts (576).
//!
//! Decision rule (per seed r = 100·(NRMSE_arm − NRMSE_ref)/NRMSE_ref pp;
//! δ = max(1.0, 2.5·SE); WIN r̄ < −δ, LOSS r̄ > δ):
//! - **G1** = A4 WIN vs A1 on C1 AND A5 not WIN vs A4.
//! - **Attribution** = A4 vs A1c (a keyed win must beat keyed bias).
//! - **Promotion-bearing** = A3 WIN vs A1.
//! - **G3** = A4 not LOSS vs A1 on C2 and C3 (single-seed corpora: δ floor
//!   1.0 pp, disclosed).
//! - **G5** = the SeasonalNaive floor's NRMSE on C3 reported beside.
//!
//! Run (release posture, on a rig — the job-lane law):
//! `cargo test --release -p katgpt-core --features estimators \
//!   --test karc_conditioned_g1 -- --nocapture`

use katgpt_core::linalg::ridge_solve_direct_f64;
use katgpt_core::{
    BankFitScratch, ChebyshevBasis, DelayRing, GateConfig, KarcConditionedReadout, KarcRegime,
    KarcRegimeGate, PointForecaster, SeasonalPoolForecaster, archetype_direction_into,
    chunked_gram_into, feature_expand, fit_correction_into, seasonal_naive_floor,
};

// ── Double-scroll ODE (karc_g1_dh18720.rs parameters, verbatim) ─────────

const R1: f64 = 1.2;
const R2: f64 = 3.44;
const R4: f64 = 0.193;
const BETA: f64 = 11.6;
const I_R: f64 = 2.25e-5;
const DT: f64 = 0.25;
const SUBSTEPS: usize = 10;
const DRIVE: f64 = 0.08; // the C1 exogenous drive magnitude (raw dV1 units;
// 0.08 is the largest bounded value — 0.1 sends the constant-drive
// double-scroll into sinh runaway, scanned 2026-11-24; disclosed in the
// bench record)
const BLOCK: usize = 200; // condition block length (the PINNED corpus law)

#[inline]
fn double_scroll_rhs(state: &[f64; 3], drive: f64, out: &mut [f64; 3]) {
    let (v1, v2, i) = (state[0], state[1], state[2]);
    let dv = v1 - v2;
    let sinh_term = 2.0 * I_R * (BETA * dv).sinh();
    out[0] = v1 / R1 - dv / R2 - sinh_term + drive;
    out[1] = dv / R2 + sinh_term - i;
    out[2] = v2 - R4 * i;
}

/// RK4 at an arbitrary step size (the drive is block-constant, so the
/// substeps inherit it unchanged).
fn rk4_step_at(state: &mut [f64; 3], drive: f64, dt: f64) {
    let mut k1 = [0.0; 3];
    let mut k2 = [0.0; 3];
    let mut k3 = [0.0; 3];
    let mut k4 = [0.0; 3];
    let mut tmp = [0.0; 3];
    double_scroll_rhs(state, drive, &mut k1);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k1[j];
    }
    double_scroll_rhs(&tmp, drive, &mut k2);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k2[j];
    }
    double_scroll_rhs(&tmp, drive, &mut k3);
    for j in 0..3 {
        tmp[j] = state[j] + dt * k3[j];
    }
    double_scroll_rhs(&tmp, drive, &mut k4);
    for j in 0..3 {
        state[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
}

fn rk4_substepped(state: &mut [f64; 3], drive: f64) {
    let dt_sub = DT / SUBSTEPS as f64;
    for _ in 0..SUBSTEPS {
        rk4_step_at(state, drive, dt_sub);
    }
}

/// Deterministic Box–Muller LCG (no external RNG dep).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0xD1B54A32D192ED03),
        )
    }
    /// Uniform in [−1, 1).
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    }
    /// Standard normal (Box–Muller).
    fn next_normal(&mut self) -> f64 {
        let u1 = ((self.next_unit() + 1.0) * 0.5).max(1e-12);
        let u2 = (self.next_unit() + 1.0) * 0.5;
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// A generated corpus: normalized observations (n×D row-major) + per-sample
/// condition labels.
struct Corpus {
    obs: Vec<f32>,
    cond: Vec<u8>,
    d: usize,
}

/// C1: regime-switching double-scroll (drive blocks) + observation noise.
fn generate_c1(n: usize, noise_seed: u64) -> Corpus {
    // 1. Noiseless seed-0 trajectory — the normalizer reference (fixed so
    //    every noise seed lives in the same normalized frame).
    let raw = integrate_double_scroll(n, 1000, true);
    let (offset, scale) = minmax_normalizer(&raw, 3);
    // 2. Normalize, then add seeded observation noise σ = 0.01 (normalized
    //    units).
    let mut lcg = Lcg::new(noise_seed);
    let mut obs = Vec::with_capacity(raw.len());
    for (i, v) in raw.iter().enumerate() {
        let x = (v - offset[i % 3]) * scale[i % 3];
        obs.push((x + 0.01 * lcg.next_normal()) as f32);
    }
    let cond: Vec<u8> = (0..n).map(|t| ((t / BLOCK) % 2) as u8).collect();
    Corpus { obs, cond, d: 3 }
}

/// C2: pure double-scroll, c ≡ 0 (no noise — the abstain leg).
fn generate_c2(n: usize) -> Corpus {
    let raw = integrate_double_scroll(n, 1000, false);
    let (offset, scale) = minmax_normalizer(&raw, 3);
    let obs: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| ((v - offset[i % 3]) * scale[i % 3]) as f32)
        .collect();
    let cond = vec![0u8; n];
    Corpus { obs, cond, d: 3 }
}

/// C3: 3-channel multi-sine, period 20, c ≡ 0.
fn generate_c3(n: usize) -> Corpus {
    let phases = [0.0f64, 7.0, 13.0];
    let mut obs = Vec::with_capacity(n * 3);
    for t in 0..n {
        for p in phases {
            // sin is already [−1,1]; scale 0.9 to keep off the rails.
            obs.push((0.9 * (2.0 * core::f64::consts::PI * (t as f64 + p) / 20.0).sin()) as f32);
        }
    }
    let cond = vec![0u8; n];
    Corpus { obs, cond, d: 3 }
}

/// Integrate the double-scroll for `n` samples; `switched` adds the DRIVE
/// on odd 200-blocks (the C1 condition-keyed exogenous drive).
fn integrate_double_scroll(n: usize, transient: usize, switched: bool) -> Vec<f64> {
    let mut state: [f64; 3] = [0.1, 0.0, 0.0];
    let drive_at = |t: usize| -> f64 {
        if switched && (t / BLOCK) % 2 == 1 {
            DRIVE
        } else {
            0.0
        }
    };
    for t in 0..transient {
        rk4_substepped(&mut state, drive_at(t));
    }
    let mut out = Vec::with_capacity(n * 3);
    for t in 0..n {
        rk4_substepped(&mut state, drive_at(t));
        out.extend_from_slice(&state);
    }
    out
}

/// Per-coordinate min/max → [−1,1] (the fixture's normalization law).
fn minmax_normalizer(raw: &[f64], d: usize) -> ([f64; 3], [f64; 3]) {
    let mut offset = [0.0f64; 3];
    let mut scale = [1.0f64; 3];
    for j in 0..d {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (i, v) in raw.iter().enumerate() {
            if i % d != j {
                continue;
            }
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[j] = (hi + lo) * 0.5;
        scale[j] = 2.0 / range;
    }
    (offset, scale)
}

// ── Shared fit / eval machinery ─────────────────────────────────────────

/// Feature rows + targets over a corpus row range: row t pairs
/// (expand(ring(obs[..=t])), obs[t+1]). Teacher-forced one-step; the delay
/// ring is D=3, K=4 for every arm (only the basis M differs).
struct Rows {
    features: Vec<f32>, // n_rows × d_h
    targets: Vec<f32>,  // n_rows × d
    cond: Vec<u8>,      // n_rows
}

fn build_rows_m(
    corpus: &Corpus,
    from: usize,
    to: usize,
    d_h: usize,
    expand: fn(&[f32], &mut [f32]),
) -> Rows {
    const K: usize = 4;
    let d = corpus.d;
    // Row t needs history obs[t-K+1..=t] — the first valid row is t = K-1.
    let row_start = from.max(K - 1);
    let n_rows = to.saturating_sub(row_start);
    let mut features = vec![0.0f32; n_rows * d_h];
    let mut targets = Vec::with_capacity(n_rows * d);
    let mut cond = Vec::with_capacity(n_rows);
    let mut ring = DelayRing::<3, K>::new();
    // Prime the ring with the history strictly before `row_start` —
    // for a from=0 split that is obs[0..K-1] (the rows themselves supply
    // obs[K-1] on their first push).
    let prime_from = row_start.saturating_sub(K);
    for t in prime_from..row_start {
        let mut arr = [0.0f32; 3];
        arr.copy_from_slice(&corpus.obs[t * d..(t + 1) * d]);
        ring.push(&arr);
    }
    for (ri, t) in (row_start..to).enumerate() {
        // Push the CURRENT observation first — the flattened state is the
        // history ending AT t (obs[t-K+1..=t]), which predicts obs[t+1].
        let mut cur = [0.0f32; 3];
        cur.copy_from_slice(&corpus.obs[t * d..(t + 1) * d]);
        ring.push(&cur);
        let mut delay = [0.0f32; K * 3];
        assert!(ring.flatten_into(&mut delay), "delay ring must be full");
        expand(&delay, &mut features[ri * d_h..(ri + 1) * d_h]);
        targets.extend_from_slice(&corpus.obs[(t + 1) * d..(t + 2) * d]);
        cond.push(corpus.cond[t]);
    }
    Rows {
        features,
        targets,
        cond,
    }
}

/// M=8 expansion (the pinned shape).
fn expand_m8(delay: &[f32], out: &mut [f32]) {
    feature_expand(delay, &ChebyshevBasis::<8>, out);
}

/// M=13 (A5 wide).
fn expand_m13(delay: &[f32], out: &mut [f32]) {
    feature_expand(delay, &ChebyshevBasis::<13>, out);
}

/// M=7 (A6 private).
fn expand_m7(delay: &[f32], out: &mut [f32]) {
    feature_expand(delay, &ChebyshevBasis::<7>, out);
}

/// Closed-form ridge readout fit: Wout = (ΨᵀΨ + λI)⁻¹ ΨᵀY, returned
/// OUTPUT-major (w[j·d_h + i], f32) — the bank's layout.
fn fit_readout(rows: &Rows, lambda: f64, d_h: usize, d: usize) -> Vec<f32> {
    let n_rows = rows.cond.len();
    let mut gram = vec![0.0f64; d_h * d_h];
    let f_rows = rows.features.chunks_exact(d_h);
    chunked_gram_into(f_rows, &mut gram, lambda, d_h);
    // cov = ΨᵀY (d_h × d, feature-major), then solve to w_t (d_h × d).
    let mut cov = vec![0.0f64; d_h * d];
    for i in 0..n_rows {
        let psi = &rows.features[i * d_h..(i + 1) * d_h];
        for k in 0..d_h {
            let pk = psi[k] as f64;
            for j in 0..d {
                cov[k * d + j] += pk * rows.targets[i * d + j] as f64;
            }
        }
    }
    let mut l = vec![0.0f64; d_h * d_h];
    let mut z = vec![0.0f64; d_h * d];
    let mut w_t = vec![0.0f64; d_h * d];
    ridge_solve_direct_f64(&mut w_t, &mut l, &mut z, &gram, &cov, d_h, d);
    // Transpose to output-major f32.
    let mut w = vec![0.0f32; d * d_h];
    for j in 0..d {
        for k in 0..d_h {
            w[j * d_h + k] = w_t[k * d + j] as f32;
        }
    }
    w
}

/// The G1 fixture's NRMSE (per-coordinate std of truth).
fn nrmse(pred: &[f32], truth: &[f32], dim: usize) -> f32 {
    let n = pred.len() / dim;
    let mut stds = [0.0f32; 8];
    debug_assert!(dim <= stds.len());
    for d in 0..dim {
        let mut mean = 0.0f64;
        for i in 0..n {
            mean += truth[i * dim + d] as f64;
        }
        mean /= n as f64;
        let mut var = 0.0f64;
        for i in 0..n {
            let dx = truth[i * dim + d] as f64 - mean;
            var += dx * dx;
        }
        var /= n as f64;
        stds[d] = var.sqrt() as f32;
    }
    let mut sum = 0.0f32;
    for d in 0..dim {
        let mut err_sq = 0.0f32;
        for i in 0..n {
            let e = pred[i * dim + d] - truth[i * dim + d];
            err_sq += e * e;
        }
        let rmse = (err_sq / n as f32).sqrt();
        sum += rmse / stds[d].max(1e-12);
    }
    sum / dim as f32
}

/// Splits (the PINNED law): fit 0–60 %, gate cal 60–70 %, test 70–100 %.
struct Splits {
    fit_end: usize,
    cal_end: usize,
    test_end: usize,
}

fn splits(n: usize) -> Splits {
    Splits {
        fit_end: n * 6 / 10,
        cal_end: n * 7 / 10,
        test_end: n - 2, // the last row's target needs obs[n-1]
    }
}

// ── The cell ────────────────────────────────────────────────────────────

const D: usize = 3;
const K: usize = 4;
const M: usize = 8;
const D_H: usize = K * D * M; // 96
const LAMBDA: f64 = 1e-4;
const N: usize = 4000;
const MIN_FIT_ROWS: usize = 32; // a condition below this gets no correction

struct ArmResult {
    name: &'static str,
    state_f32: usize,
    /// Per-seed NRMSE (C1: 5 seeds; C2/C3: single run).
    seeds: Vec<f32>,
    train_nrmse: Option<f32>,
}

/// Fit the two keyed corrections on `rows` + `base_res` and add them to a
/// bank (the archetype-u / ridge-v law). Returns the count actually built.
fn add_keyed_corrections(
    bank: &mut KarcConditionedReadout,
    rows: &Rows,
    base_res: &[f32],
    d_h: usize,
    scratch: &mut BankFitScratch,
) -> usize {
    let n_fit = rows.cond.len();
    let mut built = 0usize;
    for cond_key in 0u8..2 {
        let keyed: Vec<usize> = (0..n_fit).filter(|&i| rows.cond[i] == cond_key).collect();
        if keyed.len() < MIN_FIT_ROWS {
            continue;
        }
        let keyed_res: Vec<f32> = keyed
            .iter()
            .flat_map(|&i| base_res[i * D..(i + 1) * D].to_vec())
            .collect();
        let mut u = vec![0.0f32; D];
        if !archetype_direction_into(&keyed_res, D, &mut u) {
            // Near-zero mean residual: author e_0 (the authored leg).
            u = vec![0.0; D];
            u[0] = 1.0;
        }
        let keyed_feats: Vec<f32> = keyed
            .iter()
            .flat_map(|&i| rows.features[i * d_h..(i + 1) * d_h].to_vec())
            .collect();
        let mut v = Vec::new();
        fit_correction_into(&keyed_feats, &keyed_res, &u, LAMBDA, d_h, scratch, &mut v);
        bank.add_correction(cond_key, u, v);
        built += 1;
    }
    built
}

fn run_cell(name: &str, corpora: Vec<Corpus>) -> Vec<ArmResult> {
    let mut arms: Vec<ArmResult> = vec![
        ArmResult {
            name: "A1 base",
            state_f32: D * D_H,
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A2 mux",
            state_f32: D * D_H,
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A3 keyed-mean",
            state_f32: D * D_H + 2 * (D_H + D),
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A3b keyed Σ/√2",
            state_f32: D * D_H + 2 * (D_H + D),
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A4 σ-gated",
            state_f32: D * D_H + 2 * (D_H + D),
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A5 wide M=13",
            state_f32: 13 * K * D * D,
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A6 private M=7 ×2",
            state_f32: 2 * (7 * K * D * D),
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A1c +cond-bias",
            state_f32: D * D_H + 2 * D,
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "A1k per-cond readouts",
            state_f32: 2 * D * D_H,
            seeds: vec![],
            train_nrmse: None,
        },
        ArmResult {
            name: "SeasonalNaive floor",
            state_f32: 0,
            seeds: vec![],
            train_nrmse: None,
        },
    ];

    for corpus in &corpora {
        let sp = splits(corpus.obs.len() / corpus.d);
        let rows_fit = build_rows_m(corpus, 0, sp.fit_end, D_H, expand_m8);
        let rows_cal = build_rows_m(corpus, sp.fit_end, sp.cal_end, D_H, expand_m8);
        let rows_test = build_rows_m(corpus, sp.cal_end, sp.test_end, D_H, expand_m8);
        let d_h5 = 13 * K * D;
        let rows_m13_fit = build_rows_m(corpus, 0, sp.fit_end, d_h5, expand_m13);
        let rows_m13_test = build_rows_m(corpus, sp.cal_end, sp.test_end, d_h5, expand_m13);
        let d_h7 = 7 * K * D;
        let rows_m7_fit = build_rows_m(corpus, 0, sp.fit_end, d_h7, expand_m7);
        let rows_m7_test = build_rows_m(corpus, sp.cal_end, sp.test_end, d_h7, expand_m7);

        // ── A1 base fit + residuals ──
        let base_w = fit_readout(&rows_fit, LAMBDA, D_H, D);
        let n_fit = rows_fit.cond.len();
        let base_res = {
            let mut res = vec![0.0f32; n_fit * D];
            let mut pred = [0.0f32; D];
            for i in 0..n_fit {
                matvec(
                    &base_w,
                    &rows_fit.features[i * D_H..(i + 1) * D_H],
                    &mut pred,
                );
                for j in 0..D {
                    res[i * D + j] = rows_fit.targets[i * D + j] - pred[j];
                }
            }
            res
        };
        arms[0].train_nrmse = Some(train_nrmse_of(&rows_fit, &base_w, D_H));

        // ── A4 the σ-gated bank + its cal-slice warm-up ──
        let mut bank = KarcConditionedReadout::new(base_w.clone(), D, D_H, GateConfig::default());
        let mut scratch = BankFitScratch::with_capacity(D_H);
        let _built = add_keyed_corrections(&mut bank, &rows_fit, &base_res, D_H, &mut scratch);
        {
            let mut scratch_obs = [0.0f32; D];
            for i in 0..rows_cal.cond.len() {
                bank.observe(
                    &rows_cal.features[i * D_H..(i + 1) * D_H],
                    rows_cal.cond[i],
                    &rows_cal.targets[i * D..(i + 1) * D],
                    &mut scratch_obs,
                );
            }
        }

        // ── A1c per-condition bias ──
        let mut bias = [[0.0f32; D]; 2];
        let mut bias_seen = [0usize; 2];
        for i in 0..n_fit {
            let c = rows_fit.cond[i] as usize;
            for j in 0..D {
                bias[c][j] += base_res[i * D + j];
            }
            bias_seen[c] += 1;
        }
        for (c, seen) in bias_seen.iter().enumerate() {
            if *seen > 0 {
                for b in bias[c].iter_mut() {
                    *b /= *seen as f32;
                }
            }
        }

        // ── A1k per-condition readouts ──
        let w_per_cond: Vec<Option<Vec<f32>>> = (0u8..2)
            .map(|cond_key| keyed_readout_fit(&rows_fit, cond_key, D_H, LAMBDA))
            .collect();

        // ── A5 wide + A6 private fits ──
        let w_wide = fit_readout(&rows_m13_fit, LAMBDA, d_h5, D);
        let w_private: Vec<Option<Vec<f32>>> = (0u8..2)
            .map(|cond_key| keyed_readout_fit(&rows_m7_fit, cond_key, d_h7, LAMBDA))
            .collect();

        // ── Seasonal-naive floor forecaster (per channel; m = the corpus's
        //    period — 20 on C3, 1 elsewhere — pure anchor, pool_weight 0) ──
        let is_periodic = name.contains("C3");
        let mut seasonal: Vec<_> = (0..D)
            .map(|_| {
                if is_periodic {
                    SeasonalPoolForecaster::new(N, 20, 0.0, 0.0)
                } else {
                    seasonal_naive_floor(N)
                }
            })
            .collect();
        {
            // Prime with the truth prefix strictly before the test walk
            // (no peeking into the test window).
            for t in 0..sp.cal_end {
                for (j, s) in seasonal.iter_mut().enumerate() {
                    s.observe(corpus.obs[t * D + j]);
                }
            }
        }
        let mut gate = KarcRegimeGate::new();

        // ── Test walk ──
        let n_test = rows_test.cond.len();
        let mut p = vec![vec![0.0f32; n_test * D]; 9]; // a1 a2 a3 a3b a4 a5 a6 a1c a1k
        let mut p_floor = vec![0.0f32; n_test * D];
        {
            let mut o = [0.0f32; D];
            let mut seas = [0.0f32; D];
            let sqrt2 = 2.0f32.sqrt();
            for i in 0..n_test {
                let f = &rows_test.features[i * D_H..(i + 1) * D_H];
                let c = rows_test.cond[i];
                let truth_t = i + sp.cal_end; // predicts obs[truth_t + 1]
                let truth = &corpus.obs[(truth_t + 1) * D..(truth_t + 2) * D];

                // A4 (gated) — first, while `o` is fresh.
                bank.apply_into(f, c, &mut o);
                p[4][i * D..(i + 1) * D].copy_from_slice(&o);

                // A1 base only.
                bank.base_predict_into(f, &mut o);
                p[0][i * D..(i + 1) * D].copy_from_slice(&o);
                let base_pred = o;

                // A3 keyed mean (gate forced open) + A3b Σ/√2 — computed
                // from the bank's public correction data (the fixed-gate
                // twin must not depend on EWMA state).
                bank.apply_fixed(f, c, 1.0, &mut o);
                p[2][i * D..(i + 1) * D].copy_from_slice(&o);
                bank.apply_fixed(f, c, 1.0 / sqrt2, &mut o);
                p[3][i * D..(i + 1) * D].copy_from_slice(&o);

                // A1c keyed bias.
                o = base_pred;
                for j in 0..D {
                    o[j] += bias[c as usize][j];
                }
                p[7][i * D..(i + 1) * D].copy_from_slice(&o);

                // A1k keyed full readout (base fallback when a condition
                // never got enough fit rows).
                match &w_per_cond[c as usize] {
                    Some(w) => matvec(w, f, &mut o),
                    None => o = base_pred,
                }
                p[8][i * D..(i + 1) * D].copy_from_slice(&o);

                // A5 wide.
                let fw = &rows_m13_test.features[i * d_h5..(i + 1) * d_h5];
                matvec(&w_wide, fw, &mut o);
                p[5][i * D..(i + 1) * D].copy_from_slice(&o);

                // A6 private forecast mean (base fallback when the
                // condition has no private fit — single-condition corpora).
                let f7 = &rows_m7_test.features[i * d_h7..(i + 1) * d_h7];
                match &w_private[c as usize] {
                    Some(w) => matvec(w, f7, &mut o),
                    None => o = base_pred,
                }
                p[6][i * D..(i + 1) * D].copy_from_slice(&o);

                // Floor + A2 mux seasonal forecast.
                for j in 0..D {
                    seasonal[j].forecast_into(f, 1, &mut seas[j]);
                }
                p_floor[i * D..(i + 1) * D].copy_from_slice(&seas);
                // A2: the gate picks between base-KARC and seasonal per
                // tick, using residuals accumulated strictly before this
                // tick (cold start routes Seasonal — the floor).
                let verdict = gate.decide();
                let pick = match verdict.preferred {
                    KarcRegime::Karc => &base_pred,
                    KarcRegime::Seasonal => &seas,
                };
                p[1][i * D..(i + 1) * D].copy_from_slice(pick);
                // Then feed the realized residuals (no peeking: both
                // forecasts for THIS tick are already committed).
                let mut rk = 0.0f32;
                let mut rs = 0.0f32;
                for j in 0..D {
                    let ek = base_pred[j] - truth[j];
                    let es = seas[j] - truth[j];
                    rk += ek * ek;
                    rs += es * es;
                }
                gate.observe_residuals(rk, rs);
                for j in 0..D {
                    seasonal[j].observe(truth[j]);
                }
            }
        }
        let truth_test: Vec<f32> = ((sp.cal_end + 1)..=sp.test_end)
            .flat_map(|t| corpus.obs[t * D..(t + 1) * D].to_vec())
            .collect();

        for (idx, slot) in [0usize, 1, 2, 3, 4, 5, 6, 7, 8].into_iter().enumerate() {
            arms[idx].seeds.push(nrmse(&p[slot], &truth_test, D));
        }
        arms[9].seeds.push(nrmse(&p_floor, &truth_test, D));
        arms[5].train_nrmse = Some(train_nrmse_of(&rows_m13_fit, &w_wide, d_h5));
        arms[8].train_nrmse = Some(percond_train_nrmse(&rows_fit, &w_per_cond, D_H));
    }

    arms
}

/// A keyed readout fit on one condition's fit rows (A1k / A6 helper);
/// `None` when the condition never reaches `MIN_FIT_ROWS`.
fn keyed_readout_fit(rows: &Rows, cond_key: u8, d_h: usize, lambda: f64) -> Option<Vec<f32>> {
    let keyed: Vec<usize> = (0..rows.cond.len())
        .filter(|&i| rows.cond[i] == cond_key)
        .collect();
    if keyed.len() < MIN_FIT_ROWS {
        return None;
    }
    let feats: Vec<f32> = keyed
        .iter()
        .flat_map(|&i| rows.features[i * d_h..(i + 1) * d_h].to_vec())
        .collect();
    let tgts: Vec<f32> = keyed
        .iter()
        .flat_map(|&i| rows.targets[i * D..(i + 1) * D].to_vec())
        .collect();
    let keyed_rows = Rows {
        features: feats,
        targets: tgts,
        cond: vec![cond_key; keyed.len()],
    };
    Some(fit_readout(&keyed_rows, lambda, d_h, D))
}

fn matvec(w: &[f32], f: &[f32], out: &mut [f32]) {
    let d_h = f.len();
    for (j, o) in out.iter_mut().enumerate() {
        let row = &w[j * d_h..(j + 1) * d_h];
        let mut acc = 0.0f32;
        for k in 0..d_h {
            acc += row[k] * f[k];
        }
        *o = acc;
    }
}

fn train_nrmse_of(rows: &Rows, w: &[f32], d_h: usize) -> f32 {
    let n = rows.cond.len();
    let mut p = Vec::with_capacity(n * D);
    let mut o = [0.0f32; D];
    for i in 0..n {
        matvec(w, &rows.features[i * d_h..(i + 1) * d_h], &mut o);
        p.extend_from_slice(&o);
    }
    nrmse(&p, &rows.targets, D)
}

fn percond_train_nrmse(rows: &Rows, w: &[Option<Vec<f32>>], d_h: usize) -> f32 {
    let n = rows.cond.len();
    let mut p = vec![0.0f32; n * D];
    let mut o = [0.0f32; D];
    for i in 0..n {
        let c = rows.cond[i] as usize;
        if let Some(w) = &w[c] {
            matvec(w, &rows.features[i * d_h..(i + 1) * d_h], &mut o);
        }
        p[i * D..(i + 1) * D].copy_from_slice(&o);
    }
    nrmse(&p, &rows.targets, D)
}

fn mean_se(rs: &[f32]) -> (f32, f32) {
    let n = rs.len() as f32;
    let mean = rs.iter().sum::<f32>() / n;
    let var = rs.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n;
    (mean, (var / n).sqrt())
}

fn rs_vs(arm: &ArmResult, a1: &ArmResult) -> Vec<f32> {
    arm.seeds
        .iter()
        .zip(a1.seeds.iter())
        .map(|(x, b)| 100.0 * (x - b) / b)
        .collect()
}

fn delta_of(se: f32) -> f32 {
    (1.0f32).max(2.5 * se)
}

/// Per-seed relative deltas + the decision-rule verdicts. Returns the G1
/// conjunction (A4 WIN vs A1 AND A5 not WIN vs A4) — the record's headline.
fn report(name: &str, arms: &[ArmResult], seed_count: usize) -> bool {
    let a1 = &arms[0];
    let a3 = &arms[2];
    let a4 = &arms[4];
    let a5 = &arms[5];
    let a1c = &arms[7];
    println!("\n══ {name} ({seed_count} seed(s)) ══");
    println!(
        "{:<24} {:>6}  {:>12}  {:>14}  {:>8}",
        "arm", "f32", "NRMSE mean", "r̄ vs A1 (pp)", "train"
    );
    for arm in arms {
        let (m, se) = mean_se(&rs_vs(arm, a1));
        let mean_nrmse = arm.seeds.iter().sum::<f32>() / arm.seeds.len() as f32;
        println!(
            "{:<24} {:>6}  {:>12.4}  {:>9.3} ±{:<5.3}  {}",
            arm.name,
            arm.state_f32,
            mean_nrmse,
            m,
            se,
            arm.train_nrmse
                .map(|t| format!("{t:.4}"))
                .unwrap_or_else(|| "—".into())
        );
        if arm.seeds.len() > 1 {
            let seeds_str = arm
                .seeds
                .iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!("    seeds: {seeds_str}");
        }
    }
    let (m4, se4) = mean_se(&rs_vs(a4, a1));
    let (m5v, se5) = mean_se(&rs_vs(a5, a4));
    let (m1c, _) = mean_se(&rs_vs(a1c, a1));
    let (m3, _) = mean_se(&rs_vs(a3, a1));
    let d4 = delta_of(se4);
    let a4_win = m4 < -d4;
    let a4_loss = m4 > d4;
    let a5_win_vs_a4 = m5v < -delta_of(se5);
    let attrib_vs_bias_loss = (m4 - m1c) > d4;
    println!("── decision rule ──");
    println!(
        "A4 vs A1:  r̄ {m4:+.3} ± {se4:.3} δ {d4:.3} → {}",
        if a4_win {
            "WIN"
        } else if a4_loss {
            "LOSS"
        } else {
            "TIE"
        }
    );
    println!(
        "A5 vs A4:  r̄ {m5v:+.3} ± {se5:.3} → {}",
        if a5_win_vs_a4 {
            "WIN — demote-to-null (branch claim fails)"
        } else {
            "not WIN ✓"
        }
    );
    println!(
        "Attribution: A4−A1c r̄ {:+.3} → keyed win {} the plain keyed-bias control",
        m4 - m1c,
        if attrib_vs_bias_loss {
            "LOSES to"
        } else if a4_win && (m4 - m1c) < -d4 {
            "beats"
        } else {
            "ties"
        }
    );
    println!(
        "Promotion-bearing A3 vs A1: r̄ {m3:+.3} → {}",
        if m3 < -delta_of(0.0) {
            "WIN (fixed gate carries)"
        } else {
            "not WIN"
        }
    );
    let g1 = a4_win && !a5_win_vs_a4;
    println!("G1 (A4 WIN ∧ A5 not WIN): {g1}");
    g1
}

/// C2/C3 G3 leg: A4 not LOSS vs A1 (single seed → δ floor 1.0 pp,
/// disclosed). Prints the floor row beside (G5 on C3).
fn report_g3(name: &str, arms: &[ArmResult]) -> bool {
    let a1 = &arms[0];
    let a2 = &arms[1];
    let a4 = &arms[4];
    let floor = &arms[9];
    let r = 100.0 * (a4.seeds[0] - a1.seeds[0]) / a1.seeds[0];
    let not_loss = r <= 1.0;
    println!(
        "\n══ {name} (G3 leg) ══\nA1 {:.4} · A2 {:.4} · A4 {:.4} · floor {:.4} → A4 r {r:+.3} pp → {}",
        a1.seeds[0],
        a2.seeds[0],
        a4.seeds[0],
        floor.seeds[0],
        if not_loss { "not LOSS ✓" } else { "LOSS ✗" }
    );
    not_loss
}

fn a5_wins_vs_a4(a4: &ArmResult, a5: &ArmResult) -> bool {
    let (m, se) = mean_se(&rs_vs(a5, a4));
    m < -delta_of(se)
}

// ── G2: the interleaved wall cell (release posture; #[ignore]d) ─────────
//
// A4/A1 forecast-step wall ≤ 1.5× at the corpus shape (D=3, d_h=96) AND an
// HLA-ish shape (D=8, d_h=192). Interleaved (a, b) chunk pairs, median
// across pairs — the sequential-A/B box-noise law. Gates ARMED (the
// adversarial posture: every correction fires every tick).

#[test]
#[ignore]
fn g2_wall_interleaved() {
    for (label, d, d_h, n_corr) in [
        ("corpus D=3 d_h=96", 3usize, 96usize, 2usize),
        ("HLA-ish D=8 d_h=192", 8, 192, 2),
    ] {
        let mut lcg = Lcg(0xB000_0000_0000_0001u64.wrapping_add(d as u64));
        let base: Vec<f32> = (0..d * d_h)
            .map(|_| lcg.next_unit() as f32 * 0.01)
            .collect();
        let mut bank = KarcConditionedReadout::new(base, d, d_h, GateConfig::default());
        for b in 0u8..n_corr as u8 {
            let mut u: Vec<f32> = (0..d).map(|_| lcg.next_unit() as f32).collect();
            let norm = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut u {
                *x /= norm;
            }
            let mut v: Vec<f32> = (0..d_h).map(|_| lcg.next_unit() as f32 * 0.5).collect();
            // Pin one coefficient so the keyed coef is bounded away from 0
            // (a coef ≈ 0 leaves ewma_base_sq ≈ 0 and the gate stays shut).
            v[0] = 1.0;
            bank.add_correction(b, u, v);
        }
        let mut feats: Vec<f32> = (0..d_h).map(|_| lcg.next_unit() as f32).collect();
        feats[0] = 1.0;
        let mut truth = vec![0.0f32; d];
        let mut scratch = vec![0.0f32; d];
        // Arm the gates (feed keyed samples until σ > 0.9 for both).
        for _ in 0..512 {
            for b in 0u8..n_corr as u8 {
                bank.base_predict_into(&feats, &mut truth);
                let coef: f32 = feats
                    .iter()
                    .zip(bank.correction(b as usize).v.iter())
                    .map(|(a, x)| a * x)
                    .sum();
                for (j, u) in bank.correction(b as usize).u.iter().enumerate() {
                    truth[j] += coef * u;
                }
                bank.observe(&feats, b, &truth, &mut scratch);
            }
        }
        assert!(bank.gate_sigma(0) > 0.9 && bank.gate_sigma(1) > 0.9);
        let mut out = vec![0.0f32; d];
        // Interleaved chunks: 200 pairs × 128 calls per arm per chunk.
        const CHUNK: usize = 128;
        const PAIRS: usize = 200;
        let mut ratios = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            let t0 = std::time::Instant::now();
            for _ in 0..CHUNK {
                bank.base_predict_into(&feats, &mut out);
            }
            let a = t0.elapsed().as_nanos() as f64;
            let t1 = std::time::Instant::now();
            for _ in 0..CHUNK {
                bank.apply_into(&feats, 0, &mut out);
            }
            let b = t1.elapsed().as_nanos() as f64;
            ratios.push(b / a.max(1.0));
        }
        ratios.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = ratios[PAIRS / 2];
        println!("{label}: median wall ratio A4/A1 = {med:.3}× (bar ≤ 1.5×) [gates armed]");
        assert!(
            med <= 1.5,
            "{label}: wall ratio {med:.3}× exceeds the 1.5× bar"
        );
    }
}

// ── The test ────────────────────────────────────────────────────────────

#[test]
fn issue_1239_conditioned_bank_cell() {
    // The matched-bytes ledger tripwire (the issue's numbers).
    let mut bank = KarcConditionedReadout::new(vec![0.0; D * D_H], D, D_H, GateConfig::default());
    assert_eq!(bank.state_f32(), 288);
    bank.add_correction(0, vec![0.0; D], vec![0.0; D_H]);
    bank.add_correction(1, vec![0.0; D], vec![0.0; D_H]);
    assert_eq!(bank.state_f32(), 486);

    // C1: 5 observation-noise seeds.
    let c1: Vec<Corpus> = (0..5).map(|s| generate_c1(N, 0xA000 + s as u64)).collect();
    let c1_arms = run_cell("C1 regime-switching", c1);
    let g1 = report("C1", &c1_arms, 5);

    // C2: pure double-scroll (c ≡ 0) — corrections must abstain.
    let c2 = vec![generate_c2(N)];
    let c2_arms = run_cell("C2 pure double-scroll", c2);
    let g3_c2 = report_g3("C2", &c2_arms);

    // C3: periodic multi-sine — G3 + the G5 floor row.
    let c3 = vec![generate_c3(N)];
    let c3_arms = run_cell("C3 periodic multi-sine", c3);
    let g3_c3 = report_g3("C3", &c3_arms);

    println!("\n══════════ ISSUE 1239 SUMMARY ══════════");
    println!("G1 (A4 WIN vs A1 on C1 AND A5 not WIN vs A4): {g1}");
    println!("G3 (A4 not LOSS vs A1 on C2): {g3_c2}");
    println!("G3 (A4 not LOSS vs A1 on C3): {g3_c3}");

    // G3 is a HARD gate (abstain where the base is good).
    assert!(
        g3_c2 && g3_c3,
        "G3 abstain failed — the bank must not hurt a good base"
    );
    // The demote-to-null shape panics: if the wide fit beats the bank, the
    // branch claim died on this host.
    let a4 = &c1_arms[4];
    let a5 = &c1_arms[5];
    assert!(
        !a5_wins_vs_a4(a4, a5),
        "demote-to-null: A5 wide beat A4 — the branch claim FAILED on this host"
    );
    if !g1 {
        println!(
            "NOTE: G1 not won on C1 — the bench record adjudicates TIE/LOSS honestly (no promotion)."
        );
    }
}
