//! Bench 945 — Issue 948 T1 GOAT + T2 paired read (Terminal-RSI modelless
//! extractions; riir-train Research 477 / arXiv:2610.22192).
//!
//! The falsifiable question (T1): does family-calibrated partial credit
//! `R = max(P / D_f, 0)` — with `D_f` the FROZEN nearest-rank p90 of the
//! family's raw check-pass counts — rank arms by true quality where the
//! incumbent raw-P ordering ranks them by SCALE?
//!
//! Controlled harness (bench_684 shape: all seeded, deterministic, modelless):
//! - `F = 10` families (hint templates / domains), check-count scales
//!   `K_f ∈ {4..80}` — a 20× spread, the premise that raw counts carry family
//!   scale, not quality.
//! - True per-family quality `θ_f`, stratified over `[0.15, 0.85]` and
//!   SHUFFLED against `K_f` each seed — the adjudicated ground truth, known
//!   by construction, independent of scale (the adversarial pairing).
//! - Raw scores `P ~ Binomial(K_f, θ_f)`: reference slice (N=128, the freeze
//!   corpus) and a fresh held-out slice (N=128, the scoring corpus) — the
//!   freeze never sees the scores it is judged on (G1c).
//! - Arms: **A0 raw-P** — rank families by mean held-out raw P (the incumbent
//!   hinge's ordering). **A1 FDR** — freeze `D_f` on the reference slice, rank
//!   by mean held-out `credit(f, P)`.
//!
//! Gates:
//! - **G1a ordering**: mean Spearman(A1, θ) ≥ pinned floor AND beats A0 by a
//!   pinned margin in ≥ 14/16 seeds (dominance, not perfect recovery — the
//!   estimator saturates at small K).
//! - **G1b premise (the inversion is real)**: mean Spearman(A0, θ) ≤ pinned
//!   ceiling — raw-P is genuinely corrupted, not marginally worse.
//! - **G1c held-out discipline**: structural — divisors come only from the
//!   reference slice; the gate reads held-out means only (enforced by
//!   construction here and by `FdrCalibration::freeze`'s read-only shape).
//! - **G2 stability**: bootstrap-resample the reference corpus B=200×,
//!   refreeze, re-rank the SAME held-out means: per-family divisor CV and
//!   ordering Kendall τ under pinned bounds.
//! - **G3 selection**: top-3-by-A1 never worse than top-3-by-A0 (16/16
//!   non-loss) + aggregate margin — top-3 quality ties constantly at 10
//!   families, so never-worse is the honest gate.
//! - **G4 zero-alloc scoring**: CountingAllocator (bench_013 pattern) around
//!   the `credit()` hot loop AND the calibrated absorb loop — 0 allocations
//!   in steady state.
//! - **T2 paired read** (spr_order): selection by the f32 `spr_reward` vs the
//!   canonical `SprKey` pair ordering over a 2 000-candidate corpus. Law:
//!   membership flips may occur ONLY inside exact-`spr_reward` tie groups
//!   (the documented f32 rounding corner); the canonical top-k never excludes
//!   a passed candidate for a failure. `mass_on_failures` reported (not
//!   gated — it is the eval-board diagnostic).
//!
//! Scope (honest): a controlled harness validates the MECHANISM (scale
//! corruption is real; frozen per-family normalization recovers the quality
//! ordering and is stable under resampling), not real-template superiority —
//! no adjudicated real-template corpus exists in-repo yet; when one lands,
//! re-run this bench's arms against it unchanged.

#![cfg(all(feature = "fdr_credit", feature = "spr_order"))]

use katgpt_pruners::g_zero::fdr_credit::FdrCalibration;
use katgpt_pruners::spr_order::{SprKey, mass_on_failures, spr_reward};

// ── Harness constants ────────────────────────────────────────────────

const FAMILIES: usize = 10;
/// Check-count scales: a 20× spread across hint templates / domains.
const K_SCALE: [u32; FAMILIES] = [4, 5, 8, 10, 16, 20, 32, 40, 64, 80];
/// Reference (freeze) + held-out (scoring) slice sizes. N=128 gives
/// tail support 128 − 116 = 12 ≥ MIN_TAIL_SUPPORT(10) at nearest-rank p90.
const N_REF: usize = 128;
const N_HOLD: usize = 128;
const SEEDS: [u64; 16] = [
    42, 1337, 20260827, 7, 91, 31337, 2024, 65537, 7919, 104729, 1299709, 15485863, 32452843,
    49979687, 67867967, 86028121,
];
const BOOTSTRAP_B: usize = 200;
const TOP_K: usize = 3;
const N_CANDIDATES: usize = 2000;

// ── Gates (pinned bounds — adjusted ONCE to the first measured run with
//    margin, then frozen; measured values in .benchmarks/945_issue948_fdr_spr_goat.md.
//    The T1 estimator honestly saturates at small K (a 4-check family's p90
//    divisor compresses its credit toward the ceiling), so the ordering gate
//    pins DOMINANCE over raw-P, not near-perfect recovery) ────────────────

/// G1a: FDR ordering must track θ at least this well (Spearman, aggregate).
const G1A_FDR_SPEARMAN_FLOOR: f64 = 0.55;
/// G1a: FDR must beat raw-P by at least this margin (mean Spearman delta).
const G1A_MARGIN: f64 = 0.20;
/// G1a: per-seed wins needed (of 16).
const G1A_MIN_WINS: u32 = 14;
/// G1b: raw-P ordering must be genuinely corrupted (mean Spearman ceiling).
const G1B_RAW_SPEARMAN_CEIL: f64 = 0.60;
/// G2: bootstrap divisor coefficient of variation bounds (mean / max).
const G2_CV_MEAN_CEIL: f64 = 0.10;
const G2_CV_MAX_CEIL: f64 = 0.40;
/// G2: bootstrap ordering agreement with the full-freeze ordering (Kendall τ).
const G2_TAU_FLOOR: f64 = 0.80;
/// G3: FDR's top-3 must never be WORSE than raw-P's (of 16 seeds) — top-3
/// quality ties constantly at 10 families, so the honest gate is
/// never-worse + aggregate margin, not strict win-rate.
const G3_MIN_NON_LOSS: u32 = 16;
/// G3: aggregate top-3 mean-θ margin.
const G3_MARGIN: f64 = 0.02;

// ── xorshift64* (bench_684 convention — platform-independent) ────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Binomial(K, p) by direct draws (K ≤ 80 — cheap, exact).
    fn binomial(&mut self, k: u32, p: f32) -> u32 {
        let mut hits = 0u32;
        for _ in 0..k {
            if self.next_f32() < p {
                hits += 1;
            }
        }
        hits
    }
}

// ── Rank statistics ──────────────────────────────────────────────────

/// Average ranks (0-based, ties share the mean rank) under the total float
/// order — Spearman needs this when discrete scores tie.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let mean = (i + j) as f64 / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = mean;
        }
        i = j + 1;
    }
    ranks
}

/// Spearman correlation between two score vectors (average-rank form).
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let ra = average_ranks(a);
    let rb = average_ranks(b);
    let ma: f64 = ra.iter().sum::<f64>() / n as f64;
    let mb: f64 = rb.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = ra[i] - ma;
        let y = rb[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Kendall τ-a between two permutations given as score vectors (higher score
/// = better rank; ties counted as concordant-neutral — 10 items, cheap).
fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = (a[i] - a[j]).total_cmp(&0.0);
            let db = (b[i] - b[j]).total_cmp(&0.0);
            if da == std::cmp::Ordering::Equal || db == std::cmp::Ordering::Equal {
                continue;
            }
            if da == db {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total = concordant + discordant;
    if total == 0 {
        return 1.0;
    }
    (concordant - discordant) as f64 / total as f64
}

// ── Per-seed harness run ─────────────────────────────────────────────

struct SeedResult {
    theta: [f64; FAMILIES],
    /// Mean held-out raw P per family (A0's arm scores).
    raw_mean: [f64; FAMILIES],
    /// Mean held-out credit per family (A1's arm scores, frozen divisors).
    fdr_mean: [f64; FAMILIES],
    /// Bootstrap divisor CVs (per family).
    boot_cv: [f64; FAMILIES],
    /// Bootstrap ordering Kendall τ vs the full-freeze ordering (per draw).
    boot_taus: Vec<f64>,
    /// Divisor CV mean + ordering τ mean for this seed (gate aggregates).
    seed_cv_mean: f64,
    seed_tau_mean: f64,
}

fn run_seed(seed: u64) -> SeedResult {
    let mut rng = Rng::new(seed);

    // θ stratified over [0.15, 0.85], shuffled against the K scale.
    let mut theta = [0.0f64; FAMILIES];
    for (i, t) in theta.iter_mut().enumerate() {
        *t = 0.15 + 0.70 * (i as f64 + 0.5) / FAMILIES as f64;
    }
    for i in (1..FAMILIES).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        theta.swap(i, j);
    }

    // Reference slice (freeze corpus) + held-out slice (scoring corpus).
    // Independent RNG streams per slice so holdout ⊥ freeze by construction.
    let mut ref_rng = Rng::new(seed ^ 0xA5A5_0000_0000_0001);
    let mut hold_rng = Rng::new(seed ^ 0x5A5A_0000_0000_0002);
    let mut ref_corpus: Vec<Vec<f32>> = Vec::with_capacity(FAMILIES);
    let mut hold_corpus: Vec<Vec<f32>> = Vec::with_capacity(FAMILIES);
    for f in 0..FAMILIES {
        let k = K_SCALE[f];
        let p = theta[f] as f32;
        let mut r = Vec::with_capacity(N_REF);
        for _ in 0..N_REF {
            r.push(ref_rng.binomial(k, p) as f32);
        }
        ref_corpus.push(r);
        let mut h = Vec::with_capacity(N_HOLD);
        for _ in 0..N_HOLD {
            h.push(hold_rng.binomial(k, p) as f32);
        }
        hold_corpus.push(h);
    }

    // Freeze (T1's primitive — refuses thin tails; N=128 ⇒ support 12).
    let ref_slices: Vec<(u64, &[f32])> = ref_corpus
        .iter()
        .enumerate()
        .map(|(f, v)| (f as u64, v.as_slice()))
        .collect();
    let cal = FdrCalibration::freeze(ref_slices)
        .unwrap_or_else(|e| panic!("freeze refused at seed {seed}: {e}"));

    // Arm scores on the held-out slice only.
    let mut raw_mean = [0.0f64; FAMILIES];
    let mut fdr_mean = [0.0f64; FAMILIES];
    for f in 0..FAMILIES {
        let sum_raw: f64 = hold_corpus[f].iter().map(|&v| v as f64).sum::<f64>() / N_HOLD as f64;
        let sum_fdr: f64 = hold_corpus[f]
            .iter()
            .map(|&v| cal.credit(f as u64, v).unwrap_or(0.0) as f64)
            .sum::<f64>()
            / N_HOLD as f64;
        raw_mean[f] = sum_raw;
        fdr_mean[f] = sum_fdr;
    }

    // G2 — bootstrap: resample each family's reference corpus with
    // replacement, refreeze, re-rank the SAME held-out means.
    let full_order = fdr_mean;
    let mut boot_cv = [0.0f64; FAMILIES];
    let mut boot_taus = Vec::with_capacity(BOOTSTRAP_B);
    let mut boot_divisors = vec![[0.0f64; FAMILIES]; BOOTSTRAP_B];
    let mut boot_rng = Rng::new(seed ^ 0x0B05_57A9_0000_0003);
    for row in boot_divisors.iter_mut().take(BOOTSTRAP_B) {
        let mut boot_rows: Vec<(u64, Vec<f32>)> = Vec::with_capacity(FAMILIES);
        for (f, samples) in ref_corpus.iter().enumerate() {
            let mut s = Vec::with_capacity(N_REF);
            for _ in 0..N_REF {
                let idx = (boot_rng.next_u64() % N_REF as u64) as usize;
                s.push(samples[idx]);
            }
            boot_rows.push((f as u64, s));
        }
        let boot_slices: Vec<(u64, &[f32])> =
            boot_rows.iter().map(|(f, v)| (*f, v.as_slice())).collect();
        let boot_cal = FdrCalibration::freeze(boot_slices)
            .unwrap_or_else(|e| panic!("bootstrap freeze refused at seed {seed}: {e}"));
        let mut boot_scores = [0.0f64; FAMILIES];
        for f in 0..FAMILIES {
            let d = boot_cal.divisor(f as u64).unwrap();
            row[f] = d as f64;
            boot_scores[f] =
                hold_corpus[f].iter().map(|&v| (v / d) as f64).sum::<f64>() / N_HOLD as f64;
        }
        boot_taus.push(kendall_tau(&full_order, &boot_scores));
    }
    for f in 0..FAMILIES {
        let col: Vec<f64> = boot_divisors.iter().map(|row| row[f]).collect();
        let mean = col.iter().sum::<f64>() / BOOTSTRAP_B as f64;
        let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / BOOTSTRAP_B as f64;
        boot_cv[f] = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
    }

    SeedResult {
        theta,
        raw_mean,
        fdr_mean,
        boot_cv,
        seed_cv_mean: mean(&boot_cv),
        seed_tau_mean: mean(&boot_taus),
        boot_taus,
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

// ── T2 candidate corpus ──────────────────────────────────────────────

struct Candidate {
    passed: bool,
    passes: f32,
    divisor: f32,
}

/// Build the paired-read corpus: sparse verified passes (`P == K_f` — all
/// checks green), dense partial credit everywhere else, per-family frozen
/// divisors from the same freeze shape as the T1 harness, plus a 1% stress
/// band with extreme pass/divisor ratios that reaches the documented f32
/// rounding corner (squash → exactly 1.0).
fn build_corpus(seed: u64) -> Vec<Candidate> {
    let mut rng = Rng::new(seed ^ 0xC0DE_0000_0000_0004);
    // Divisors from a mid-θ freeze so realistic ratios appear.
    let theta: [f64; FAMILIES] = [0.2, 0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.55, 0.40, 0.31];
    let mut ref_rows: Vec<(u64, Vec<f32>)> = Vec::with_capacity(FAMILIES);
    for (f, &k) in K_SCALE.iter().enumerate() {
        let mut s = Vec::with_capacity(N_REF);
        for _ in 0..N_REF {
            s.push(rng.binomial(k, theta[f] as f32) as f32);
        }
        ref_rows.push((f as u64, s));
    }
    let slices: Vec<(u64, &[f32])> = ref_rows.iter().map(|(f, v)| (*f, v.as_slice())).collect();
    let cal = FdrCalibration::freeze(slices).expect("corpus freeze");

    let mut cands = Vec::with_capacity(N_CANDIDATES);
    for i in 0..N_CANDIDATES {
        let f = (rng.next_u64() % FAMILIES as u64) as usize;
        let k = K_SCALE[f];
        let d = cal.divisor(f as u64).unwrap();
        if i % 100 == 0 {
            // Stress band: ratios at and beyond the 2^24 rounding corner.
            let passes = match i % 300 {
                0 => d * (1u64 << 25) as f32, // squash rounds to exactly 1.0
                100 => d * (1u64 << 20) as f32,
                _ => 0.0, // zero passes — the corner's other side
            };
            cands.push(Candidate {
                passed: false,
                passes,
                divisor: d,
            });
        } else {
            let p = rng.binomial(k, theta[f] as f32);
            cands.push(Candidate {
                passed: p == k,
                passes: p as f32,
                divisor: d,
            });
        }
    }
    cands
}

fn main() {
    println!("=== Bench 945 — Issue 948 T1 GOAT + T2 paired read (FDR credit + SPR order) ===\n");

    // ── T1: G1a/G1b/G3 + G2 over seeds (one pass — G2 aggregates ride
    //    the same SeedResults) ────────────────────────────────────────
    let mut sum_spear_raw = 0.0f64;
    let mut sum_spear_fdr = 0.0f64;
    let mut wins_fdr = 0u32;
    let mut sum_q_raw = 0.0f64;
    let mut sum_q_fdr = 0.0f64;
    let mut wins_q = 0u32;
    let mut nonloss_q = 0u32;
    let mut all_cvs: Vec<f64> = Vec::new();
    let mut all_taus: Vec<f64> = Vec::new();

    for &seed in &SEEDS {
        let r = run_seed(seed);
        let theta: Vec<f64> = r.theta.to_vec();

        let s_raw = spearman(&r.raw_mean, &theta);
        let s_fdr = spearman(&r.fdr_mean, &theta);
        sum_spear_raw += s_raw;
        sum_spear_fdr += s_fdr;
        if s_fdr > s_raw {
            wins_fdr += 1;
        }

        // G3 — top-3 selection quality.
        let pick = |scores: &[f64; FAMILIES]| -> (Vec<usize>, f64) {
            let mut idx: Vec<usize> = (0..FAMILIES).collect();
            idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let q = idx[..TOP_K].iter().map(|&f| r.theta[f]).sum::<f64>() / TOP_K as f64;
            (idx, q)
        };
        let (raw_pick, q_raw) = pick(&r.raw_mean);
        let (fdr_pick, q_fdr) = pick(&r.fdr_mean);
        sum_q_raw += q_raw;
        sum_q_fdr += q_fdr;
        if q_fdr > q_raw {
            wins_q += 1;
        }
        if q_fdr >= q_raw {
            nonloss_q += 1;
        }

        println!(
            "seed {seed:>9}: θ[top raw {:?}] vs θ[top fdr {:?}] | Spearman raw {s_raw:+.3} fdr {s_fdr:+.3} | top-3 θ raw {q_raw:.3} fdr {q_fdr:.3}",
            raw_pick[..TOP_K]
                .iter()
                .map(|&f| (r.theta[f] * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
            fdr_pick[..TOP_K]
                .iter()
                .map(|&f| (r.theta[f] * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
        );

        // Per-seed ordering stability read (mean over bootstrap draws).
        println!(
            "           bootstrap: divisor CV mean {:.4} max {:.4} | ordering τ mean {:.4}",
            r.seed_cv_mean,
            r.boot_cv.iter().copied().fold(0.0f64, f64::max),
            r.seed_tau_mean,
        );
        all_cvs.extend_from_slice(&r.boot_cv);
        all_taus.extend_from_slice(&r.boot_taus);
    }

    let n = SEEDS.len() as f64;
    let mean_raw = sum_spear_raw / n;
    let mean_fdr = sum_spear_fdr / n;
    let mean_q_raw = sum_q_raw / n;
    let mean_q_fdr = sum_q_fdr / n;
    println!("\n=== T1 aggregate over {n} seeds ===");
    println!("Spearman vs θ: raw-P {mean_raw:+.4} | FDR {mean_fdr:+.4} | FDR wins {wins_fdr}/16");
    println!(
        "top-3 mean θ:  raw-P {mean_q_raw:.4} | FDR {mean_q_fdr:.4} | FDR wins {wins_q}/16 | non-loss {nonloss_q}/16"
    );

    // Gates.
    assert!(
        mean_fdr >= G1A_FDR_SPEARMAN_FLOOR,
        "G1a FAIL: FDR Spearman {mean_fdr:.4} < floor {G1A_FDR_SPEARMAN_FLOOR}"
    );
    assert!(
        mean_fdr - mean_raw >= G1A_MARGIN,
        "G1a FAIL: FDR margin {:.4} < {G1A_MARGIN}",
        mean_fdr - mean_raw
    );
    assert!(
        wins_fdr >= G1A_MIN_WINS,
        "G1a FAIL: FDR won only {wins_fdr}/16 seeds (need ≥ {G1A_MIN_WINS})"
    );
    assert!(
        mean_raw <= G1B_RAW_SPEARMAN_CEIL,
        "G1b FAIL: raw-P Spearman {mean_raw:.4} above premise ceiling {G1B_RAW_SPEARMAN_CEIL} — the inversion this gate exists for is not real in-harness"
    );
    assert!(
        mean_q_fdr > mean_q_raw,
        "G3 FAIL: FDR top-3 mean θ {mean_q_fdr:.4} must beat raw-P {mean_q_raw:.4}"
    );
    assert!(
        mean_q_fdr - mean_q_raw >= G3_MARGIN,
        "G3 FAIL: FDR top-3 margin {:.4} < {G3_MARGIN}",
        mean_q_fdr - mean_q_raw
    );
    assert!(
        nonloss_q >= G3_MIN_NON_LOSS,
        "G3 FAIL: FDR top-3 lost {} seeds (never-worse requires {G3_MIN_NON_LOSS}/16)",
        16 - nonloss_q
    );

    // ── T1: G2 stability aggregate (collected in the seed pass) ─────
    let cv_mean = mean(&all_cvs);
    let cv_max = all_cvs.iter().copied().fold(0.0f64, f64::max);
    let tau_mean = mean(&all_taus);
    println!("\n=== G2 bootstrap stability (B={BOOTSTRAP_B} × {n} seeds) ===");
    println!(
        "divisor CV mean {cv_mean:.4} (ceil {G2_CV_MEAN_CEIL}) | max {cv_max:.4} (ceil {G2_CV_MAX_CEIL}) | ordering τ mean {tau_mean:.4} (floor {G2_TAU_FLOOR})"
    );
    assert!(
        cv_mean <= G2_CV_MEAN_CEIL,
        "G2 FAIL: mean divisor CV {cv_mean:.4} > {G2_CV_MEAN_CEIL}"
    );
    assert!(
        cv_max <= G2_CV_MAX_CEIL,
        "G2 FAIL: max divisor CV {cv_max:.4} > {G2_CV_MAX_CEIL}"
    );
    assert!(
        tau_mean >= G2_TAU_FLOOR,
        "G2 FAIL: bootstrap ordering τ {tau_mean:.4} < {G2_TAU_FLOOR}"
    );

    // ── T2: paired read — float reward vs canonical pair ordering ────
    println!("\n=== T2 paired read (N={N_CANDIDATES} candidates) ===");
    let cands = build_corpus(SEEDS[0]);

    // Reward forms.
    let float_rewards: Vec<f32> = cands
        .iter()
        .map(|c| spr_reward(c.passed, c.passes, c.divisor))
        .collect();
    let pair_keys: Vec<SprKey> = cands
        .iter()
        .map(|c| SprKey::new(c.passed, c.passes, c.divisor))
        .collect();

    // Selections: stable sort desc, index tiebreak — deterministic both arms.
    let mut float_order: Vec<usize> = (0..N_CANDIDATES).collect();
    float_order.sort_by(|&a, &b| float_rewards[b].total_cmp(&float_rewards[a]));
    let mut pair_order: Vec<usize> = (0..N_CANDIDATES).collect();
    pair_order.sort_by(|&a, &b| pair_keys[b].cmp_key(&pair_keys[a]));

    let k = 200;
    let float_set: std::collections::HashSet<usize> = float_order[..k].iter().copied().collect();
    let pair_set: std::collections::HashSet<usize> = pair_order[..k].iter().copied().collect();
    let flips: Vec<usize> = float_set.symmetric_difference(&pair_set).copied().collect();

    // The drift guard: every flip must sit inside an exact spr_reward tie
    // group — the documented f32 rounding corner, never a real inversion.
    let mut flips_outside_ties = 0usize;
    for &c in &flips {
        // A candidate participates in a tie iff some OTHER candidate with a
        // DIFFERENT passed bit carries the exact same float reward.
        let tied_across_classes = float_rewards
            .iter()
            .enumerate()
            .any(|(j, &r)| j != c && r == float_rewards[c] && cands[j].passed != cands[c].passed);
        if !tied_across_classes {
            flips_outside_ties += 1;
        }
    }
    println!(
        "top-{k} membership flips: {} ({} outside exact-tie groups)",
        flips.len(),
        flips_outside_ties
    );

    // The law on the canonical selection: no excluded candidate is a pass
    // while an included one is a failure (strict class dominates selection).
    let law_violations = pair_set
        .iter()
        .filter(|&&i| !cands[i].passed)
        .filter(|_| pair_order[k..].iter().any(|&j| cands[j].passed))
        .count();
    println!("canonical-selection law violations: {law_violations}");

    // Eval-board diagnostic (report, not a gate).
    let spr_rows: Vec<(bool, f32)> = cands
        .iter()
        .map(|c| (c.passed, spr_reward(c.passed, c.passes, c.divisor)))
        .collect();
    let raw_rows: Vec<(bool, f32)> = cands.iter().map(|c| (c.passed, c.passes)).collect();
    let mof_spr = mass_on_failures(spr_rows.iter());
    let mof_raw = mass_on_failures(raw_rows.iter());
    println!(
        "mass_on_failures: raw-P rewards {mof_raw:.4} | SPR rewards {mof_spr:.4} (report-only diagnostic)"
    );

    assert_eq!(
        flips_outside_ties, 0,
        "T2 FAIL: {flips_outside_ties} membership flips outside exact spr_reward tie groups — \
         the float form inverted the canonical order on real (non-corner) data"
    );
    assert_eq!(
        law_violations, 0,
        "T2 FAIL: canonical top-{k} excludes a passed candidate for a failure"
    );

    // ── G4: zero-alloc scoring ───────────────────────────────────────
    println!("\n=== G4 zero-alloc scoring ===");
    let g4 = run_g4();
    println!(
        "credit() loop: {} allocs over {} calls | calibrated absorb loop: {} allocs over {} observations",
        g4.credit_allocs, g4.credit_calls, g4.absorb_allocs, g4.absorb_obs
    );
    assert_eq!(
        g4.credit_allocs, 0,
        "G4 FAIL: credit() allocated in steady state"
    );
    assert_eq!(
        g4.absorb_allocs, 0,
        "G4 FAIL: observe_hint_delta_calibrated allocated in steady state"
    );

    println!(
        "\nVERDICT: PASS — G1a/G1b/G3 ordering+selection, G2 bootstrap stability, \
         T2 paired read ({} tie-group flips), G4 zero-alloc. 16 seeds.",
        flips.len()
    );
}

// ── G4 — CountingAllocator (bench_013/bench_022 pattern) ─────────────

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

static G4_ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        G4_ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: layout is valid (caller's contract); System.alloc is sound.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr was allocated by System.alloc with this layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static G4_A: CountingAllocator = CountingAllocator;

struct G4Result {
    credit_allocs: usize,
    credit_calls: usize,
    absorb_allocs: usize,
    absorb_obs: usize,
}

fn run_g4() -> G4Result {
    use katgpt_pruners::g_zero::types::HintDelta;
    use katgpt_speculative::ScreeningPruner;

    // Freeze a calibration over 8 families.
    let mut rng = Rng::new(0x948);
    let mut ref_rows: Vec<(u64, Vec<f32>)> = Vec::with_capacity(8);
    for f in 0..8u64 {
        let mut s = Vec::with_capacity(N_REF);
        for _ in 0..N_REF {
            s.push(rng.binomial(K_SCALE[(f % FAMILIES as u64) as usize], 0.5) as f32);
        }
        ref_rows.push((f, s));
    }
    let slices: Vec<(u64, &[f32])> = ref_rows.iter().map(|(f, v)| (*f, v.as_slice())).collect();
    let cal = FdrCalibration::freeze(slices).expect("g4 freeze");

    // credit() hot loop: warm-up, then measure.
    let samples: Vec<f32> = (0..4096).map(|i| (i % 97) as f32).collect();
    let mut sink = 0.0f32;
    for &s in &samples {
        sink += cal.credit(3, s).unwrap_or(0.0);
    }
    G4_ALLOCS.store(0, Ordering::Relaxed);
    for &s in &samples {
        sink += cal.credit(3, s).unwrap_or(0.0);
    }
    let credit_allocs = G4_ALLOCS.load(Ordering::Relaxed);
    let credit_calls = samples.len();
    std::hint::black_box(sink);

    // Calibrated absorb loop: warm-up observations first so any inner Vec
    // capacities stabilize, then measure.
    struct AllowAll;
    impl ScreeningPruner for AllowAll {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }
    fn hint(value: f32) -> HintDelta {
        HintDelta {
            value,
            query: "q".into(),
            hint: "h".into(),
            a_hard: "a".into(),
            a_assisted: "b".into(),
            logp_q: -2.0,
            logp_qh: -2.0 - value,
        }
    }
    use katgpt_pruners::absorb_compress::{AbsorbCompressLayer, CompressConfig};
    use katgpt_pruners::g_zero::delta_absorb::{DeltaGatedAbsorbCompress, DeltaGatedConfig};

    let compress_config = CompressConfig::new(1, 0.05, 3, 1);
    let inner = AbsorbCompressLayer::new(AllowAll, 8, compress_config);
    let mut layer = DeltaGatedAbsorbCompress::new(inner, 8, DeltaGatedConfig::new(0.02));
    // HintDeltas pre-built — the measurement loop must not construct Strings.
    let hints: Vec<HintDelta> = samples.iter().map(|&v| hint(v)).collect();
    for h in &hints {
        layer.observe_hint_delta_calibrated(3, h, &cal);
    }
    G4_ALLOCS.store(0, Ordering::Relaxed);
    for h in &hints {
        layer.observe_hint_delta_calibrated(3, h, &cal);
    }
    let absorb_allocs = G4_ALLOCS.load(Ordering::Relaxed);
    let absorb_obs = hints.len();
    std::hint::black_box(&layer);

    G4Result {
        credit_allocs,
        credit_calls,
        absorb_allocs,
        absorb_obs,
    }
}
