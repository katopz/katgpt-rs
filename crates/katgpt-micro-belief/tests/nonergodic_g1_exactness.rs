#![cfg(feature = "nonergodic_belief")]
//! G1 exactness: brute-force posterior vs [`NonergodicFilter`] (Plan 592
//! T2.1) + G1 identities (T2.2).
//!
//! The reference posterior is computed by DEFINITION in f64 — completely
//! independent of the filter's f32 recursion: for each component, march the
//! unnormalized inner state through the sequence's transition operators and
//! take the surviving mass, `w_n ∝ μ_n·P(seq | generator n)`. Coin sequences
//! are checked EXHAUSTIVELY at L=12 (2¹² = 4096 sequences); Mess3
//! compositions are checked exhaustively at L=6..=8 (3⁶..3⁸ = 729..6561) and
//! with seeded random sequences at L=12. Agreement tolerance: 1e-5 (plan
//! T2.1).

use katgpt_micro_belief::{
    bernoulli_pair, BernoulliCoin, ComponentModel, Mess3Block, NonergodicFilter,
};

const TOL: f64 = 1e-5;

// ─── f64 reference models (independent re-derivation of the math) ──────────

#[derive(Clone, Copy)]
enum RefModel {
    /// Memoryless coin: `p` = P(token 1).
    Coin(f64),
    /// Mess3 block (3 states, tokens 0/1/2).
    Mess3 { alpha: f64, x: f64 },
}

impl RefModel {
    fn dim(&self) -> usize {
        match self {
            Self::Coin(_) => 1,
            Self::Mess3 { .. } => 3,
        }
    }

    fn init(&self) -> Vec<f64> {
        match self {
            Self::Coin(_) => vec![1.0],
            Self::Mess3 { .. } => vec![1.0 / 3.0; 3],
        }
    }

    fn advance(&self, v: &[f64], token: u8, out: &mut [f64]) {
        match *self {
            Self::Coin(p) => {
                let q = match token {
                    1 => p,
                    0 => 1.0 - p,
                    _ => 0.0,
                };
                out[0] = v[0] * q;
            }
            Self::Mess3 { alpha, x } => {
                if !(0..=2).contains(&token) {
                    for e in out.iter_mut() {
                        *e = 0.0;
                    }
                    return;
                }
                let beta = (1.0 - alpha) / 2.0;
                let y = 1.0 - 2.0 * x;
                let s = v[0] + v[1] + v[2];
                let t = token as usize;
                for j in 0..3 {
                    let cw = if j == t { alpha } else { beta };
                    out[j] = cw * (v[j] * y + (s - v[j]) * x);
                }
            }
        }
    }

    fn to_model(self) -> Box<dyn ComponentModel> {
        match self {
            Self::Coin(p) => Box::new(BernoulliCoin::new(p as f32)),
            Self::Mess3 { alpha, x } => Box::new(Mess3Block::new(alpha as f32, x as f32)),
        }
    }

    fn alphabet(&self) -> u8 {
        match self {
            Self::Coin(_) => 2,
            Self::Mess3 { .. } => 3,
        }
    }
}

/// Exact posterior by definition: `w_n ∝ μ_n · Σ(η_n^(∅) Π T_n(x_t))`.
/// `None` when the sequence is impossible under every component.
fn reference_posterior(models: &[RefModel], prior: &[f64], seq: &[u8]) -> Option<Vec<f64>> {
    let mut masses: Vec<f64> = Vec::with_capacity(models.len());
    for (n, m) in models.iter().enumerate() {
        let mut v = m.init();
        let mut out = vec![0.0; m.dim()];
        for &t in seq {
            m.advance(&v, t, &mut out);
            v.copy_from_slice(&out);
        }
        masses.push(prior[n] * v.iter().sum::<f64>());
    }
    let total: f64 = masses.iter().sum();
    if total <= 0.0 {
        return None;
    }
    Some(masses.iter().map(|m| m / total).collect())
}

// ─── shared sweep harness ───────────────────────────────────────────────────

/// Run the filter over one sequence: assert Σw = 1 after EVERY tick, compare
/// the final posterior against the f64 definition (1e-5), and assert
/// telescope block sums equal the weights.
fn check_sequence<const K: usize, const D: usize>(
    specs: &[RefModel; K],
    prior: &[f64],
    seq: &[u8],
    ctx: &str,
) {
    let boxed: Vec<Box<dyn ComponentModel>> = specs.iter().map(|s| s.to_model()).collect();
    let models: [&dyn ComponentModel; K] = std::array::from_fn(|n| boxed[n].as_ref());
    let prior_f32: [f32; K] = std::array::from_fn(|n| prior[n] as f32);
    let mut f = NonergodicFilter::<K, D>::new(models, prior_f32);
    for (i, &t) in seq.iter().enumerate() {
        f.tick(t);
        let sum: f32 = f.weights().iter().sum();
        assert!(
            (sum as f64 - 1.0).abs() <= TOL,
            "{ctx} tick {i}: Σw={sum}"
        );
    }
    if let Some(wref) = reference_posterior(specs, prior, seq) {
        for (n, &w) in wref.iter().enumerate() {
            let diff = (f.weights()[n] as f64 - w).abs();
            assert!(
                diff <= TOL,
                "{ctx}: w[{n}] filter={} ref={w} diff={diff}",
                f.weights()[n]
            );
        }
    }
    // Heap here (test code): stable Rust forbids `[f32; K * D]` — arithmetic
    // on const generics in const positions.
    let mut tele = vec![0.0f32; K * D];
    f.telescope_into(&mut tele);
    for n in 0..K {
        let block: f32 = tele[n * D..(n + 1) * D].iter().sum();
        assert!(
            (block as f64 - f.weights()[n] as f64).abs() <= TOL,
            "{ctx}: telescope block {n}={block} vs w={}",
            f.weights()[n]
        );
    }
}

/// Every token sequence of the given length over the composition's alphabet.
fn sweep_exhaustive<const K: usize, const D: usize>(
    name: &str,
    specs: &[RefModel; K],
    prior: [f64; K],
    len: usize,
) {
    let alphabet = specs[0].alphabet();
    let mut seq = vec![0u8; len];
    let mut count = 0usize;
    loop {
        check_sequence::<K, D>(specs, &prior, &seq, name);
        count += 1;
        let mut wrapped = true;
        for pos in (0..len).rev() {
            seq[pos] += 1;
            if seq[pos] < alphabet {
                wrapped = false;
                break;
            }
            seq[pos] = 0;
        }
        if wrapped {
            break;
        }
    }
    println!("{name}: {count} sequences checked (exhaustive, L={len})");
}

/// `n_seqs` seeded random sequences at the given length.
fn sweep_random<const K: usize, const D: usize>(
    name: &str,
    specs: &[RefModel; K],
    prior: [f64; K],
    len: usize,
    n_seqs: usize,
    seed: u64,
) {
    let alphabet = specs[0].alphabet();
    let mut rng = fastrand::Rng::with_seed(seed);
    for s in 0..n_seqs {
        let seq: Vec<u8> = (0..len).map(|_| rng.u8(0..alphabet)).collect();
        check_sequence::<K, D>(specs, &prior, &seq, &format!("{name}[{s}]"));
    }
    println!("{name}: {n_seqs} random sequences checked (L={len})");
}

// ─── T2.1 exhaustive + random exactness ─────────────────────────────────────

#[test]
fn exact_two_coins_blog_exhaustive_l12() {
    let specs = [RefModel::Coin(0.5), RefModel::Coin(0.7)];
    sweep_exhaustive::<2, 1>("two_coins_blog_l12", &specs, [0.5, 0.5], 12);
}

#[test]
fn exact_three_coins_exhaustive_l12() {
    let specs = [RefModel::Coin(0.25), RefModel::Coin(0.5), RefModel::Coin(0.75)];
    sweep_exhaustive::<3, 1>("three_coins_l12", &specs, [1.0 / 3.0; 3], 12);
}

#[test]
fn exact_three_coins_nonuniform_prior_l10() {
    let specs = [RefModel::Coin(0.25), RefModel::Coin(0.5), RefModel::Coin(0.75)];
    sweep_exhaustive::<3, 1>("three_coins_prior_l10", &specs, [0.5, 0.3, 0.2], 10);
}

#[test]
fn exact_two_mess3_blog_exhaustive_l8() {
    // The blog's exact composed pair: Mess3A (α=0.6, x=0.15), Mess3B (α=0.66, x=0.5).
    let specs = [
        RefModel::Mess3 { alpha: 0.6, x: 0.15 },
        RefModel::Mess3 { alpha: 0.66, x: 0.5 },
    ];
    sweep_exhaustive::<2, 3>("two_mess3_blog_l8", &specs, [0.5, 0.5], 8);
}

#[test]
fn exact_two_mess3_random_l12() {
    let specs = [
        RefModel::Mess3 { alpha: 0.6, x: 0.15 },
        RefModel::Mess3 { alpha: 0.66, x: 0.5 },
    ];
    sweep_random::<2, 3>("two_mess3_random_l12", &specs, [0.5, 0.5], 12, 1000, 592);
}

#[test]
fn exact_three_mess3_l7_and_random_l12() {
    let specs = [
        RefModel::Mess3 { alpha: 0.6, x: 0.15 },
        RefModel::Mess3 { alpha: 0.66, x: 0.5 },
        RefModel::Mess3 { alpha: 0.45, x: 0.25 },
    ];
    sweep_exhaustive::<3, 3>("three_mess3_l7", &specs, [0.4, 0.35, 0.25], 7);
    sweep_random::<3, 3>("three_mess3_random_l12", &specs, [0.4, 0.35, 0.25], 12, 1000, 1592);
}

#[test]
fn exact_five_mess3_l6_and_random_l12() {
    let specs = [
        RefModel::Mess3 { alpha: 0.6, x: 0.15 },
        RefModel::Mess3 { alpha: 0.66, x: 0.5 },
        RefModel::Mess3 { alpha: 0.45, x: 0.25 },
        RefModel::Mess3 { alpha: 0.75, x: 0.30 },
        RefModel::Mess3 { alpha: 0.55, x: 0.42 },
    ];
    sweep_exhaustive::<5, 3>("five_mess3_l6", &specs, [0.2; 5], 6);
    sweep_random::<5, 3>("five_mess3_random_l12", &specs, [0.2; 5], 12, 500, 2592);
}

// ─── T2.2 behavior identities ───────────────────────────────────────────────

/// Consistent evidence → max posterior weight rises in expectation
/// (measured: mean over 400 seeded streams at ticks 10/30/60 strictly
/// increases).
#[test]
fn collapse_monotonicity_in_expectation() {
    let coins = bernoulli_pair(0.35, 0.65);
    let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
    let mut rng = fastrand::Rng::with_seed(5926);
    let (mut m10, mut m30, mut m60) = (0.0f64, 0.0f64, 0.0f64);
    let streams = 400usize;
    for _ in 0..streams {
        let mut f = NonergodicFilter::<2, 1>::new(models, [0.5, 0.5]);
        for t in 0..60 {
            let token = if rng.f32() < 0.65 { 1u8 } else { 0u8 };
            f.tick(token);
            let (_, unc) = f.committed();
            match t {
                9 => m10 += 1.0 - unc as f64,
                29 => m30 += 1.0 - unc as f64,
                59 => m60 += 1.0 - unc as f64,
                _ => {}
            }
        }
    }
    let (m10, m30, m60) = (m10 / streams as f64, m30 / streams as f64, m60 / streams as f64);
    println!("mean max w @10={m10:.4} @30={m30:.4} @60={m60:.4}");
    assert!(
        m10 < m30 && m30 < m60,
        "collapse monotonicity violated: {m10} !< {m30} !< {m60}"
    );
}

/// A collapsed hypothesis REVIVES when the stream switches to the other
/// generator — the capability single-belief kernels cannot represent
/// (Research 545 §2).
#[test]
fn revival_reinflates_collapsed_hypothesis() {
    let coins = bernoulli_pair(0.15, 0.85);
    let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
    let mut f = NonergodicFilter::<2, 1>::new(models, [0.5, 0.5]);
    // 40 tails: coin 0 (P(tails)=0.85) explains best → collapse onto coin 0.
    for _ in 0..40 {
        f.tick(0);
    }
    let (idx, unc) = f.committed();
    assert_eq!(idx, 0);
    assert!(unc < 1e-3, "not collapsed: unc={unc}");
    let collapsed_floor = f.weights()[1];
    // Contradiction: 45 heads — coin 1 (P(heads)=0.85) explains best. The
    // 40 tails built a +69-nat commitment; each head shifts ln(0.85/0.15)
    // = 1.73 nats toward coin 1, so 45 heads swing the net to ≈ −9 nats.
    for _ in 0..45 {
        f.tick(1);
    }
    let (idx2, unc2) = f.committed();
    assert_eq!(idx2, 1, "no revival after contradiction stream");
    assert!(unc2 < 1e-2, "revived but not confident: unc2={unc2}");
    assert!(
        f.weights()[1] > 0.99,
        "revived weight did not re-inflate above the collapse floor ({collapsed_floor})"
    );
}
