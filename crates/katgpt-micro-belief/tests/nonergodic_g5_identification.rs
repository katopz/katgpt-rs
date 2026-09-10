#![cfg(feature = "nonergodic_belief")]
//! T2.5 identification bench (Plan 592): can the two-level posterior identify
//! WHICH generator produced a stream — and at what quality — versus (b) a
//! hand-tuned single-belief leaky integrator and (c) a BoM K-particle +
//! hard `select_best` baseline?
//!
//! Streams: K Mess3 generators, one chosen per stream, tokens sampled from
//! the TRUE generator's predictive distribution (exact HMM sampling, f64).
//! Metrics per predictor: accuracy, mean log-loss (nats), calibration error
//! (ECE, 10 bins).
//!
//! Predictors:
//! - **(a) NonergodicFilter** — posterior `w` over generators.
//! - **(b) leaky integrator** — same per-generator inner trackers, but the
//!   weight update is `s_n ← γ·s_n + ln ℓ_n` with a hand-tuned decay γ
//!   (the constant Research 545 says the principled posterior avoids).
//!   Reported at the BEST of γ ∈ {0.8, 0.9, 0.95} (fair tuning).
//! - **(c) BoM K + select_best** — unnormalized log-likelihood accumulation
//!   (the one-shot/tilt selection semantics of `bom.rs::select_best`):
//!   hard one-hot prediction on the argmax component (1e-12 floor).
//!
//! Honest gates (the plan's demotion rule, Research 545 §5): (a) must beat
//! (b) on log-loss at K = 2..4, and (a) must calibrate better than (c)
//! (ECE) — hard selection cannot express "not sure" and pays unboundedly
//! when wrong. (a) vs (c) on raw log-loss is RECORDED, not gated: when both
//! are always right, an overconfident one-hot scores an artificially
//! perfect 0 — the ECE axis is the honest discriminator.

use katgpt_micro_belief::{ComponentModel, Mess3Block, NonergodicFilter};

const STREAMS: usize = 600;
const STREAM_LEN: usize = 24;
const SEED: u64 = 592;
const GAMMAS: [f32; 3] = [0.8, 0.9, 0.95];

fn mess3_params(k: usize) -> Vec<(f64, f64)> {
    match k {
        2 => vec![(0.60, 0.15), (0.66, 0.50)],
        3 => vec![(0.60, 0.15), (0.66, 0.50), (0.45, 0.25)],
        4 => vec![(0.60, 0.15), (0.66, 0.50), (0.45, 0.25), (0.75, 0.30)],
        _ => unreachable!("only K=2..4"),
    }
}

/// f64 generative Mess3 (exact sampling of the TRUE generator).
struct Gen {
    alpha: f64,
    x: f64,
}

impl Gen {
    fn beta(&self) -> f64 {
        (1.0 - self.alpha) / 2.0
    }

    fn y(&self) -> f64 {
        1.0 - 2.0 * self.x
    }

    fn init(&self) -> [f64; 3] {
        [1.0 / 3.0; 3]
    }

    /// Row mass of T^(t) for state i (= P(token t) weight for state i).
    fn row_sum(&self, i: usize, t: usize) -> f64 {
        let cw = if i == t { self.alpha } else { self.beta() };
        self.x + cw * (self.y() - self.x)
    }

    fn advance(&self, v: &[f64; 3], t: usize, out: &mut [f64; 3]) {
        let s = v[0] + v[1] + v[2];
        for j in 0..3 {
            let cw = if j == t { self.alpha } else { self.beta() };
            out[j] = cw * (v[j] * self.y() + (s - v[j]) * self.x);
        }
    }
}

/// Sample one stream of length STREAM_LEN from generator `g`.
fn sample_stream(g: &Gen, rng: &mut fastrand::Rng) -> Vec<u8> {
    let mut v = g.init();
    let mut seq = Vec::with_capacity(STREAM_LEN);
    for _ in 0..STREAM_LEN {
        let u = rng.f64();
        let mut cum = 0.0f64;
        let mut tok = 2usize;
        for t in 0..3 {
            let mut px = 0.0f64;
            for (i, &vi) in v.iter().enumerate() {
                px += vi * g.row_sum(i, t);
            }
            cum += px;
            if u < cum {
                tok = t;
                break;
            }
        }
        seq.push(tok as u8);
        let mut nxt = [0.0f64; 3];
        g.advance(&v, tok, &mut nxt);
        let s: f64 = nxt.iter().sum();
        for j in 0..3 {
            v[j] = nxt[j] / s;
        }
    }
    seq
}

/// Accumulates accuracy / log-loss / calibration for one predictor.
struct Metrics {
    correct: usize,
    nll: f64,
    bins: [(f64, f64, usize); 10],
}

impl Metrics {
    fn new() -> Self {
        Self {
            correct: 0,
            nll: 0.0,
            bins: [(0.0, 0.0, 0); 10],
        }
    }

    fn observe(&mut self, pred: &[f64], truth: usize) {
        let mut am = 0usize;
        let mut bw = f64::NEG_INFINITY;
        for (n, &p) in pred.iter().enumerate() {
            if p > bw {
                bw = p;
                am = n;
            }
        }
        let hit = am == truth;
        if hit {
            self.correct += 1;
        }
        self.nll -= pred[truth].max(1e-12).ln();
        let b = ((bw * 10.0) as usize).min(9);
        let (cs, ks, c) = self.bins[b];
        self.bins[b] = (cs + bw, ks + if hit { 1.0 } else { 0.0 }, c + 1);
    }

    fn accuracy(&self, n: usize) -> f64 {
        self.correct as f64 / n as f64
    }

    fn log_loss(&self, n: usize) -> f64 {
        self.nll / n as f64
    }

    fn ece(&self, n: usize) -> f64 {
        let mut e = 0.0f64;
        for &(cs, ks, c) in &self.bins {
            if c > 0 {
                e += (c as f64 / n as f64) * ((ks / c as f64) - (cs / c as f64)).abs();
            }
        }
        e
    }
}

/// LSE-normalize a f32 log-score slice into probabilities (f64 out).
fn normalize_logs(logs: &[f32], k: usize) -> Vec<f64> {
    let mut m = f32::NEG_INFINITY;
    for v in logs.iter().take(k) {
        if *v > m {
            m = *v;
        }
    }
    let mut s = 0.0f64;
    let mut ex = [0.0f64; 4];
    for (n, &l) in logs.iter().take(k).enumerate() {
        ex[n] = (l - m).exp() as f64;
        s += ex[n];
    }
    (0..k).map(|n| ex[n] / s).collect()
}

type Suite = ([f64; 3], [f64; 3], [f64; 3], f64, [f64; 3]);

/// Run the full stream suite at a given K. Returns
/// `(acc[a, b, c], nll[a, b, c], ece[a, b, c], best_gamma, acc_per_leaky_gamma)`.
fn run_suite<const K: usize>(k_label: usize) -> Suite {
    let params = mess3_params(k_label);
    let gens: Vec<Gen> = params
        .iter()
        .map(|&(a, x)| Gen { alpha: a, x })
        .collect();
    let models: Vec<Mess3Block> = params
        .iter()
        .map(|&(a, x)| Mess3Block::new(a as f32, x as f32))
        .collect();
    let model_refs: [&dyn ComponentModel; K] =
        std::array::from_fn(|n| &models[n] as &dyn ComponentModel);

    let mut rng = fastrand::Rng::with_seed(SEED);
    let mut m_filter = Metrics::new();
    let mut m_bom = Metrics::new();
    let mut m_leaky = [Metrics::new(), Metrics::new(), Metrics::new()];
    let mut leaky_acc = [0.0f64; 3];

    for s in 0..STREAMS {
        let truth = s % K;
        let seq = sample_stream(&gens[truth], &mut rng);

        // (a) NonergodicFilter.
        let mut f = NonergodicFilter::<K, 3>::new(model_refs, [1.0f32 / K as f32; K]);
        for &t in &seq {
            f.tick(t);
        }
        let wa: Vec<f64> = f.weights().iter().map(|&x| x as f64).collect();
        m_filter.observe(&wa, truth);

        // (b) leaky × 3 + (c) BoM hard-select: manual per-generator trackers.
        // g index 0..2 = leaky at GAMMAS[g]; g = 3 = BoM accumulator.
        let mut eta: [[[f32; 3]; K]; 4] = [[[1.0f32 / 3.0; 3]; K]; 4];
        let mut log_b: [[f32; K]; 3] = [[0.0f32; K]; 3];
        let mut acc_c = [0.0f32; K];
        for &t in &seq {
            for n in 0..K {
                for g in 0..4 {
                    let l = models[n].likelihood(&eta[g][n], t);
                    if l > 0.0 && l.is_finite() {
                        let mut nxt = [0.0f32; 3];
                        models[n].update_into(&eta[g][n], t, &mut nxt);
                        let total: f32 = nxt.iter().sum();
                        if total > 0.0 && total.is_finite() {
                            let inv = 1.0 / total;
                            for v in nxt.iter_mut() {
                                *v *= inv;
                            }
                            eta[g][n] = nxt;
                        }
                        match g {
                            0..=2 => log_b[g][n] = GAMMAS[g] * log_b[g][n] + l.ln(),
                            _ => acc_c[n] += l.ln(),
                        }
                    } else {
                        match g {
                            0..=2 => log_b[g][n] = f32::NEG_INFINITY,
                            _ => acc_c[n] = f32::NEG_INFINITY,
                        }
                    }
                }
            }
        }
        for g in 0..3 {
            let pred = normalize_logs(&log_b[g], K);
            m_leaky[g].observe(&pred, truth);
            // track leaky accuracy too (argmax of log scores == argmax pred)
            let mut am = 0usize;
            let mut bw = f64::NEG_INFINITY;
            for (n, &p) in pred.iter().enumerate() {
                if p > bw {
                    bw = p;
                    am = n;
                }
            }
            if am == truth {
                leaky_acc[g] += 1.0;
            }
        }
        let mut am = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (n, &score) in acc_c.iter().enumerate().take(K) {
            if score > best {
                best = score;
                am = n;
            }
        }
        let mut pred_c = vec![1e-12f64; K];
        pred_c[am] = 1.0;
        m_bom.observe(&pred_c, truth);
    }

    // Best-γ leaky by log-loss (fair tuning of the baseline).
    let mut best_g = 0usize;
    for g in 1..3 {
        if m_leaky[g].log_loss(STREAMS) < m_leaky[best_g].log_loss(STREAMS) {
            best_g = g;
        }
    }
    for a in leaky_acc.iter_mut() {
        *a /= STREAMS as f64;
    }

    (
        [
            m_filter.accuracy(STREAMS),
            m_leaky[best_g].accuracy(STREAMS),
            m_bom.accuracy(STREAMS),
        ],
        [
            m_filter.log_loss(STREAMS),
            m_leaky[best_g].log_loss(STREAMS),
            m_bom.log_loss(STREAMS),
        ],
        [
            m_filter.ece(STREAMS),
            m_leaky[best_g].ece(STREAMS),
            m_bom.ece(STREAMS),
        ],
        GAMMAS[best_g] as f64,
        leaky_acc,
    )
}

#[test]
fn identification_bench_t2_5() {
    println!("T2.5 identification — {STREAMS} streams × L={STREAM_LEN}, seed {SEED}");
    println!("(a) NonergodicFilter  (b) leaky best-γ  (c) BoM-K hard select_best");
    for k in [2usize, 3, 4] {
        let (acc, nll, ece, best_gamma, leaky_acc) = match k {
            2 => run_suite::<2>(2),
            3 => run_suite::<3>(3),
            _ => run_suite::<4>(4),
        };
        println!("\nK={k}  (leaky best γ={best_gamma}, γ-accuracies {leaky_acc:?}):");
        println!("  predictor          accuracy   log-loss(nats)   ECE(10-bin)");
        println!(
            "  (a) filter         {:.4}     {:.6}        {:.6}",
            acc[0], nll[0], ece[0]
        );
        println!(
            "  (b) leaky γ-tuned  {:.4}     {:.6}        {:.6}",
            acc[1], nll[1], ece[1]
        );
        println!(
            "  (c) BoM hard-sel   {:.4}     {:.6}        {:.6}",
            acc[2], nll[2], ece[2]
        );

        // Honest gates: (a) beats the TUNED leaky baseline on log-loss, and
        // (a) calibrates better than hard selection (ECE). The (a)-vs-(c)
        // log-loss comparison is printed above and recorded in the bench
        // doc — NOT asserted (one-hot scores an artificial 0 when never
        // wrong; ECE is the honest discriminator).
        assert!(
            nll[0] < nll[1],
            "K={k}: filter log-loss {} >= tuned-leaky log-loss {}",
            nll[0],
            nll[1]
        );
        assert!(
            ece[0] < ece[2],
            "K={k}: filter ECE {} >= BoM hard-select ECE {}",
            ece[0],
            ece[2]
        );
    }
}
